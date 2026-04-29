from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np

from nf_agent.data.rref_state_shards import RREFStateActionSamples, load_rref_state_shard
from nf_agent.env.elementary_ops import Matrix, normalize_matrix
from nf_agent.env.rref_modp import RowOp, is_rref_modp, replay_row_ops
from nf_agent.models import RREFMatrixFormer
from nf_agent.train import (
    RREFMatrixFormerTrainConfig,
    restore_latest_rref_matrixformer_checkpoint,
)

RolloutStatus: TypeAlias = Literal["success", "max_steps_exceeded"]
LogitsProvider: TypeAlias = Callable[[Matrix, int], Mapping[str, Any]]
BatchLogitsProvider: TypeAlias = Callable[[Sequence[Matrix], int], Sequence[Mapping[str, Any]]]

_BREAKDOWN_KEYS = ("action_kind", "action_target", "action_source", "action_scalar")
PADDING_VALUE = -1


@dataclass(frozen=True)
class RREFMatrixFormerRolloutConfig:
    data_path: str | Path
    checkpoint_dir: str | Path
    sample_index: int | None = None
    max_steps: int | None = None
    row_embedding_dim: int = 32
    col_embedding_dim: int = 32
    hidden_dim: int = 256
    layers: int = 2
    num_heads: int = 4


@dataclass(frozen=True)
class RREFMatrixFormerRolloutResult:
    status: RolloutStatus
    success: bool
    step_count: int
    invalid_action_count: int
    masked_action_count: int
    invalid_action_breakdown: dict[str, int]
    initial_matrix: Matrix
    final_matrix: Matrix
    ops: list[RowOp]
    final_is_rref: bool
    checkpoint_step: int | None
    modulus: int

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "step_count": self.step_count,
            "invalid_action_count": self.invalid_action_count,
            "masked_action_count": self.masked_action_count,
            "invalid_action_breakdown": dict(self.invalid_action_breakdown),
            "initial_matrix": self.initial_matrix,
            "final_matrix": self.final_matrix,
            "ops": [_row_op_to_dict(op) for op in self.ops],
            "final_is_rref": self.final_is_rref,
            "checkpoint_step": self.checkpoint_step,
            "modulus": self.modulus,
        }


@dataclass(frozen=True)
class RREFVerifierBeamConfig:
    data_path: str | Path
    checkpoint_dir: str | Path
    sample_index: int | None = None
    max_steps: int | None = None
    beam_width: int = 8
    batch_size: int | Literal["auto"] = "auto"
    row_embedding_dim: int = 32
    col_embedding_dim: int = 32
    hidden_dim: int = 256
    layers: int = 2
    num_heads: int = 4


@dataclass(frozen=True)
class RREFVerifierBeamResult:
    status: RolloutStatus
    success: bool
    step_count: int
    invalid_action_count: int
    masked_action_count: int
    invalid_action_breakdown: dict[str, int]
    initial_matrix: Matrix
    final_matrix: Matrix
    ops: list[RowOp]
    final_is_rref: bool
    checkpoint_step: int | None
    modulus: int
    beam_width: int
    score: float
    replay_ok: bool
    expanded_count: int
    pruned_count: int
    device_batch_size: int

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "step_count": self.step_count,
            "invalid_action_count": self.invalid_action_count,
            "masked_action_count": self.masked_action_count,
            "invalid_action_breakdown": dict(self.invalid_action_breakdown),
            "initial_matrix": self.initial_matrix,
            "final_matrix": self.final_matrix,
            "ops": [_row_op_to_dict(op) for op in self.ops],
            "final_is_rref": self.final_is_rref,
            "checkpoint_step": self.checkpoint_step,
            "modulus": self.modulus,
            "beam_width": self.beam_width,
            "score": self.score,
            "replay_ok": self.replay_ok,
            "expanded_count": self.expanded_count,
            "pruned_count": self.pruned_count,
            "device_batch_size": self.device_batch_size,
        }


@dataclass(frozen=True)
class _BeamEntry:
    matrix: Matrix
    ops: tuple[RowOp, ...]
    score: float
    action_tuple: tuple[int, int, int, int]


def _row_op_to_dict(op: RowOp) -> dict[str, int | str]:
    payload: dict[str, int | str] = {"kind": op.kind, "target": op.target}
    if op.source is not None:
        payload["source"] = op.source
    if op.scalar is not None:
        payload["scalar"] = op.scalar
    return payload


def _new_breakdown() -> dict[str, int]:
    return {key: 0 for key in _BREAKDOWN_KEYS}


def _as_logits(value: object, name: str, expected_size: int) -> np.ndarray:
    logits = np.asarray(value, dtype=np.float32)
    expected_shape = (expected_size,)
    if logits.shape != expected_shape:
        raise ValueError(f"{name} logits must have shape {expected_shape}, got {logits.shape}")
    return logits


def _argmax(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    log_total = float(np.log(np.exp(shifted).sum()))
    return cast(np.ndarray, shifted - log_total)


def _topk_indices(scores: np.ndarray, legal_indices: Iterable[int], k: int) -> list[int]:
    ranked = sorted(legal_indices, key=lambda index: (-float(scores[index]), int(index)))
    return [int(index) for index in ranked[:k]]


def _masked_argmax(logits: np.ndarray, legal_indices: Sequence[int]) -> int:
    if not legal_indices:
        raise ValueError("at least one legal action is required")
    masked = np.full_like(logits, -np.inf, dtype=np.float32)
    masked[list(legal_indices)] = logits[list(legal_indices)]
    return _argmax(masked)


def _legal_kind_indices(current: Matrix, p: int) -> list[int]:
    rows = len(current)
    if is_rref_modp(current, p):
        return [0]
    if rows >= 2:
        return [1, 2, 3]
    if rows == 1:
        return [2]
    return []


def _select_kind(
    logits: Mapping[str, Any],
    current: Matrix,
    p: int,
    breakdown: dict[str, int],
) -> int:
    kind_logits = _as_logits(logits["action_kind_logits"], "action_kind", 4)
    proposed = _argmax(kind_logits)
    legal_kinds = _legal_kind_indices(current, p)
    if proposed in legal_kinds:
        return proposed
    breakdown["action_kind"] += 1
    return _masked_argmax(kind_logits, legal_kinds)


def _select_target(logits: Mapping[str, Any], rows: int) -> int:
    target_logits = _as_logits(logits["action_target_logits"], "action_target", rows)
    return _argmax(target_logits)


def _select_source(
    logits: Mapping[str, Any],
    rows: int,
    target: int,
    breakdown: dict[str, int],
) -> int:
    source_logits = _as_logits(logits["action_source_logits"], "action_source", rows)
    proposed = _argmax(source_logits)
    if proposed != target:
        return proposed
    breakdown["action_source"] += 1
    return _masked_argmax(source_logits, [row for row in range(rows) if row != target])


def _select_nonzero_scalar(
    logits: Mapping[str, Any],
    p: int,
    breakdown: dict[str, int],
) -> int:
    scalar_logits = _as_logits(logits["action_scalar_logits"], "action_scalar", p)
    proposed = _argmax(scalar_logits) % p
    if proposed != 0:
        return proposed
    breakdown["action_scalar"] += 1
    return _masked_argmax(scalar_logits, range(1, p))


def _decode_legal_row_op(
    current: Matrix,
    p: int,
    logits: Mapping[str, Any],
    breakdown: dict[str, int],
) -> RowOp | None:
    rows = len(current)
    if rows == 0:
        raise ValueError("cannot decode a row operation for a matrix with no rows")

    kind = _select_kind(logits, current, p, breakdown)
    if kind == 0:
        return None
    target = _select_target(logits, rows)

    if kind == 1:
        source = _select_source(logits, rows, target, breakdown)
        return RowOp.swap(target, source)
    if kind == 2:
        scalar = _select_nonzero_scalar(logits, p, breakdown)
        return RowOp.scale(target, scalar)
    if kind == 3:
        source = _select_source(logits, rows, target, breakdown)
        scalar = _select_nonzero_scalar(logits, p, breakdown)
        return RowOp.add(target, source, scalar)
    raise ValueError(f"unknown decoded row operation kind: {kind}")


def _finish_result(
    *,
    status: RolloutStatus,
    initial: Matrix,
    current: Matrix,
    ops: list[RowOp],
    breakdown: dict[str, int],
    checkpoint_step: int | None,
    modulus: int,
) -> RREFMatrixFormerRolloutResult:
    final_is_rref = is_rref_modp(current, modulus)
    invalid_count = sum(breakdown.values())
    return RREFMatrixFormerRolloutResult(
        status=status,
        success=status == "success",
        step_count=len(ops),
        invalid_action_count=invalid_count,
        masked_action_count=invalid_count,
        invalid_action_breakdown=dict(breakdown),
        initial_matrix=initial,
        final_matrix=current,
        ops=list(ops),
        final_is_rref=final_is_rref,
        checkpoint_step=checkpoint_step,
        modulus=modulus,
    )


def rollout_rref_matrixformer_with_logits(
    matrix: Sequence[Sequence[int]],
    *,
    modulus: int,
    max_steps: int,
    logits_provider: LogitsProvider,
    checkpoint_step: int | None = None,
) -> RREFMatrixFormerRolloutResult:
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")

    initial = normalize_matrix(matrix, modulus)
    current = [row[:] for row in initial]
    ops: list[RowOp] = []
    breakdown = _new_breakdown()

    for step_index in range(max_steps):
        if is_rref_modp(current, modulus):
            return _finish_result(
                status="success",
                initial=initial,
                current=current,
                ops=ops,
                breakdown=breakdown,
                checkpoint_step=checkpoint_step,
                modulus=modulus,
            )
        logits = logits_provider(current, step_index)
        op = _decode_legal_row_op(current, modulus, logits, breakdown)
        if op is None:
            return _finish_result(
                status="success",
                initial=initial,
                current=current,
                ops=ops,
                breakdown=breakdown,
                checkpoint_step=checkpoint_step,
                modulus=modulus,
            )
        current = replay_row_ops(current, [op], modulus)
        ops.append(op)

    final_status: RolloutStatus = (
        "success" if is_rref_modp(current, modulus) else "max_steps_exceeded"
    )
    return _finish_result(
        status=final_status,
        initial=initial,
        current=current,
        ops=ops,
        breakdown=breakdown,
        checkpoint_step=checkpoint_step,
        modulus=modulus,
    )


def _matrix_key(matrix: Matrix) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in matrix)


def _beam_replay_ok(initial: Matrix, ops: Sequence[RowOp], final: Matrix, p: int) -> bool:
    try:
        return replay_row_ops(initial, ops, p) == final
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return False


def _finish_beam_result(
    *,
    status: RolloutStatus,
    initial: Matrix,
    entry: _BeamEntry,
    checkpoint_step: int | None,
    modulus: int,
    beam_width: int,
    expanded_count: int,
    pruned_count: int,
    device_batch_size: int,
) -> RREFVerifierBeamResult:
    ops = list(entry.ops)
    return RREFVerifierBeamResult(
        status=status,
        success=status == "success",
        step_count=len(ops),
        invalid_action_count=0,
        masked_action_count=0,
        invalid_action_breakdown=_new_breakdown(),
        initial_matrix=initial,
        final_matrix=entry.matrix,
        ops=ops,
        final_is_rref=is_rref_modp(entry.matrix, modulus),
        checkpoint_step=checkpoint_step,
        modulus=modulus,
        beam_width=beam_width,
        score=entry.score,
        replay_ok=_beam_replay_ok(initial, ops, entry.matrix, modulus),
        expanded_count=expanded_count,
        pruned_count=pruned_count,
        device_batch_size=device_batch_size,
    )


def _best_finished(beam: Sequence[_BeamEntry], p: int) -> _BeamEntry | None:
    finished = [entry for entry in beam if is_rref_modp(entry.matrix, p)]
    if not finished:
        return None
    return sorted(finished, key=lambda entry: (-entry.score, entry.action_tuple))[0]


def _candidate_row_ops(
    current: Matrix,
    p: int,
    logits: Mapping[str, Any],
    beam_width: int,
) -> list[tuple[RowOp | None, tuple[int, int, int, int], float]]:
    rows = len(current)
    kind_scores = _log_softmax(_as_logits(logits["action_kind_logits"], "action_kind", 4))
    target_scores = _log_softmax(
        _as_logits(logits["action_target_logits"], "action_target", rows)
    )
    source_scores = _log_softmax(
        _as_logits(logits["action_source_logits"], "action_source", rows)
    )
    scalar_scores = _log_softmax(_as_logits(logits["action_scalar_logits"], "action_scalar", p))

    candidates: list[tuple[RowOp | None, tuple[int, int, int, int], float]] = []
    for kind in _topk_indices(kind_scores, _legal_kind_indices(current, p), beam_width):
        kind_score = float(kind_scores[kind])
        if kind == 0:
            candidates.append(
                (
                    None,
                    (0, PADDING_VALUE, PADDING_VALUE, PADDING_VALUE),
                    kind_score,
                )
            )
            continue

        for target in _topk_indices(target_scores, range(rows), beam_width):
            target_score = kind_score + float(target_scores[target])
            if kind == 1:
                for source in _topk_indices(
                    source_scores,
                    (row for row in range(rows) if row != target),
                    beam_width,
                ):
                    candidates.append(
                        (
                            RowOp.swap(target, source),
                            (1, target, source, PADDING_VALUE),
                            target_score + float(source_scores[source]),
                        )
                    )
            elif kind == 2:
                for scalar in _topk_indices(scalar_scores, range(1, p), beam_width):
                    candidates.append(
                        (
                            RowOp.scale(target, scalar),
                            (2, target, PADDING_VALUE, scalar),
                            target_score + float(scalar_scores[scalar]),
                        )
                    )
            elif kind == 3:
                sources = _topk_indices(
                    source_scores,
                    (row for row in range(rows) if row != target),
                    beam_width,
                )
                scalars = _topk_indices(scalar_scores, range(1, p), beam_width)
                for source in sources:
                    source_score = target_score + float(source_scores[source])
                    for scalar in scalars:
                        candidates.append(
                            (
                                RowOp.add(target, source, scalar),
                                (3, target, source, scalar),
                                source_score + float(scalar_scores[scalar]),
                            )
                        )
    candidates.sort(key=lambda item: (-item[2], item[1]))
    return candidates[:beam_width]


def rollout_rref_verifier_beam_with_logits(
    matrix: Sequence[Sequence[int]],
    *,
    modulus: int,
    max_steps: int,
    beam_width: int,
    logits_provider: BatchLogitsProvider,
    checkpoint_step: int | None = None,
) -> RREFVerifierBeamResult:
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")

    initial = normalize_matrix(matrix, modulus)
    beam = [
        _BeamEntry(
            matrix=[row[:] for row in initial],
            ops=(),
            score=0.0,
            action_tuple=(0, PADDING_VALUE, PADDING_VALUE, PADDING_VALUE),
        )
    ]
    expanded_count = 0
    pruned_count = 0
    max_device_batch_size = 0

    for step_index in range(max_steps + 1):
        finished = _best_finished(beam, modulus)
        if finished is not None:
            return _finish_beam_result(
                status="success",
                initial=initial,
                entry=finished,
                checkpoint_step=checkpoint_step,
                modulus=modulus,
                beam_width=beam_width,
                expanded_count=expanded_count,
                pruned_count=pruned_count,
                device_batch_size=max(1, max_device_batch_size),
            )
        if step_index == max_steps:
            break

        max_device_batch_size = max(max_device_batch_size, len(beam))
        batch_logits = list(logits_provider([entry.matrix for entry in beam], step_index))
        if len(batch_logits) != len(beam):
            raise ValueError("logits_provider must return one logits mapping per beam entry")

        dedup: dict[tuple[tuple[int, ...], ...], _BeamEntry] = {}
        generated_this_step = 0
        for entry, logits in zip(beam, batch_logits, strict=True):
            for op, action_tuple, action_score in _candidate_row_ops(
                entry.matrix,
                modulus,
                logits,
                beam_width,
            ):
                generated_this_step += 1
                if op is None:
                    next_matrix = [row[:] for row in entry.matrix]
                    next_ops = entry.ops
                else:
                    next_matrix = replay_row_ops(entry.matrix, [op], modulus)
                    next_ops = (*entry.ops, op)
                candidate = _BeamEntry(
                    matrix=next_matrix,
                    ops=next_ops,
                    score=entry.score + action_score,
                    action_tuple=action_tuple,
                )
                key = _matrix_key(next_matrix)
                old = dedup.get(key)
                if old is None or candidate.score > old.score or (
                    candidate.score == old.score and candidate.action_tuple < old.action_tuple
                ):
                    dedup[key] = candidate

        expanded_count += generated_this_step
        ranked = sorted(dedup.values(), key=lambda entry: (-entry.score, entry.action_tuple))
        beam = ranked[:beam_width]
        pruned_count += max(0, generated_this_step - len(beam))
        if not beam:
            break

    best = sorted(beam, key=lambda entry: (-entry.score, entry.action_tuple))[0]
    final_status: RolloutStatus = (
        "success" if is_rref_modp(best.matrix, modulus) else "max_steps_exceeded"
    )
    return _finish_beam_result(
        status=final_status,
        initial=initial,
        entry=best,
        checkpoint_step=checkpoint_step,
        modulus=modulus,
        beam_width=beam_width,
        expanded_count=expanded_count,
        pruned_count=pruned_count,
        device_batch_size=max(1, max_device_batch_size),
    )


def _resolve_max_steps(config: RREFMatrixFormerRolloutConfig, metadata: Mapping[str, Any]) -> int:
    max_steps = int(metadata["max_ops"]) if config.max_steps is None else config.max_steps
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    return max_steps


def _checkpoint_step_from_state_step(step: Any) -> int:
    return int(np.asarray(jax.device_get(step)))


def rollout_rref_matrixformer(
    config: RREFMatrixFormerRolloutConfig,
    matrix: Sequence[Sequence[int]],
) -> RREFMatrixFormerRolloutResult:
    samples = RREFStateActionSamples(config.data_path)
    current_matrix = normalize_matrix(matrix, samples.modulus)
    if len(current_matrix) != samples.rows:
        raise ValueError(f"matrix must have {samples.rows} rows, got {len(current_matrix)}")
    if current_matrix and len(current_matrix[0]) != samples.cols:
        raise ValueError(f"matrix must have {samples.cols} cols, got {len(current_matrix[0])}")
    _arrays, metadata = load_rref_state_shard(config.data_path)
    max_steps = _resolve_max_steps(config, metadata)

    train_config = RREFMatrixFormerTrainConfig(
        data_path=config.data_path,
        steps=1,
        batch_size=1,
        out_dir=config.checkpoint_dir,
        row_embedding_dim=config.row_embedding_dim,
        col_embedding_dim=config.col_embedding_dim,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        num_heads=config.num_heads,
    )
    if not Path(config.checkpoint_dir).exists():
        raise ValueError(f"no checkpoint found in {Path(config.checkpoint_dir)}")
    state = restore_latest_rref_matrixformer_checkpoint(train_config)
    checkpoint_step = _checkpoint_step_from_state_step(state.step)
    model = RREFMatrixFormer(
        rows=samples.rows,
        cols=samples.cols,
        modulus=samples.modulus,
        row_embedding_dim=config.row_embedding_dim,
        col_embedding_dim=config.col_embedding_dim,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        num_heads=config.num_heads,
    )

    def logits_provider(current: Matrix, _step_index: int) -> Mapping[str, Any]:
        inputs = np.asarray(current, dtype=np.float32)
        inputs %= float(samples.modulus)
        inputs /= float(samples.modulus - 1)
        outputs = cast(
            dict[str, Any],
            model.apply({"params": state.params}, jnp.asarray(inputs[np.newaxis, ...])),
        )
        return {
            "action_kind_logits": np.asarray(jax.device_get(outputs["action_kind_logits"][0])),
            "action_target_logits": np.asarray(
                jax.device_get(outputs["action_target_logits"][0])
            ),
            "action_source_logits": np.asarray(
                jax.device_get(outputs["action_source_logits"][0])
            ),
            "action_scalar_logits": np.asarray(
                jax.device_get(outputs["action_scalar_logits"][0])
            ),
        }

    return rollout_rref_matrixformer_with_logits(
        current_matrix,
        modulus=samples.modulus,
        max_steps=max_steps,
        logits_provider=logits_provider,
        checkpoint_step=checkpoint_step,
    )


def _resolve_beam_max_steps(config: RREFVerifierBeamConfig, metadata: Mapping[str, Any]) -> int:
    max_steps = int(metadata["max_ops"]) if config.max_steps is None else config.max_steps
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    return max_steps


def _resolve_batch_size(batch_size: int | Literal["auto"], beam_width: int) -> int:
    if batch_size == "auto":
        return max(1, beam_width)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive or 'auto'")
    return batch_size


def rollout_rref_verifier_beam(
    config: RREFVerifierBeamConfig,
    matrix: Sequence[Sequence[int]],
) -> RREFVerifierBeamResult:
    if config.beam_width <= 0:
        raise ValueError("beam_width must be positive")
    samples = RREFStateActionSamples(config.data_path)
    current_matrix = normalize_matrix(matrix, samples.modulus)
    if len(current_matrix) != samples.rows:
        raise ValueError(f"matrix must have {samples.rows} rows, got {len(current_matrix)}")
    if current_matrix and len(current_matrix[0]) != samples.cols:
        raise ValueError(f"matrix must have {samples.cols} cols, got {len(current_matrix[0])}")
    _arrays, metadata = load_rref_state_shard(config.data_path)
    max_steps = _resolve_beam_max_steps(config, metadata)
    resolved_batch_size = _resolve_batch_size(config.batch_size, config.beam_width)

    train_config = RREFMatrixFormerTrainConfig(
        data_path=config.data_path,
        steps=1,
        batch_size=1,
        out_dir=config.checkpoint_dir,
        row_embedding_dim=config.row_embedding_dim,
        col_embedding_dim=config.col_embedding_dim,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        num_heads=config.num_heads,
    )
    if not Path(config.checkpoint_dir).exists():
        raise ValueError(f"no checkpoint found in {Path(config.checkpoint_dir)}")
    state = restore_latest_rref_matrixformer_checkpoint(train_config)
    checkpoint_step = _checkpoint_step_from_state_step(state.step)
    model = RREFMatrixFormer(
        rows=samples.rows,
        cols=samples.cols,
        modulus=samples.modulus,
        row_embedding_dim=config.row_embedding_dim,
        col_embedding_dim=config.col_embedding_dim,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        num_heads=config.num_heads,
    )

    max_chunk_size = 0

    def logits_provider(matrices: Sequence[Matrix], _step_index: int) -> list[Mapping[str, Any]]:
        nonlocal max_chunk_size
        outputs_by_matrix: list[Mapping[str, Any]] = []
        for start in range(0, len(matrices), resolved_batch_size):
            chunk = matrices[start : start + resolved_batch_size]
            max_chunk_size = max(max_chunk_size, len(chunk))
            inputs = np.asarray(chunk, dtype=np.float32)
            inputs %= float(samples.modulus)
            inputs /= float(samples.modulus - 1)
            outputs = cast(
                dict[str, Any],
                model.apply({"params": state.params}, jnp.asarray(inputs)),
            )
            device_outputs = {
                key: np.asarray(jax.device_get(value)) for key, value in outputs.items()
            }
            for index in range(len(chunk)):
                outputs_by_matrix.append(
                    {
                        "action_kind_logits": device_outputs["action_kind_logits"][index],
                        "action_target_logits": device_outputs["action_target_logits"][index],
                        "action_source_logits": device_outputs["action_source_logits"][index],
                        "action_scalar_logits": device_outputs["action_scalar_logits"][index],
                    }
                )
        return outputs_by_matrix

    result = rollout_rref_verifier_beam_with_logits(
        current_matrix,
        modulus=samples.modulus,
        max_steps=max_steps,
        beam_width=config.beam_width,
        logits_provider=logits_provider,
        checkpoint_step=checkpoint_step,
    )
    return replace(result, device_batch_size=max(result.device_batch_size, max_chunk_size))


def _sample_index(config: RREFMatrixFormerRolloutConfig, sample_index: int | None) -> int:
    index = config.sample_index if sample_index is None else sample_index
    if index is None:
        raise ValueError("sample_index is required")
    return index


def _trace_initial_matrix(data_path: str | Path, sample_index: int) -> Matrix:
    arrays, metadata = load_rref_state_shard(data_path)
    trace_count = int(metadata["trace_count"])
    if sample_index < 0 or sample_index >= trace_count:
        raise ValueError(f"sample_index out of range: {sample_index}")
    if not bool(arrays["trace_step_mask"][sample_index, 0]):
        raise ValueError(f"trace {sample_index} has no initial state")
    initial_state = np.asarray(arrays["trace_states"][sample_index, 0], dtype=np.int64)
    return cast(Matrix, initial_state.tolist())


def rollout_rref_matrixformer_sample(
    config: RREFMatrixFormerRolloutConfig,
    sample_index: int | None = None,
) -> RREFMatrixFormerRolloutResult:
    index = _sample_index(config, sample_index)
    matrix = _trace_initial_matrix(config.data_path, index)
    return rollout_rref_matrixformer(config, matrix)


def _beam_sample_index(config: RREFVerifierBeamConfig, sample_index: int | None) -> int:
    index = config.sample_index if sample_index is None else sample_index
    if index is None:
        raise ValueError("sample_index is required")
    return index


def rollout_rref_verifier_beam_sample(
    config: RREFVerifierBeamConfig,
    sample_index: int | None = None,
) -> RREFVerifierBeamResult:
    index = _beam_sample_index(config, sample_index)
    matrix = _trace_initial_matrix(config.data_path, index)
    return rollout_rref_verifier_beam(config, matrix)
