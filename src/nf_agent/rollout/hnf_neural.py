from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np

from nf_agent.data.hnf_shards import HNFShardSamples
from nf_agent.env.elementary_ops import Matrix
from nf_agent.env.hnf_int import (
    IntegerRowOp,
    is_row_hnf,
    normalize_integer_matrix,
    replay_integer_row_ops,
)
from nf_agent.models import HNFPolicyMLP
from nf_agent.train import HNFTrainConfig, restore_latest_hnf_policy_checkpoint

HNFRolloutStatus: TypeAlias = Literal["success", "max_steps_exceeded", "invalid_action"]
HNFLogitsProvider: TypeAlias = Callable[[Matrix, int], Mapping[str, Any]]

_BREAKDOWN_KEYS = ("op_kind", "op_source")


@dataclass(frozen=True)
class HNFRolloutConfig:
    data_path: str | Path
    checkpoint_dir: str | Path
    max_steps: int | None = None
    hidden_sizes: tuple[int, ...] = (256, 256)
    sample_index: int | None = None
    beam_width: int = 8


@dataclass(frozen=True)
class HNFRolloutResult:
    status: HNFRolloutStatus
    success: bool
    step_count: int
    invalid_action_count: int
    masked_action_count: int
    invalid_action_breakdown: dict[str, int]
    initial_matrix: Matrix
    final_matrix: Matrix
    ops: list[IntegerRowOp]
    final_is_hnf: bool
    checkpoint_step: int | None
    scalar_vocab: list[int]
    visited_matrices: list[Matrix]
    beam_width: int | None = None
    score: float | None = None

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
            "ops": [_integer_row_op_to_dict(op) for op in self.ops],
            "final_is_hnf": self.final_is_hnf,
            "checkpoint_step": self.checkpoint_step,
            "scalar_vocab": list(self.scalar_vocab),
            "beam_width": self.beam_width,
            "score": self.score,
        }


@dataclass(frozen=True)
class HNFPolicyRuntime:
    samples: HNFShardSamples
    model: HNFPolicyMLP
    params: Any
    checkpoint_step: int
    scalar_vocab: list[int]


@dataclass(frozen=True)
class _ActionCandidate:
    op: IntegerRowOp
    log_probability: float


@dataclass(frozen=True)
class _BeamState:
    matrix: Matrix
    ops: tuple[IntegerRowOp, ...]
    visited: tuple[Matrix, ...]
    score: float
    max_bitlength_seen: int


def _integer_row_op_to_dict(op: IntegerRowOp) -> dict[str, int | str]:
    payload: dict[str, int | str] = {"kind": op.kind, "target": op.target}
    if op.source is not None:
        payload["source"] = op.source
    if op.scalar is not None:
        payload["scalar"] = op.scalar
    return payload


def _new_breakdown() -> dict[str, int]:
    return {key: 0 for key in _BREAKDOWN_KEYS}


FloatArray = np.ndarray[Any, np.dtype[np.float32]]


def _as_logits(value: object, name: str, expected_size: int) -> FloatArray:
    logits = np.asarray(value, dtype=np.float32)
    expected_shape = (expected_size,)
    if logits.shape != expected_shape:
        raise ValueError(f"{name} logits must have shape {expected_shape}, got {logits.shape}")
    return logits


def _argmax(logits: FloatArray) -> int:
    return int(np.argmax(logits))


def _masked_argmax(
    logits: FloatArray,
    legal_indices: Sequence[int],
) -> int:
    if not legal_indices:
        raise ValueError("at least one legal action is required")
    masked = np.full_like(logits, -np.inf, dtype=np.float32)
    masked[list(legal_indices)] = logits[list(legal_indices)]
    return _argmax(masked)


def _legal_kinds(rows: int, scalar_vocab: Sequence[int]) -> list[int]:
    if rows <= 0:
        return []
    kinds = [2]
    if rows >= 2:
        kinds.extend([1])
        if scalar_vocab:
            kinds.append(3)
    return sorted(kinds)


def _select_kind(
    logits: Mapping[str, Any],
    rows: int,
    scalar_vocab: Sequence[int],
    breakdown: dict[str, int],
) -> int:
    kind_logits = _as_logits(logits["op_kind_logits"], "op_kind", 4)
    proposed = _argmax(kind_logits)
    legal = _legal_kinds(rows, scalar_vocab)
    if proposed in legal:
        return proposed
    breakdown["op_kind"] += 1
    return _masked_argmax(kind_logits, legal)


def _select_target(logits: Mapping[str, Any], rows: int) -> int:
    target_logits = _as_logits(logits["op_target_logits"], "op_target", rows)
    return _argmax(target_logits)


def _select_source(
    logits: Mapping[str, Any],
    rows: int,
    target: int,
    breakdown: dict[str, int],
) -> int:
    source_logits = _as_logits(logits["op_source_logits"], "op_source", rows)
    proposed = _argmax(source_logits)
    if proposed != target:
        return proposed
    breakdown["op_source"] += 1
    return _masked_argmax(source_logits, [row for row in range(rows) if row != target])


def _select_scalar(logits: Mapping[str, Any], scalar_vocab: Sequence[int]) -> int:
    if not scalar_vocab:
        raise ValueError("scalar_vocab must be non-empty for add operations")
    scalar_logits = _as_logits(logits["op_scalar_logits"], "op_scalar", len(scalar_vocab))
    return scalar_vocab[_argmax(scalar_logits)]


def _decode_legal_integer_row_op(
    current: Matrix,
    scalar_vocab: Sequence[int],
    logits: Mapping[str, Any],
    breakdown: dict[str, int],
) -> IntegerRowOp:
    rows = len(current)
    if rows == 0:
        raise ValueError("cannot decode a row operation for a matrix with no rows")
    kind = _select_kind(logits, rows, scalar_vocab, breakdown)
    target = _select_target(logits, rows)
    if kind == 1:
        source = _select_source(logits, rows, target, breakdown)
        return IntegerRowOp.swap(target, source)
    if kind == 2:
        return IntegerRowOp.negate(target)
    if kind == 3:
        source = _select_source(logits, rows, target, breakdown)
        scalar = _select_scalar(logits, scalar_vocab)
        return IntegerRowOp.add(target, source, scalar)
    raise ValueError(f"unknown decoded integer row operation kind: {kind}")


def _finish_result(
    *,
    status: HNFRolloutStatus,
    initial: Matrix,
    current: Matrix,
    ops: list[IntegerRowOp],
    breakdown: dict[str, int],
    checkpoint_step: int | None,
    scalar_vocab: Sequence[int],
    visited_matrices: list[Matrix],
    beam_width: int | None = None,
    score: float | None = None,
) -> HNFRolloutResult:
    final_is_hnf = is_row_hnf(current)
    invalid_count = sum(breakdown.values())
    return HNFRolloutResult(
        status=status,
        success=status == "success" and final_is_hnf,
        step_count=len(ops),
        invalid_action_count=invalid_count,
        masked_action_count=invalid_count,
        invalid_action_breakdown=dict(breakdown),
        initial_matrix=initial,
        final_matrix=current,
        ops=list(ops),
        final_is_hnf=final_is_hnf,
        checkpoint_step=checkpoint_step,
        scalar_vocab=list(scalar_vocab),
        visited_matrices=[matrix for matrix in visited_matrices],
        beam_width=beam_width,
        score=score,
    )


def rollout_hnf_policy_with_logits(
    matrix: Sequence[Sequence[int]],
    *,
    scalar_vocab: Sequence[int],
    max_steps: int,
    logits_provider: HNFLogitsProvider,
    checkpoint_step: int | None = None,
) -> HNFRolloutResult:
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    initial = normalize_integer_matrix(matrix)
    current = [row[:] for row in initial]
    visited = [[row[:] for row in current]]
    ops: list[IntegerRowOp] = []
    breakdown = _new_breakdown()
    for step_index in range(max_steps):
        if is_row_hnf(current):
            return _finish_result(
                status="success",
                initial=initial,
                current=current,
                ops=ops,
                breakdown=breakdown,
                checkpoint_step=checkpoint_step,
                scalar_vocab=scalar_vocab,
                visited_matrices=visited,
            )
        logits = logits_provider(current, step_index)
        try:
            op = _decode_legal_integer_row_op(current, scalar_vocab, logits, breakdown)
            current = replay_integer_row_ops(current, [op])
        except (TypeError, ValueError, IndexError):
            breakdown["op_kind"] += 1
            return _finish_result(
                status="invalid_action",
                initial=initial,
                current=current,
                ops=ops,
                breakdown=breakdown,
                checkpoint_step=checkpoint_step,
                scalar_vocab=scalar_vocab,
                visited_matrices=visited,
            )
        ops.append(op)
        visited.append([row[:] for row in current])

    final_status: HNFRolloutStatus = "success" if is_row_hnf(current) else "max_steps_exceeded"
    return _finish_result(
        status=final_status,
        initial=initial,
        current=current,
        ops=ops,
        breakdown=breakdown,
        checkpoint_step=checkpoint_step,
        scalar_vocab=scalar_vocab,
        visited_matrices=visited,
    )


def _resolve_max_steps(config: HNFRolloutConfig, samples: HNFShardSamples) -> int:
    max_steps = samples.max_ops if config.max_steps is None else config.max_steps
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    return min(max_steps, samples.max_ops)


def _validate_matrix_shape(matrix: Matrix, samples: HNFShardSamples) -> None:
    if len(matrix) != samples.rows:
        raise ValueError(f"matrix must have {samples.rows} rows, got {len(matrix)}")
    if matrix and len(matrix[0]) != samples.cols:
        raise ValueError(f"matrix must have {samples.cols} cols, got {len(matrix[0])}")


def _checkpoint_step_from_state_step(step: Any) -> int:
    return int(np.asarray(jax.device_get(step)))


def load_hnf_policy_runtime(
    config: HNFRolloutConfig,
) -> HNFPolicyRuntime:
    samples = HNFShardSamples(config.data_path)
    if not Path(config.checkpoint_dir).exists():
        raise ValueError(f"no checkpoint found in {Path(config.checkpoint_dir)}")
    train_config = HNFTrainConfig(
        data_path=config.data_path,
        steps=1,
        batch_size=1,
        out_dir=config.checkpoint_dir,
        hidden_sizes=config.hidden_sizes,
    )
    state = restore_latest_hnf_policy_checkpoint(train_config)
    checkpoint_step = _checkpoint_step_from_state_step(state.step)
    model = HNFPolicyMLP(
        rows=samples.rows,
        cols=samples.cols,
        max_ops=samples.max_ops,
        scalar_vocab_size=max(1, samples.scalar_vocab_size),
        hidden_sizes=config.hidden_sizes,
    )
    scalar_vocab = [int(value) for value in samples.scalar_vocab]
    return HNFPolicyRuntime(
        samples=samples,
        model=model,
        params=state.params,
        checkpoint_step=checkpoint_step,
        scalar_vocab=scalar_vocab,
    )


def _runtime_logits_provider(runtime: HNFPolicyRuntime) -> HNFLogitsProvider:
    def logits_provider(current: Matrix, step_index: int) -> Mapping[str, Any]:
        inputs = np.asarray(current, dtype=np.float32)
        inputs /= float(runtime.samples.input_scale)
        outputs = cast(
            dict[str, Any],
            runtime.model.apply(
                {"params": runtime.params},
                jnp.asarray(inputs[np.newaxis, ...]),
            ),
        )
        return {
            "op_kind_logits": np.asarray(
                jax.device_get(outputs["op_kind_logits"][0, step_index])
            ),
            "op_target_logits": np.asarray(
                jax.device_get(outputs["op_target_logits"][0, step_index])
            ),
            "op_source_logits": np.asarray(
                jax.device_get(outputs["op_source_logits"][0, step_index])
            ),
            "op_scalar_logits": np.asarray(
                jax.device_get(outputs["op_scalar_logits"][0, step_index])
            ),
            "value": np.asarray(jax.device_get(outputs["value"][0])),
        }

    return logits_provider


def rollout_hnf_policy_with_runtime(
    runtime: HNFPolicyRuntime,
    config: HNFRolloutConfig,
    matrix: Sequence[Sequence[int]],
) -> HNFRolloutResult:
    current_matrix = normalize_integer_matrix(matrix)
    _validate_matrix_shape(current_matrix, runtime.samples)
    max_steps = _resolve_max_steps(config, runtime.samples)
    return rollout_hnf_policy_with_logits(
        current_matrix,
        scalar_vocab=runtime.scalar_vocab,
        max_steps=max_steps,
        logits_provider=_runtime_logits_provider(runtime),
        checkpoint_step=runtime.checkpoint_step,
    )


def rollout_hnf_policy(
    config: HNFRolloutConfig,
    matrix: Sequence[Sequence[int]],
) -> HNFRolloutResult:
    runtime = load_hnf_policy_runtime(config)
    return rollout_hnf_policy_with_runtime(runtime, config, matrix)


def _sample_index(config: HNFRolloutConfig, sample_index: int | None) -> int:
    index = config.sample_index if sample_index is None else sample_index
    if index is None:
        raise ValueError("sample_index is required")
    return index


def _raw_sample_matrix(
    samples: HNFShardSamples,
    data_path: str | Path,
    sample_index: int,
) -> Matrix:
    if sample_index < 0 or sample_index >= len(samples):
        raise ValueError(f"sample_index out of range: {sample_index}")
    with np.load(data_path, allow_pickle=False) as shard:
        return cast(Matrix, np.asarray(shard["inputs"][sample_index], dtype=np.int64).tolist())


def rollout_hnf_policy_sample(
    config: HNFRolloutConfig,
    sample_index: int | None = None,
) -> HNFRolloutResult:
    samples = HNFShardSamples(config.data_path)
    index = _sample_index(config, sample_index)
    matrix = _raw_sample_matrix(samples, config.data_path, index)
    return rollout_hnf_policy(config, matrix)


def _log_softmax(logits: FloatArray) -> FloatArray:
    finite = logits.astype(np.float64)
    finite -= np.max(finite)
    probs = np.exp(finite)
    probs /= np.sum(probs)
    return np.log(probs).astype(np.float32)


def _top_indices(
    logits: FloatArray,
    legal: Sequence[int],
    limit: int,
) -> list[tuple[int, float]]:
    if not legal:
        return []
    log_probs = _log_softmax(logits)
    ordered = sorted(legal, key=lambda index: float(log_probs[index]), reverse=True)
    return [(index, float(log_probs[index])) for index in ordered[:limit]]


def _candidate_actions(
    logits: Mapping[str, Any],
    *,
    rows: int,
    scalar_vocab: Sequence[int],
    beam_width: int,
) -> list[_ActionCandidate]:
    kind_logits = _as_logits(logits["op_kind_logits"], "op_kind", 4)
    target_logits = _as_logits(logits["op_target_logits"], "op_target", rows)
    source_logits = _as_logits(logits["op_source_logits"], "op_source", rows)
    scalar_logits = _as_logits(logits["op_scalar_logits"], "op_scalar", max(1, len(scalar_vocab)))
    candidates: list[_ActionCandidate] = []
    for kind, kind_lp in _top_indices(kind_logits, _legal_kinds(rows, scalar_vocab), beam_width):
        for target, target_lp in _top_indices(target_logits, list(range(rows)), beam_width):
            if kind == 2:
                candidates.append(
                    _ActionCandidate(IntegerRowOp.negate(target), kind_lp + target_lp)
                )
                continue
            source_choices = [row for row in range(rows) if row != target]
            for source, source_lp in _top_indices(source_logits, source_choices, beam_width):
                if kind == 1:
                    candidates.append(
                        _ActionCandidate(
                            IntegerRowOp.swap(target, source),
                            kind_lp + target_lp + source_lp,
                        )
                    )
                    continue
                for scalar_id, scalar_lp in _top_indices(
                    scalar_logits,
                    list(range(len(scalar_vocab))),
                    beam_width,
                ):
                    candidates.append(
                        _ActionCandidate(
                            IntegerRowOp.add(target, source, scalar_vocab[scalar_id]),
                            kind_lp + target_lp + source_lp + scalar_lp,
                        )
                    )
    return sorted(candidates, key=lambda item: item.log_probability, reverse=True)[:beam_width]


def _max_bitlength(matrix: Sequence[Sequence[int]]) -> int:
    return max((abs(entry).bit_length() for row in matrix for entry in row), default=0)


def rollout_hnf_beam(
    config: HNFRolloutConfig,
    matrix: Sequence[Sequence[int]],
) -> HNFRolloutResult:
    if config.beam_width <= 0:
        raise ValueError("beam_width must be positive")
    runtime = load_hnf_policy_runtime(config)
    return rollout_hnf_beam_with_runtime(runtime, config, matrix)


def rollout_hnf_beam_with_runtime(
    runtime: HNFPolicyRuntime,
    config: HNFRolloutConfig,
    matrix: Sequence[Sequence[int]],
) -> HNFRolloutResult:
    if config.beam_width <= 0:
        raise ValueError("beam_width must be positive")
    initial = normalize_integer_matrix(matrix)
    _validate_matrix_shape(initial, runtime.samples)
    max_steps = _resolve_max_steps(config, runtime.samples)

    logits_provider = _runtime_logits_provider(runtime)

    initial_state = _BeamState(
        matrix=[row[:] for row in initial],
        ops=(),
        visited=([row[:] for row in initial],),
        score=0.0,
        max_bitlength_seen=_max_bitlength(initial),
    )
    beam = [initial_state]
    best = initial_state
    for step_index in range(max_steps):
        for state in beam:
            if is_row_hnf(state.matrix):
                return _finish_result(
                    status="success",
                    initial=initial,
                    current=state.matrix,
                    ops=list(state.ops),
                    breakdown=_new_breakdown(),
                    checkpoint_step=runtime.checkpoint_step,
                    scalar_vocab=runtime.scalar_vocab,
                    visited_matrices=list(state.visited),
                    beam_width=config.beam_width,
                    score=state.score,
                )
        expanded: list[_BeamState] = []
        for state in beam:
            logits = logits_provider(state.matrix, step_index)
            for action in _candidate_actions(
                logits,
                rows=len(state.matrix),
                scalar_vocab=runtime.scalar_vocab,
                beam_width=config.beam_width,
            ):
                try:
                    next_matrix = replay_integer_row_ops(state.matrix, [action.op])
                except (TypeError, ValueError, IndexError):
                    continue
                bitlength = max(state.max_bitlength_seen, _max_bitlength(next_matrix))
                score = state.score + action.log_probability - 0.001 * bitlength
                expanded.append(
                    _BeamState(
                        matrix=next_matrix,
                        ops=(*state.ops, action.op),
                        visited=(*state.visited, [row[:] for row in next_matrix]),
                        score=score,
                        max_bitlength_seen=bitlength,
                    )
                )
        if not expanded:
            return _finish_result(
                status="invalid_action",
                initial=initial,
                current=best.matrix,
                ops=list(best.ops),
                breakdown={"op_kind": 1, "op_source": 0},
                checkpoint_step=runtime.checkpoint_step,
                scalar_vocab=runtime.scalar_vocab,
                visited_matrices=list(best.visited),
                beam_width=config.beam_width,
                score=best.score,
            )
        beam = sorted(expanded, key=lambda item: item.score, reverse=True)[: config.beam_width]
        best = beam[0]

    status: HNFRolloutStatus = "success" if is_row_hnf(best.matrix) else "max_steps_exceeded"
    return _finish_result(
        status=status,
        initial=initial,
        current=best.matrix,
        ops=list(best.ops),
        breakdown=_new_breakdown(),
        checkpoint_step=runtime.checkpoint_step,
        scalar_vocab=runtime.scalar_vocab,
        visited_matrices=list(best.visited),
        beam_width=config.beam_width,
        score=best.score if not math.isnan(best.score) else None,
    )


def rollout_hnf_beam_sample(
    config: HNFRolloutConfig,
    sample_index: int | None = None,
) -> HNFRolloutResult:
    samples = HNFShardSamples(config.data_path)
    index = _sample_index(config, sample_index)
    matrix = _raw_sample_matrix(samples, config.data_path, index)
    return rollout_hnf_beam(config, matrix)
