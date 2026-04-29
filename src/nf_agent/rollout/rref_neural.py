from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np

from nf_agent.data.rref_shards import RREFShardSamples
from nf_agent.env.elementary_ops import Matrix, normalize_matrix
from nf_agent.env.rref_modp import RowOp, is_rref_modp, replay_row_ops
from nf_agent.models import PivotMLP
from nf_agent.train import TrainConfig, restore_latest_rref_pivot_checkpoint

RolloutStatus: TypeAlias = Literal["success", "max_steps_exceeded"]
LogitsProvider: TypeAlias = Callable[[Matrix, int], Mapping[str, Any]]

_BREAKDOWN_KEYS = ("op_kind", "op_source", "op_scalar")


@dataclass(frozen=True)
class RREFPivotRolloutConfig:
    data_path: str | Path
    checkpoint_dir: str | Path
    max_steps: int | None = None
    hidden_sizes: tuple[int, ...] = (256, 256)
    sample_index: int | None = None


@dataclass(frozen=True)
class RREFPivotRolloutResult:
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


def _masked_argmax(logits: np.ndarray, legal_indices: Sequence[int]) -> int:
    if not legal_indices:
        raise ValueError("at least one legal action is required")
    masked = np.full_like(logits, -np.inf, dtype=np.float32)
    masked[list(legal_indices)] = logits[list(legal_indices)]
    return _argmax(masked)


def _select_kind(logits: Mapping[str, Any], rows: int, breakdown: dict[str, int]) -> int:
    kind_logits = _as_logits(logits["op_kind_logits"], "op_kind", 4)
    proposed = _argmax(kind_logits)
    legal_kinds = [2]
    if rows >= 2:
        legal_kinds = [1, 2, 3]
    if proposed in legal_kinds:
        return proposed
    breakdown["op_kind"] += 1
    return _masked_argmax(kind_logits, legal_kinds)


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


def _select_nonzero_scalar(
    logits: Mapping[str, Any],
    p: int,
    breakdown: dict[str, int],
) -> int:
    scalar_logits = _as_logits(logits["op_scalar_logits"], "op_scalar", p)
    proposed = _argmax(scalar_logits) % p
    if proposed != 0:
        return proposed
    breakdown["op_scalar"] += 1
    return _masked_argmax(scalar_logits, range(1, p))


def _decode_legal_row_op(
    current: Matrix,
    p: int,
    logits: Mapping[str, Any],
    breakdown: dict[str, int],
) -> RowOp:
    rows = len(current)
    if rows == 0:
        raise ValueError("cannot decode a row operation for a matrix with no rows")

    kind = _select_kind(logits, rows, breakdown)
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
) -> RREFPivotRolloutResult:
    final_is_rref = is_rref_modp(current, modulus)
    invalid_count = sum(breakdown.values())
    return RREFPivotRolloutResult(
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


def rollout_rref_pivot_with_logits(
    matrix: Sequence[Sequence[int]],
    *,
    modulus: int,
    max_steps: int,
    logits_provider: LogitsProvider,
    checkpoint_step: int | None = None,
) -> RREFPivotRolloutResult:
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


def _resolve_max_steps(config: RREFPivotRolloutConfig, samples: RREFShardSamples) -> int:
    max_steps = samples.max_ops if config.max_steps is None else config.max_steps
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    if max_steps > samples.max_ops:
        raise ValueError(f"max_steps must be <= shard max_ops ({samples.max_ops})")
    return max_steps


def _validate_matrix_shape(matrix: Matrix, samples: RREFShardSamples) -> None:
    if len(matrix) != samples.rows:
        raise ValueError(f"matrix must have {samples.rows} rows, got {len(matrix)}")
    if matrix and len(matrix[0]) != samples.cols:
        raise ValueError(f"matrix must have {samples.cols} cols, got {len(matrix[0])}")


def _checkpoint_step_from_state_step(step: Any) -> int:
    return int(np.asarray(jax.device_get(step)))


def rollout_rref_pivot(
    config: RREFPivotRolloutConfig,
    matrix: Sequence[Sequence[int]],
) -> RREFPivotRolloutResult:
    samples = RREFShardSamples(config.data_path)
    current_matrix = normalize_matrix(matrix, samples.modulus)
    _validate_matrix_shape(current_matrix, samples)
    max_steps = _resolve_max_steps(config, samples)

    train_config = TrainConfig(
        data_path=config.data_path,
        steps=1,
        batch_size=1,
        out_dir=config.checkpoint_dir,
        hidden_sizes=config.hidden_sizes,
    )
    if not Path(config.checkpoint_dir).exists():
        raise ValueError(f"no checkpoint found in {Path(config.checkpoint_dir)}")
    state = restore_latest_rref_pivot_checkpoint(train_config)
    checkpoint_step = _checkpoint_step_from_state_step(state.step)
    model = PivotMLP(
        rows=samples.rows,
        cols=samples.cols,
        max_pivots=samples.max_pivots,
        max_ops=samples.max_ops,
        modulus=samples.modulus,
        hidden_sizes=config.hidden_sizes,
    )

    def logits_provider(current: Matrix, step_index: int) -> Mapping[str, Any]:
        inputs = np.asarray(current, dtype=np.float32)
        inputs %= float(samples.modulus)
        inputs /= float(samples.modulus - 1)
        outputs = cast(
            dict[str, Any],
            model.apply({"params": state.params}, jnp.asarray(inputs[np.newaxis, ...])),
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
        }

    return rollout_rref_pivot_with_logits(
        current_matrix,
        modulus=samples.modulus,
        max_steps=max_steps,
        logits_provider=logits_provider,
        checkpoint_step=checkpoint_step,
    )


def _sample_index(config: RREFPivotRolloutConfig, sample_index: int | None) -> int:
    index = config.sample_index if sample_index is None else sample_index
    if index is None:
        raise ValueError("sample_index is required")
    return index


def _raw_sample_matrix(
    samples: RREFShardSamples,
    data_path: str | Path,
    sample_index: int,
) -> Matrix:
    if sample_index < 0 or sample_index >= len(samples):
        raise ValueError(f"sample_index out of range: {sample_index}")
    with np.load(data_path, allow_pickle=False) as shard:
        return cast(Matrix, np.asarray(shard["inputs"][sample_index], dtype=np.int64).tolist())


def rollout_rref_pivot_sample(
    config: RREFPivotRolloutConfig,
    sample_index: int | None = None,
) -> RREFPivotRolloutResult:
    samples = RREFShardSamples(config.data_path)
    index = _sample_index(config, sample_index)
    matrix = _raw_sample_matrix(samples, config.data_path, index)
    return rollout_rref_pivot(config, matrix)
