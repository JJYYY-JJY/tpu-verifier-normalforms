from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nf_agent.data.rref_shards import write_rref_shard
from nf_agent.env.rref_modp import RowOp, is_rref_modp, replay_row_ops
from nf_agent.rollout import (
    RREFPivotRolloutConfig,
    rollout_rref_pivot_sample,
    rollout_rref_pivot_with_logits,
)
from nf_agent.train import TrainConfig, train_rref_pivot

CONFIG = Path("configs/rref_8x8_mod101.yaml")


def _logits(
    *,
    kind: int,
    target: int = 0,
    source: int = 0,
    scalar: int = 0,
    rows: int = 2,
    p: int = 5,
    scalar_second: int | None = None,
) -> dict[str, np.ndarray]:
    op_kind = np.full((4,), -100.0, dtype=np.float32)
    op_target = np.full((rows,), -100.0, dtype=np.float32)
    op_source = np.full((rows,), -100.0, dtype=np.float32)
    op_scalar = np.full((p,), -100.0, dtype=np.float32)
    op_kind[kind] = 100.0
    op_target[target] = 100.0
    op_source[source] = 100.0
    op_scalar[scalar % p] = 100.0
    if scalar_second is not None:
        op_scalar[scalar_second % p] = 90.0
    return {
        "op_kind_logits": op_kind,
        "op_target_logits": op_target,
        "op_source_logits": op_source,
        "op_scalar_logits": op_scalar,
    }


def test_stop_pad_kind_on_non_rref_is_masked_to_legal_action() -> None:
    result = rollout_rref_pivot_with_logits(
        [[2]],
        modulus=5,
        max_steps=1,
        logits_provider=lambda _matrix, _step: _logits(kind=0, scalar=3, rows=1, p=5),
    )

    assert result.success
    assert result.step_count == 1
    assert result.invalid_action_breakdown["op_kind"] == 1
    assert result.invalid_action_count == 1
    assert result.masked_action_count == 1
    assert result.ops == [RowOp.scale(0, 3)]
    assert result.final_matrix == [[1]]


def test_zero_scale_scalar_is_masked_before_exact_replay() -> None:
    result = rollout_rref_pivot_with_logits(
        [[2]],
        modulus=5,
        max_steps=1,
        logits_provider=lambda _matrix, _step: _logits(
            kind=2,
            scalar=0,
            scalar_second=3,
            rows=1,
            p=5,
        ),
    )

    assert result.success
    assert result.invalid_action_breakdown["op_scalar"] == 1
    assert result.ops == [RowOp.scale(0, 3)]
    assert result.final_matrix == [[1]]


def test_source_equal_target_for_swap_is_masked_when_another_row_exists() -> None:
    result = rollout_rref_pivot_with_logits(
        [[0, 1], [1, 0]],
        modulus=5,
        max_steps=1,
        logits_provider=lambda _matrix, _step: _logits(kind=1, target=0, source=0, rows=2, p=5),
    )

    assert result.success
    assert result.invalid_action_breakdown["op_source"] == 1
    assert result.ops == [RowOp.swap(0, 1)]
    assert result.final_matrix == [[1, 0], [0, 1]]


def test_zero_add_scalar_is_masked_to_avoid_noop_add() -> None:
    result = rollout_rref_pivot_with_logits(
        [[1, 0], [1, 1]],
        modulus=5,
        max_steps=1,
        logits_provider=lambda _matrix, _step: _logits(
            kind=3,
            target=1,
            source=0,
            scalar=0,
            scalar_second=4,
            rows=2,
            p=5,
        ),
    )

    assert result.success
    assert result.invalid_action_breakdown["op_scalar"] == 1
    assert result.ops == [RowOp.add(1, 0, 4)]
    assert result.final_matrix == [[1, 0], [0, 1]]


def test_scripted_logits_solve_small_matrix_and_replay_matches_final() -> None:
    result = rollout_rref_pivot_with_logits(
        [[2, 0], [0, 1]],
        modulus=5,
        max_steps=1,
        logits_provider=lambda _matrix, _step: _logits(kind=2, target=0, scalar=3),
    )

    assert result.success
    assert result.status == "success"
    assert result.final_is_rref
    assert replay_row_ops(result.initial_matrix, result.ops, result.modulus) == result.final_matrix


def test_max_step_exhaustion_reports_partial_trace_without_teacher_fallback() -> None:
    result = rollout_rref_pivot_with_logits(
        [[2]],
        modulus=5,
        max_steps=1,
        logits_provider=lambda _matrix, _step: _logits(kind=2, scalar=1, rows=1, p=5),
    )

    assert result.status == "max_steps_exceeded"
    assert not result.success
    assert result.step_count == 1
    assert not result.final_is_rref
    assert result.final_matrix == [[2]]


def test_rollout_rref_pivot_sample_restores_checkpoint_and_reports_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_path = tmp_path / "rref_rollout.npz"
    ckpt_dir = tmp_path / "ckpt"
    write_rref_shard(config_path=CONFIG, count=8, seed_start=0, out_path=shard_path)
    train_rref_pivot(
        TrainConfig(
            data_path=shard_path,
            steps=2,
            batch_size=4,
            out_dir=ckpt_dir,
            hidden_sizes=(32,),
        )
    )

    def fail_teacher_call(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("neural rollout must not call the leftmost teacher")

    monkeypatch.setattr("nf_agent.teachers.leftmost.LeftmostRREFTeacher.solve", fail_teacher_call)

    result = rollout_rref_pivot_sample(
        RREFPivotRolloutConfig(
            data_path=shard_path,
            checkpoint_dir=ckpt_dir,
            max_steps=2,
            hidden_sizes=(32,),
            sample_index=0,
        )
    )

    assert result.status in {"success", "max_steps_exceeded"}
    assert result.success is (result.status == "success")
    assert result.checkpoint_step == 2
    assert result.modulus == 101
    assert isinstance(result.initial_matrix[0][0], int)
    assert result.final_is_rref is is_rref_modp(result.final_matrix, result.modulus)


def test_rollout_rref_pivot_sample_rejects_bad_paths_and_sample_index(tmp_path: Path) -> None:
    shard_path = tmp_path / "rref_rollout.npz"
    write_rref_shard(config_path=CONFIG, count=2, seed_start=0, out_path=shard_path)

    with pytest.raises(ValueError, match="no checkpoint found"):
        rollout_rref_pivot_sample(
            RREFPivotRolloutConfig(
                data_path=shard_path,
                checkpoint_dir=tmp_path / "missing_ckpt",
                sample_index=0,
            )
        )

    with pytest.raises(ValueError, match="data path does not exist"):
        rollout_rref_pivot_sample(
            RREFPivotRolloutConfig(
                data_path=tmp_path / "missing.npz",
                checkpoint_dir=tmp_path / "missing_ckpt",
                sample_index=0,
            )
        )

    with pytest.raises(ValueError, match="sample_index out of range"):
        rollout_rref_pivot_sample(
            RREFPivotRolloutConfig(
                data_path=shard_path,
                checkpoint_dir=tmp_path / "missing_ckpt",
                sample_index=3,
            )
        )
