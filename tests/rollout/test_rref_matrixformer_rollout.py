from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nf_agent.data.rref_backward_shards import write_rref_backward_shard
from nf_agent.data.rref_state_shards import load_rref_state_shard, write_rref_state_shard
from nf_agent.env.rref_modp import RowOp, is_rref_modp, replay_row_ops
from nf_agent.rollout import (
    RREFMatrixFormerRolloutConfig,
    rollout_rref_matrixformer_sample,
    rollout_rref_matrixformer_with_logits,
)
from nf_agent.train import RREFMatrixFormerTrainConfig, train_rref_matrixformer


def _write_backward_config(tmp_path: Path, *, max_backward_ops: int = 4) -> Path:
    config_path = tmp_path / "rref_backward.yaml"
    config_path.write_text(
        "task: rref_backward_state_shards\n"
        "field:\n"
        "  modulus: 101\n"
        "matrix:\n"
        "  family: dense\n"
        "  rows: 4\n"
        "  cols: 4\n"
        "backward_trace:\n"
        "  schema: rref-backward-trace-npz-v1\n"
        "  format: npz\n"
        f"  max_backward_ops: {max_backward_ops}\n"
        "  require_exact_replay: true\n"
    )
    return config_path


def _state_shard(tmp_path: Path, *, count: int = 4, max_ops: int = 4) -> Path:
    config_path = _write_backward_config(tmp_path, max_backward_ops=max_ops)
    trace_path = tmp_path / "backward.npz"
    write_rref_backward_shard(
        config_path=config_path,
        count=count,
        seed_start=0,
        out_path=trace_path,
    )
    state_path = tmp_path / "state.npz"
    write_rref_state_shard(trace_path, state_path)
    return state_path


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
    action_kind = np.full((4,), -100.0, dtype=np.float32)
    action_target = np.full((rows,), -100.0, dtype=np.float32)
    action_source = np.full((rows,), -100.0, dtype=np.float32)
    action_scalar = np.full((p,), -100.0, dtype=np.float32)
    action_kind[kind] = 100.0
    action_target[target] = 100.0
    action_source[source] = 100.0
    action_scalar[scalar % p] = 100.0
    if scalar_second is not None:
        action_scalar[scalar_second % p] = 90.0
    return {
        "action_kind_logits": action_kind,
        "action_target_logits": action_target,
        "action_source_logits": action_source,
        "action_scalar_logits": action_scalar,
    }


def test_scripted_matrixformer_logits_solve_small_matrix_by_exact_replay() -> None:
    result = rollout_rref_matrixformer_with_logits(
        [[2, 0], [0, 1]],
        modulus=5,
        max_steps=1,
        logits_provider=lambda _matrix, _step: _logits(kind=2, target=0, scalar=3),
    )

    assert result.status == "success"
    assert result.success
    assert result.final_is_rref
    assert result.ops == [RowOp.scale(0, 3)]
    assert replay_row_ops(result.initial_matrix, result.ops, result.modulus) == result.final_matrix


def test_scripted_matrixformer_masks_illegal_stop_on_non_rref() -> None:
    result = rollout_rref_matrixformer_with_logits(
        [[2]],
        modulus=5,
        max_steps=1,
        logits_provider=lambda _matrix, _step: _logits(kind=0, scalar=3, rows=1, p=5),
    )

    assert result.success
    assert result.invalid_action_breakdown["action_kind"] == 1
    assert result.masked_action_count == 1
    assert result.ops == [RowOp.scale(0, 3)]
    assert result.final_matrix == [[1]]


def test_scripted_matrixformer_masks_source_equal_target_and_zero_scalar() -> None:
    swap_result = rollout_rref_matrixformer_with_logits(
        [[0, 1], [1, 0]],
        modulus=5,
        max_steps=1,
        logits_provider=lambda _matrix, _step: _logits(kind=1, target=0, source=0, rows=2, p=5),
    )
    add_result = rollout_rref_matrixformer_with_logits(
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

    assert swap_result.success
    assert swap_result.invalid_action_breakdown["action_source"] == 1
    assert swap_result.ops == [RowOp.swap(0, 1)]
    assert add_result.success
    assert add_result.invalid_action_breakdown["action_scalar"] == 1
    assert add_result.ops == [RowOp.add(1, 0, 4)]


def test_checkpoint_matrixformer_rollout_uses_trace_initial_state_without_teacher_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_path = _state_shard(tmp_path)
    ckpt_dir = tmp_path / "ckpt"
    train_rref_matrixformer(
        RREFMatrixFormerTrainConfig(
            data_path=shard_path,
            steps=2,
            batch_size=4,
            out_dir=ckpt_dir,
            row_embedding_dim=8,
            col_embedding_dim=8,
            hidden_dim=32,
            layers=1,
            num_heads=1,
        )
    )

    def fail_teacher_call(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("matrixformer rollout must not call the leftmost teacher")

    monkeypatch.setattr("nf_agent.teachers.leftmost.LeftmostRREFTeacher.solve", fail_teacher_call)

    result = rollout_rref_matrixformer_sample(
        RREFMatrixFormerRolloutConfig(
            data_path=shard_path,
            checkpoint_dir=ckpt_dir,
            sample_index=0,
            max_steps=4,
            row_embedding_dim=8,
            col_embedding_dim=8,
            hidden_dim=32,
            layers=1,
            num_heads=1,
        )
    )
    arrays, _metadata = load_rref_state_shard(shard_path)

    assert result.status in {"success", "max_steps_exceeded"}
    assert result.success is (result.status == "success")
    assert result.checkpoint_step == 2
    assert result.modulus == 101
    assert result.initial_matrix == arrays["trace_states"][0, 0].tolist()
    assert result.final_is_rref is is_rref_modp(result.final_matrix, result.modulus)


def test_matrixformer_rollout_rejects_bad_paths_and_sample_index(tmp_path: Path) -> None:
    shard_path = _state_shard(tmp_path, count=2)

    with pytest.raises(ValueError, match="no checkpoint found"):
        rollout_rref_matrixformer_sample(
            RREFMatrixFormerRolloutConfig(
                data_path=shard_path,
                checkpoint_dir=tmp_path / "missing_ckpt",
                sample_index=0,
            )
        )

    with pytest.raises(ValueError, match="data path does not exist"):
        rollout_rref_matrixformer_sample(
            RREFMatrixFormerRolloutConfig(
                data_path=tmp_path / "missing.npz",
                checkpoint_dir=tmp_path / "missing_ckpt",
                sample_index=0,
            )
        )

    with pytest.raises(ValueError, match="sample_index out of range"):
        rollout_rref_matrixformer_sample(
            RREFMatrixFormerRolloutConfig(
                data_path=shard_path,
                checkpoint_dir=tmp_path / "missing_ckpt",
                sample_index=3,
            )
        )
