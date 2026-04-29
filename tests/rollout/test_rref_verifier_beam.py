from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nf_agent.data.rref_backward_shards import write_rref_backward_shard
from nf_agent.data.rref_state_shards import write_rref_state_shard
from nf_agent.env.rref_modp import RowOp, is_rref_modp, replay_row_ops
from nf_agent.rollout import (
    RREFVerifierBeamConfig,
    rollout_rref_verifier_beam_sample,
    rollout_rref_verifier_beam_with_logits,
)
from nf_agent.train import RREFMatrixFormerTrainConfig, train_rref_matrixformer


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


def _state_shard(tmp_path: Path, *, suffix: str = ".zarr") -> Path:
    trace_path = tmp_path / f"backward{suffix}"
    write_rref_backward_shard(
        config_path=_write_backward_config(tmp_path),
        count=4,
        seed_start=0,
        out_path=trace_path,
    )
    state_path = tmp_path / f"state{suffix}"
    write_rref_state_shard(trace_path, state_path)
    return state_path


def test_scripted_verifier_beam_solves_small_matrix_by_exact_replay() -> None:
    result = rollout_rref_verifier_beam_with_logits(
        [[2, 0], [0, 1]],
        modulus=5,
        max_steps=1,
        beam_width=2,
        logits_provider=lambda _matrices, _step: [
            _logits(kind=2, target=0, scalar=3, rows=2, p=5)
        ],
    )

    assert result.status == "success"
    assert result.success
    assert result.final_is_rref
    assert result.replay_ok
    assert result.ops == [RowOp.scale(0, 3)]
    assert replay_row_ops(result.initial_matrix, result.ops, result.modulus) == result.final_matrix
    assert result.expanded_count >= 1
    assert result.pruned_count >= 0
    assert result.device_batch_size == 1


def test_verifier_beam_masks_illegal_stop_source_equal_target_and_zero_scalar() -> None:
    stop_result = rollout_rref_verifier_beam_with_logits(
        [[2]],
        modulus=5,
        max_steps=1,
        beam_width=2,
        logits_provider=lambda _matrices, _step: [
            _logits(kind=0, target=0, scalar=3, rows=1, p=5)
        ],
    )
    add_result = rollout_rref_verifier_beam_with_logits(
        [[1, 0], [1, 1]],
        modulus=5,
        max_steps=1,
        beam_width=2,
        logits_provider=lambda _matrices, _step: [
            _logits(
                kind=3,
                target=1,
                source=1,
                scalar=0,
                scalar_second=4,
                rows=2,
                p=5,
            )
        ],
    )

    assert stop_result.success
    assert stop_result.ops == [RowOp.scale(0, 3)]
    assert add_result.success
    assert add_result.ops == [RowOp.add(1, 0, 4)]


def test_verifier_beam_rejects_invalid_beam_width() -> None:
    with pytest.raises(ValueError, match="beam_width must be positive"):
        rollout_rref_verifier_beam_with_logits(
            [[1]],
            modulus=5,
            max_steps=1,
            beam_width=0,
            logits_provider=lambda _matrices, _step: [],
        )


def test_checkpoint_verifier_beam_restores_matrixformer_without_teacher_fallback(
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
        raise AssertionError("verifier beam must not call the leftmost teacher")

    monkeypatch.setattr("nf_agent.teachers.leftmost.LeftmostRREFTeacher.solve", fail_teacher_call)

    result = rollout_rref_verifier_beam_sample(
        RREFVerifierBeamConfig(
            data_path=shard_path,
            checkpoint_dir=ckpt_dir,
            sample_index=0,
            max_steps=4,
            beam_width=4,
            batch_size="auto",
            row_embedding_dim=8,
            col_embedding_dim=8,
            hidden_dim=32,
            layers=1,
            num_heads=1,
        )
    )

    assert result.status in {"success", "max_steps_exceeded"}
    assert result.success is (result.status == "success")
    assert result.final_is_rref is is_rref_modp(result.final_matrix, result.modulus)
    assert result.replay_ok
    assert result.beam_width == 4
    assert result.checkpoint_step == 2

