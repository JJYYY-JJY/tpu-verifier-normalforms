from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nf_agent.data.hnf_shards import write_hnf_shard
from nf_agent.env.hnf_int import IntegerRowOp, is_row_hnf, replay_integer_row_ops
from nf_agent.rollout import (
    HNFRolloutConfig,
    rollout_hnf_beam_sample,
    rollout_hnf_policy_sample,
    rollout_hnf_policy_with_logits,
)
from nf_agent.train import HNFTrainConfig, train_hnf_policy


def _logits(
    *,
    kind: int,
    target: int = 0,
    source: int = 0,
    scalar_id: int = 0,
    rows: int = 2,
    scalar_vocab_size: int = 2,
) -> dict[str, np.ndarray]:
    op_kind = np.full((4,), -100.0, dtype=np.float32)
    op_target = np.full((rows,), -100.0, dtype=np.float32)
    op_source = np.full((rows,), -100.0, dtype=np.float32)
    op_scalar = np.full((scalar_vocab_size,), -100.0, dtype=np.float32)
    op_kind[kind] = 100.0
    op_target[target] = 100.0
    op_source[source] = 100.0
    op_scalar[scalar_id] = 100.0
    return {
        "op_kind_logits": op_kind,
        "op_target_logits": op_target,
        "op_source_logits": op_source,
        "op_scalar_logits": op_scalar,
        "value": np.asarray(0.0, dtype=np.float32),
    }


def _config(tmp_path: Path) -> Path:
    config_path = tmp_path / "hnf.yaml"
    config_path.write_text(
        "task: hnf\n"
        "integer_matrix:\n"
        "  family: sparse\n"
        "  rows: 3\n"
        "  cols: 3\n"
        "  density: 0.5\n"
        "  entry_bound: 4\n"
    )
    return config_path


def _trained_policy(tmp_path: Path) -> tuple[Path, Path]:
    shard_path = tmp_path / "hnf_rollout.npz"
    ckpt_dir = tmp_path / "ckpt"
    write_hnf_shard(_config(tmp_path), count=6, seed_start=0, out_path=shard_path)
    train_hnf_policy(
        HNFTrainConfig(
            data_path=shard_path,
            steps=1,
            batch_size=3,
            out_dir=ckpt_dir,
            hidden_sizes=(16,),
        )
    )
    return shard_path, ckpt_dir


def test_scripted_negate_logits_solve_one_row_hnf_without_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_oracle(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("HNF rollout must not call row_hnf")

    monkeypatch.setattr("nf_agent.env.hnf_int.row_hnf", fail_oracle)

    result = rollout_hnf_policy_with_logits(
        [[-2]],
        scalar_vocab=[1],
        max_steps=1,
        logits_provider=lambda _matrix, _step: _logits(
            kind=2,
            rows=1,
            scalar_vocab_size=1,
        ),
    )

    assert result.success
    assert result.status == "success"
    assert result.ops == [IntegerRowOp.negate(0)]
    assert result.final_matrix == [[2]]
    assert result.final_is_hnf


def test_source_equal_target_for_add_is_masked_to_legal_source() -> None:
    result = rollout_hnf_policy_with_logits(
        [[1, 0], [1, 1]],
        scalar_vocab=[-1],
        max_steps=1,
        logits_provider=lambda _matrix, _step: _logits(
            kind=3,
            target=1,
            source=1,
            scalar_id=0,
            rows=2,
            scalar_vocab_size=1,
        ),
    )

    assert result.success
    assert result.invalid_action_breakdown["op_source"] == 1
    assert result.ops == [IntegerRowOp.add(1, 0, -1)]
    assert replay_integer_row_ops(result.initial_matrix, result.ops) == result.final_matrix


def test_max_step_exhaustion_reports_failure_without_teacher_fallback() -> None:
    result = rollout_hnf_policy_with_logits(
        [[-2]],
        scalar_vocab=[1],
        max_steps=0,
        logits_provider=lambda _matrix, _step: _logits(kind=2, rows=1, scalar_vocab_size=1),
    )

    assert result.status == "max_steps_exceeded"
    assert not result.success
    assert not result.final_is_hnf
    assert result.final_matrix == [[-2]]


def test_checkpoint_rollout_and_beam_restore_policy_without_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_path, ckpt_dir = _trained_policy(tmp_path)

    def fail_oracle(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("HNF evaluation rollout must not call row_hnf")

    monkeypatch.setattr("nf_agent.env.hnf_int.row_hnf", fail_oracle)

    greedy = rollout_hnf_policy_sample(
        HNFRolloutConfig(
            data_path=shard_path,
            checkpoint_dir=ckpt_dir,
            sample_index=0,
            max_steps=2,
            hidden_sizes=(16,),
        )
    )
    beam = rollout_hnf_beam_sample(
        HNFRolloutConfig(
            data_path=shard_path,
            checkpoint_dir=ckpt_dir,
            sample_index=0,
            max_steps=2,
            hidden_sizes=(16,),
            beam_width=2,
        )
    )

    assert greedy.status in {"success", "max_steps_exceeded", "invalid_action"}
    assert beam.status in {"success", "max_steps_exceeded", "invalid_action"}
    assert greedy.final_is_hnf is is_row_hnf(greedy.final_matrix)
    assert beam.final_is_hnf is is_row_hnf(beam.final_matrix)
