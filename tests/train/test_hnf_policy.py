from pathlib import Path

import numpy as np
import pytest

from nf_agent.data.hnf_shards import HNFShardSamples, write_hnf_shard
from nf_agent.train import (
    HNFActorCriticConfig,
    HNFDaggerConfig,
    HNFTrainConfig,
    evaluate_hnf_policy_batch,
    restore_latest_hnf_policy_checkpoint,
    train_hnf_actor_critic,
    train_hnf_dagger,
    train_hnf_policy,
)


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


def _shard(tmp_path: Path, count: int = 6) -> Path:
    shard_path = tmp_path / "hnf_train.npz"
    write_hnf_shard(_config(tmp_path), count=count, seed_start=0, out_path=shard_path)
    return shard_path


def test_train_hnf_policy_saves_checkpoint_and_reports_metrics(tmp_path: Path) -> None:
    shard_path = _shard(tmp_path)
    out_dir = tmp_path / "ckpt"

    result = train_hnf_policy(
        HNFTrainConfig(
            data_path=shard_path,
            steps=2,
            batch_size=3,
            out_dir=out_dir,
            hidden_sizes=(16,),
        )
    )

    assert result["status"] == "ok"
    assert result["final_step"] == 2
    assert result["latest_step"] == 2
    assert np.isfinite(result["final_loss"])
    assert result["parameters_changed"]
    assert result["data_schema_version"] == "hnf-teacher-trajectory-npz-v0.8"
    assert result["scalar_vocab_size"] >= 1
    assert Path(result["checkpoint_dir"]).exists()


def test_hnf_supervised_loss_ignores_inactive_padding_labels(tmp_path: Path) -> None:
    shard_path = _shard(tmp_path)
    samples = HNFShardSamples(shard_path)
    batch = next(iter([samples[0]]))
    config = HNFTrainConfig(
        data_path=shard_path,
        steps=1,
        batch_size=2,
        out_dir=tmp_path / "ckpt",
        hidden_sizes=(16,),
    )
    train_hnf_policy(config)
    state = restore_latest_hnf_policy_checkpoint(config)

    base = evaluate_hnf_policy_batch(config, state.params, batch)
    corrupted = {key: np.copy(value) for key, value in batch.items()}
    inactive = np.logical_not(corrupted["op_mask"])
    corrupted["op_kind"][inactive] = 999
    corrupted["op_target"][inactive] = 999
    corrupted["op_source"][inactive] = 999
    corrupted["op_scalar"][inactive] = 999
    changed = evaluate_hnf_policy_batch(config, state.params, corrupted)

    assert base["loss"] == pytest.approx(changed["loss"])


def test_restore_latest_hnf_policy_checkpoint_continues_step_count(tmp_path: Path) -> None:
    shard_path = _shard(tmp_path)
    out_dir = tmp_path / "ckpt"
    config = HNFTrainConfig(
        data_path=shard_path,
        steps=1,
        batch_size=3,
        out_dir=out_dir,
        hidden_sizes=(16,),
    )
    train_hnf_policy(config)

    resumed = train_hnf_policy(config)
    state = restore_latest_hnf_policy_checkpoint(config)

    assert resumed["final_step"] == 2
    assert int(state.step) == 2


def test_hnf_dagger_builds_aggregate_with_oracle_continuations(tmp_path: Path) -> None:
    shard_path = _shard(tmp_path, count=4)
    result = train_hnf_dagger(
        HNFDaggerConfig(
            data_path=shard_path,
            iterations=1,
            train_steps=1,
            batch_size=2,
            out_dir=tmp_path / "dagger",
            hidden_sizes=(16,),
            rollout_sample_count=2,
            rollout_max_steps=1,
        )
    )

    assert result["status"] == "ok"
    assert result["initial_count"] == 4
    assert result["aggregate_count"] >= 4
    assert result["oracle_continuation_count"] >= 0
    assert Path(result["aggregate_data_path"]).exists()
    assert Path(result["checkpoint_dir"]).exists()


def test_hnf_actor_critic_update_changes_params_and_reports_reward_metrics(tmp_path: Path) -> None:
    shard_path = _shard(tmp_path, count=4)
    dagger = train_hnf_dagger(
        HNFDaggerConfig(
            data_path=shard_path,
            iterations=1,
            train_steps=1,
            batch_size=2,
            out_dir=tmp_path / "dagger",
            hidden_sizes=(16,),
            rollout_sample_count=2,
            rollout_max_steps=1,
        )
    )

    result = train_hnf_actor_critic(
        HNFActorCriticConfig(
            data_path=dagger["aggregate_data_path"],
            checkpoint_dir=dagger["checkpoint_dir"],
            steps=1,
            batch_size=2,
            out_dir=tmp_path / "actor_critic",
            hidden_sizes=(16,),
            rollout_max_steps=2,
        )
    )

    assert result["status"] == "ok"
    assert result["final_step"] >= 1
    assert result["parameters_changed"]
    assert "mean_reward" in result["metrics"]
    assert "policy_loss" in result["metrics"]
    assert "value_loss" in result["metrics"]
