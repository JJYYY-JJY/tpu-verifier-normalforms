from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nf_agent.data.rref_backward_shards import write_rref_backward_shard
from nf_agent.data.rref_state_shards import write_rref_state_shard
from nf_agent.train import (
    RREFMatrixFormerTrainConfig,
    restore_latest_rref_matrixformer_checkpoint,
    train_rref_matrixformer,
)


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


def _small_config(data_path: Path, out_dir: Path, *, steps: int = 2) -> RREFMatrixFormerTrainConfig:
    return RREFMatrixFormerTrainConfig(
        data_path=data_path,
        steps=steps,
        batch_size=4,
        learning_rate=0.001,
        seed=0,
        out_dir=out_dir,
        row_embedding_dim=8,
        col_embedding_dim=8,
        hidden_dim=32,
        layers=1,
        num_heads=1,
    )


def test_train_rref_matrixformer_saves_checkpoint_and_reports_metrics(tmp_path: Path) -> None:
    shard_path = _state_shard(tmp_path)
    out_dir = tmp_path / "ckpt"

    result = train_rref_matrixformer(_small_config(shard_path, out_dir))

    assert result["status"] == "ok"
    assert result["model"] == "rref-matrixformer"
    assert result["final_step"] == 2
    assert result["latest_step"] == 2
    assert np.isfinite(result["final_loss"])
    assert result["parameters_changed"]
    assert Path(result["checkpoint_dir"]).exists()
    assert result["data_schema_version"] == "rref-state-action-npz-v1"
    assert set(result["per_head_metrics"]) == {
        "action_kind_loss",
        "action_target_loss",
        "action_source_loss",
        "action_scalar_loss",
    }


def test_restore_latest_rref_matrixformer_checkpoint_continues_step_count(tmp_path: Path) -> None:
    shard_path = _state_shard(tmp_path)
    out_dir = tmp_path / "ckpt"
    config = _small_config(shard_path, out_dir)
    train_rref_matrixformer(config)

    resumed = train_rref_matrixformer(_small_config(shard_path, out_dir, steps=1))
    state = restore_latest_rref_matrixformer_checkpoint(config)

    assert resumed["final_step"] == 3
    assert int(state.step) == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"steps": 0}, "steps must be positive"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"learning_rate": 0.0}, "learning_rate must be positive"),
        ({"hidden_dim": 0}, "hidden_dim must be positive"),
        ({"num_heads": 0}, "num_heads must be positive"),
        ({"data_path": Path("missing.npz")}, "data path does not exist"),
    ],
)
def test_train_rref_matrixformer_rejects_invalid_config(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    shard_path = _state_shard(tmp_path)
    config_kwargs: dict[str, object] = {
        "data_path": shard_path,
        "steps": 1,
        "batch_size": 2,
        "out_dir": tmp_path / "ckpt",
        "row_embedding_dim": 4,
        "col_embedding_dim": 4,
        "hidden_dim": 16,
        "layers": 1,
        "num_heads": 1,
    }
    config_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        train_rref_matrixformer(RREFMatrixFormerTrainConfig(**config_kwargs))


def test_train_rref_matrixformer_rejects_wrong_state_schema(tmp_path: Path) -> None:
    good_path = _state_shard(tmp_path)
    bad_path = tmp_path / "wrong_schema.npz"
    with np.load(good_path, allow_pickle=False) as shard:
        arrays = {key: np.asarray(shard[key]) for key in shard.files}
    metadata = json.loads(str(arrays["metadata_json"]))
    metadata["schema_version"] = "wrong-schema"
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez(bad_path, **arrays)

    with pytest.raises(ValueError, match="unsupported RREF state-action shard schema_version"):
        train_rref_matrixformer(_small_config(bad_path, tmp_path / "ckpt"))
