from pathlib import Path

import numpy as np
import pytest

from nf_agent.data.rref_shards import write_rref_shard
from nf_agent.train import (
    TrainConfig,
    restore_latest_rref_pivot_checkpoint,
    train_rref_pivot,
)

CONFIG = Path("configs/rref_8x8_mod101.yaml")


def _shard(tmp_path: Path, count: int = 8) -> Path:
    shard_path = tmp_path / "rref_train_api.npz"
    write_rref_shard(config_path=CONFIG, count=count, seed_start=0, out_path=shard_path)
    return shard_path


def test_train_rref_pivot_saves_checkpoint_and_reports_metrics(tmp_path: Path) -> None:
    shard_path = _shard(tmp_path)
    out_dir = tmp_path / "ckpt"

    result = train_rref_pivot(
        TrainConfig(
            data_path=shard_path,
            steps=2,
            batch_size=4,
            learning_rate=0.001,
            seed=0,
            out_dir=out_dir,
            hidden_sizes=(32,),
        )
    )

    assert result["status"] == "ok"
    assert result["final_step"] == 2
    assert result["latest_step"] == 2
    assert np.isfinite(result["final_loss"])
    assert result["parameters_changed"]
    assert Path(result["checkpoint_dir"]).exists()
    assert result["data_schema_version"] == "rref-teacher-trajectory-npz-v0.2"


def test_restore_latest_rref_pivot_checkpoint_continues_step_count(tmp_path: Path) -> None:
    shard_path = _shard(tmp_path)
    out_dir = tmp_path / "ckpt"
    config = TrainConfig(
        data_path=shard_path,
        steps=2,
        batch_size=4,
        learning_rate=0.001,
        seed=0,
        out_dir=out_dir,
        hidden_sizes=(32,),
    )
    train_rref_pivot(config)

    resumed = train_rref_pivot(
        TrainConfig(
            data_path=shard_path,
            steps=1,
            batch_size=4,
            learning_rate=0.001,
            seed=0,
            out_dir=out_dir,
            hidden_sizes=(32,),
        )
    )
    state = restore_latest_rref_pivot_checkpoint(config)

    assert resumed["final_step"] == 3
    assert int(state.step) == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"steps": 0}, "steps must be positive"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"data_path": Path("missing.npz")}, "data path does not exist"),
    ],
)
def test_train_rref_pivot_rejects_invalid_config(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    shard_path = _shard(tmp_path)
    config_kwargs: dict[str, object] = {
        "data_path": shard_path,
        "steps": 1,
        "batch_size": 2,
        "out_dir": tmp_path / "ckpt",
        "hidden_sizes": (16,),
    }
    config_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        train_rref_pivot(TrainConfig(**config_kwargs))
