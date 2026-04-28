import json
from pathlib import Path

from click.testing import CliRunner

from nf_agent.cli import main
from nf_agent.data.rref_shards import write_rref_shard
from nf_agent.train import TrainConfig, train_rref_pivot

CONFIG = Path("configs/rref_8x8_mod101.yaml").resolve()


def test_rollout_rref_neural_cli_runs_and_emits_json(tmp_path: Path) -> None:
    shard_path = tmp_path / "rref_cli_rollout.npz"
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
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "rollout",
            "rref-neural",
            "--data",
            str(shard_path),
            "--checkpoint",
            str(ckpt_dir),
            "--sample-index",
            "0",
            "--max-steps",
            "2",
            "--hidden-size",
            "32",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] in {"success", "max_steps_exceeded"}
    assert payload["success"] is (payload["status"] == "success")
    assert payload["checkpoint_step"] == 2
    assert payload["modulus"] == 101
    assert "invalid_action_breakdown" in payload
    assert "ops" in payload


def test_rollout_rref_neural_cli_rejects_invalid_args(tmp_path: Path) -> None:
    shard_path = tmp_path / "rref_cli_rollout.npz"
    write_rref_shard(config_path=CONFIG, count=2, seed_start=0, out_path=shard_path)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "rollout",
            "rref-neural",
            "--data",
            str(shard_path),
            "--checkpoint",
            str(tmp_path / "missing_ckpt"),
            "--sample-index",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "no checkpoint found" in result.output
