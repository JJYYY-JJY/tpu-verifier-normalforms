import json
from pathlib import Path

from click.testing import CliRunner

from nf_agent.cli import main
from nf_agent.data.rref_shards import write_rref_shard

CONFIG = Path("configs/rref_8x8_mod101.yaml").resolve()


def test_train_status_lists_implemented_training_commands() -> None:
    result = CliRunner().invoke(main, ["train", "status"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "implemented"
    assert payload["commands"] == [
        "rref-pivot",
        "hnf-policy",
        "hnf-dagger",
        "hnf-actor-critic",
    ]
    assert payload["families"] == ["rref", "hnf"]


def test_train_rref_pivot_cli_runs_and_emits_json(tmp_path: Path) -> None:
    shard_path = tmp_path / "rref_cli.npz"
    write_rref_shard(config_path=CONFIG, count=8, seed_start=0, out_path=shard_path)
    out_dir = tmp_path / "ckpt"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "train",
            "rref-pivot",
            "--data",
            str(shard_path),
            "--steps",
            "2",
            "--batch-size",
            "4",
            "--learning-rate",
            "0.001",
            "--seed",
            "0",
            "--hidden-size",
            "32",
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["final_step"] == 2
    assert payload["latest_step"] == 2
    assert Path(payload["checkpoint_dir"]).exists()
    assert payload["data_schema_version"] == "rref-teacher-trajectory-npz-v0.2"


def test_train_rref_pivot_cli_rejects_invalid_args(tmp_path: Path) -> None:
    shard_path = tmp_path / "rref_cli.npz"
    write_rref_shard(config_path=CONFIG, count=2, seed_start=0, out_path=shard_path)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "train",
            "rref-pivot",
            "--data",
            str(shard_path),
            "--steps",
            "0",
            "--batch-size",
            "2",
            "--out",
            str(tmp_path / "ckpt"),
        ],
    )

    assert result.exit_code != 0
    assert "steps must be positive" in result.output


def test_train_rref_pivot_cli_rejects_bad_data_path(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "train",
            "rref-pivot",
            "--data",
            str(tmp_path / "missing.npz"),
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--out",
            str(tmp_path / "ckpt"),
        ],
    )

    assert result.exit_code != 0
    assert "data path does not exist" in result.output
