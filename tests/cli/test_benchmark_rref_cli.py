import json
from pathlib import Path

from click.testing import CliRunner

from nf_agent.cli import main
from nf_agent.data.rref_shards import write_rref_shard
from nf_agent.train import TrainConfig, train_rref_pivot

CONFIG = Path("configs/rref_8x8_mod101.yaml").resolve()


def test_benchmark_rref_generated_cli_emits_compact_json() -> None:
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "benchmark",
            "rref",
            "--source",
            "generated",
            "--rows",
            "4",
            "--cols",
            "4",
            "--p",
            "101",
            "--family",
            "dense",
            "--count",
            "4",
            "--seed-start",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["source"] == "generated"
    assert payload["policies"]["leftmost"]["aggregate"]["success_rate"] == 1.0
    sample = payload["policies"]["leftmost"]["samples"][0]
    assert "initial_matrix" not in sample
    assert "ops" not in sample
    assert {"trace_length", "fill_in_delta", "replay_ok", "final_is_rref"} <= set(sample)


def test_benchmark_rref_shard_neural_cli_runs_tiny_checkpoint(tmp_path: Path) -> None:
    shard_path = tmp_path / "rref_cli_bench.npz"
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
            "benchmark",
            "rref",
            "--source",
            "shard",
            "--data",
            str(shard_path),
            "--count",
            "2",
            "--checkpoint",
            str(ckpt_dir),
            "--max-steps",
            "2",
            "--hidden-size",
            "32",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["source"] == "shard"
    assert payload["count"] == 2
    assert "neural" in payload["policies"]
    assert sum(payload["policies"]["neural"]["aggregate"]["status_counts"].values()) == 2


def test_benchmark_rref_cli_rejects_invalid_args(tmp_path: Path) -> None:
    runner = CliRunner()

    generated_missing_shape = runner.invoke(
        main,
        ["benchmark", "rref", "--source", "generated", "--count", "1"],
    )
    assert generated_missing_shape.exit_code != 0
    assert (
        "rows, cols, and count are required for generated source"
        in generated_missing_shape.output
    )

    shard_missing_data = runner.invoke(main, ["benchmark", "rref", "--source", "shard"])
    assert shard_missing_data.exit_code != 0
    assert "data_path is required for shard source" in shard_missing_data.output

    generated_missing_model_data = runner.invoke(
        main,
        [
            "benchmark",
            "rref",
            "--source",
            "generated",
            "--rows",
            "4",
            "--cols",
            "4",
            "--count",
            "1",
            "--checkpoint",
            str(tmp_path / "ckpt"),
        ],
    )
    assert generated_missing_model_data.exit_code != 0
    assert "model_data_path is required when checkpoint_dir is provided" in (
        generated_missing_model_data.output
    )

    shard_path = tmp_path / "rref_cli_model_data.npz"
    write_rref_shard(config_path=CONFIG, count=2, seed_start=0, out_path=shard_path)
    mismatch = runner.invoke(
        main,
        [
            "benchmark",
            "rref",
            "--source",
            "generated",
            "--rows",
            "4",
            "--cols",
            "4",
            "--count",
            "1",
            "--model-data",
            str(shard_path),
            "--checkpoint",
            str(tmp_path / "ckpt"),
        ],
    )
    assert mismatch.exit_code != 0
    assert "model data shape/modulus must match generated benchmark config" in mismatch.output
