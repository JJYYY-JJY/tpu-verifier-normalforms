import json
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from nf_agent.cli import main

CONFIG = Path("configs/rref_8x8_mod101.yaml").resolve()


def test_make_rref_shard_cli_writes_schema_npz() -> None:
    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(
            main,
            [
                "data",
                "make-rref-shard",
                "--config",
                str(CONFIG),
                "--count",
                "2",
                "--seed-start",
                "3",
                "--out",
                "out/shard.npz",
            ],
        )

        assert result.exit_code == 0, result.output
        with np.load("out/shard.npz", allow_pickle=False) as shard:
            assert shard["inputs"].shape == (2, 8, 8)
            assert shard["op_kind"].shape == (2, 72)
            metadata = json.loads(str(shard["metadata_json"]))

    assert metadata["config"]["field"]["modulus"] == 101
    assert metadata["config"]["matrix"]["family"] == "dense"
    assert metadata["count"] == 2


def test_make_rref_shard_cli_rejects_invalid_args() -> None:
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "data",
            "make-rref-shard",
            "--config",
            "configs/rref_8x8_mod101.yaml",
            "--count",
            "0",
            "--seed-start",
            "0",
            "--out",
            "shard.npz",
        ],
    )

    assert result.exit_code != 0
    assert "count must be positive" in result.output


def test_make_rref_backward_shard_cli_writes_schema_npz(tmp_path: Path) -> None:
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
        "  max_backward_ops: 5\n"
        "  require_exact_replay: true\n"
    )
    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(
            main,
            [
                "data",
                "make-rref-backward-shard",
                "--config",
                str(config_path),
                "--count",
                "2",
                "--seed-start",
                "3",
                "--out",
                "out/backward.npz",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["status"] == "ok"
        assert payload["schema_version"] == "rref-backward-trace-npz-v1"
        with np.load("out/backward.npz", allow_pickle=False) as shard:
            assert shard["inputs"].shape == (2, 4, 4)
            assert shard["ops"].shape == (2, 5, 4)
            metadata = json.loads(str(shard["metadata_json"]))

    assert metadata["schema_version"] == "rref-backward-trace-npz-v1"
    assert metadata["count"] == 2


def test_make_rref_backward_shard_cli_rejects_invalid_args() -> None:
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "data",
            "make-rref-backward-shard",
            "--config",
            "configs/v6e1/rref_backward_state_shards.yaml",
            "--count",
            "0",
            "--seed-start",
            "0",
            "--out",
            "backward.npz",
        ],
    )

    assert result.exit_code != 0
    assert "count must be positive" in result.output


def test_make_rref_state_shard_cli_writes_schema_npz(tmp_path: Path) -> None:
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
        "  max_backward_ops: 5\n"
        "  require_exact_replay: true\n"
    )
    runner = CliRunner()

    with runner.isolated_filesystem():
        backward_result = runner.invoke(
            main,
            [
                "data",
                "make-rref-backward-shard",
                "--config",
                str(config_path),
                "--count",
                "2",
                "--seed-start",
                "3",
                "--out",
                "out/backward.npz",
            ],
        )
        assert backward_result.exit_code == 0, backward_result.output

        result = runner.invoke(
            main,
            [
                "data",
                "make-rref-state-shard",
                "--trace-shard",
                "out/backward.npz",
                "--out",
                "out/state.npz",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["status"] == "ok"
        assert payload["schema_version"] == "rref-state-action-npz-v1"
        assert payload["out"] == "out/state.npz"
        assert payload["trace_count"] == 2
        assert payload["flat_count"] == 12
        assert payload["max_steps"] == 6
        with np.load("out/state.npz", allow_pickle=False) as shard:
            metadata = json.loads(str(shard["metadata_json"]))

    assert metadata["schema_version"] == "rref-state-action-npz-v1"
    assert metadata["trace_count"] == 2


def test_make_rref_state_shard_cli_rejects_missing_trace_shard() -> None:
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "data",
            "make-rref-state-shard",
            "--trace-shard",
            "missing.npz",
            "--out",
            "state.npz",
        ],
    )

    assert result.exit_code != 0
    assert "trace shard path does not exist" in result.output


def test_make_rref_state_shard_cli_rejects_non_npz_output(tmp_path: Path) -> None:
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
        "  max_backward_ops: 5\n"
        "  require_exact_replay: true\n"
    )
    runner = CliRunner()

    with runner.isolated_filesystem():
        backward_result = runner.invoke(
            main,
            [
                "data",
                "make-rref-backward-shard",
                "--config",
                str(config_path),
                "--count",
                "1",
                "--seed-start",
                "0",
                "--out",
                "out/backward.npz",
            ],
        )
        assert backward_result.exit_code == 0, backward_result.output

        result = runner.invoke(
            main,
            [
                "data",
                "make-rref-state-shard",
                "--trace-shard",
                "out/backward.npz",
                "--out",
                "out/state.zip",
            ],
        )

    assert result.exit_code != 0
    assert "output path must end with .npz" in result.output
