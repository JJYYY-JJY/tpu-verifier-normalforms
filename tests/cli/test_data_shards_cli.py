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
