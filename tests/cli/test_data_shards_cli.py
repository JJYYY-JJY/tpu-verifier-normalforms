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
