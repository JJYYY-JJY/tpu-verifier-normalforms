import json

from click.testing import CliRunner

from nf_agent.cli import main


def test_benchmark_snf_cli_emits_compact_json() -> None:
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "benchmark",
            "snf",
            "--rows",
            "3",
            "--cols",
            "3",
            "--count",
            "2",
            "--diagonal-factor-bound",
            "5",
            "--row-op-count",
            "2",
            "--col-op-count",
            "1",
            "--op-scalar-bound",
            "3",
            "--seed-start",
            "7",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["source"] == "generated"
    assert payload["family"] == "snf_certificate"
    assert payload["count"] == 2
    assert payload["policies"]["certificate_replay"]["aggregate"]["success_rate"] == 1.0
    sample = payload["policies"]["certificate_replay"]["samples"][0]
    assert {"operation_count", "replay_ok", "verified", "fill_in_delta"} <= set(sample)
    assert "input" not in sample
    assert "diagonal" not in sample
    assert "left_transform" not in sample
    assert "right_transform" not in sample
    assert "row_ops" not in sample
    assert "col_ops" not in sample


def test_benchmark_snf_cli_rejects_invalid_args() -> None:
    runner = CliRunner()

    invalid_count = runner.invoke(
        main,
        ["benchmark", "snf", "--rows", "2", "--cols", "2", "--count", "0"],
    )
    assert invalid_count.exit_code != 0
    assert "count must be positive" in invalid_count.output

    invalid_scalar_bound = runner.invoke(
        main,
        [
            "benchmark",
            "snf",
            "--rows",
            "2",
            "--cols",
            "2",
            "--count",
            "1",
            "--op-scalar-bound",
            "0",
        ],
    )
    assert invalid_scalar_bound.exit_code != 0
    assert "op_scalar_bound must be positive" in invalid_scalar_bound.output
