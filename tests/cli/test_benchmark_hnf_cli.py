import json

from click.testing import CliRunner

from nf_agent.cli import main


def test_benchmark_hnf_cli_emits_compact_json() -> None:
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "benchmark",
            "hnf",
            "--rows",
            "3",
            "--cols",
            "3",
            "--count",
            "2",
            "--density",
            "0.4",
            "--entry-bound",
            "5",
            "--seed-start",
            "7",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["source"] == "generated"
    assert payload["family"] == "sparse_integer"
    assert payload["count"] == 2
    assert payload["aggregate"]["success_rate"] == 1.0
    sample = payload["samples"][0]
    assert "input" not in sample
    assert "final_matrix" not in sample
    assert "ops" not in sample
    assert {"trace_length", "fill_in_delta", "replay_ok", "final_is_hnf"} <= set(sample)


def test_benchmark_hnf_cli_rejects_invalid_args() -> None:
    runner = CliRunner()

    invalid_count = runner.invoke(
        main,
        ["benchmark", "hnf", "--rows", "2", "--cols", "2", "--count", "0"],
    )
    assert invalid_count.exit_code != 0
    assert "count must be positive" in invalid_count.output

    invalid_density = runner.invoke(
        main,
        [
            "benchmark",
            "hnf",
            "--rows",
            "2",
            "--cols",
            "2",
            "--count",
            "1",
            "--density",
            "1.1",
        ],
    )
    assert invalid_density.exit_code != 0
    assert "density must lie in [0, 1]" in invalid_density.output
