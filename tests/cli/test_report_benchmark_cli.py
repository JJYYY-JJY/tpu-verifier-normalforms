import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from nf_agent.benchmarks import (
    HNFBenchmarkConfig,
    RREFBenchmarkConfig,
    run_hnf_benchmark,
    run_rref_benchmark,
)
from nf_agent.cli import main


def test_report_status_lists_implemented_report_commands() -> None:
    result = CliRunner().invoke(main, ["report", "status"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "implemented"
    assert payload["commands"] == ["benchmark", "rref-certificate"]


def test_report_benchmark_run_mode_writes_markdown_metrics_and_plots(tmp_path: Path) -> None:
    out_dir = tmp_path / "report"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "report",
            "benchmark",
            "--out-dir",
            str(out_dir),
            "--sample-count",
            "2",
            "--rows",
            "3",
            "--cols",
            "3",
            "--p",
            "5",
            "--seed-start",
            "0",
            "--sparse-density",
            "0.3",
            "--low-rank",
            "2",
            "--hnf-entry-bound",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["mode"] == "run"
    assert Path(payload["report_md"]) == out_dir / "report.md"
    assert Path(payload["metrics_json"]) == out_dir / "metrics.json"

    assert (out_dir / "report.md").is_file()
    metrics_path = out_dir / "metrics.json"
    assert metrics_path.is_file()
    metrics = json.loads(metrics_path.read_text())
    rows = metrics["normalized_rows"]
    assert any(row["kind"] == "rref" and row["policy"] == "leftmost" for row in rows)
    assert any(row["kind"] == "hnf" and row["policy"] == "row_hnf" for row in rows)
    assert all("ops" not in sample for item in metrics["benchmarks"] for sample in _samples(item))

    for relative_plot in metrics["artifacts"]["plots"].values():
        plot_path = out_dir / relative_plot
        assert plot_path.is_file()
        assert plot_path.stat().st_size > 0


def test_report_benchmark_summary_mode_consumes_existing_compact_json_without_rerun(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    rref_json = tmp_path / "rref.json"
    hnf_json = tmp_path / "hnf.json"
    rref_json.write_text(
        json.dumps(
            run_rref_benchmark(
                RREFBenchmarkConfig(
                    source="generated",
                    count=2,
                    rows=3,
                    cols=3,
                    modulus=5,
                    family="dense",
                    seed_start=0,
                )
            )
        )
    )
    hnf_json.write_text(
        json.dumps(
            run_hnf_benchmark(
                HNFBenchmarkConfig(
                    count=2,
                    rows=3,
                    cols=3,
                    density=0.4,
                    entry_bound=5,
                    seed_start=0,
                )
            )
        )
    )

    def fail_rerun(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("summary mode must not rerun benchmarks")

    monkeypatch.setattr("nf_agent.reports.benchmark_report.run_rref_benchmark", fail_rerun)
    monkeypatch.setattr("nf_agent.reports.benchmark_report.run_hnf_benchmark", fail_rerun)

    out_dir = tmp_path / "summary-report"
    result = CliRunner().invoke(
        main,
        [
            "report",
            "benchmark",
            "--out-dir",
            str(out_dir),
            "--input-json",
            str(rref_json),
            "--input-json",
            str(hnf_json),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "summary"
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert [item["source_path"] for item in metrics["benchmarks"]] == [
        str(rref_json),
        str(hnf_json),
    ]
    assert len(metrics["normalized_rows"]) == 2


def test_report_benchmark_rejects_unknown_input_json(tmp_path: Path) -> None:
    bad_json = tmp_path / "unknown.json"
    bad_json.write_text(json.dumps({"status": "ok", "kind": "rref_modp"}))

    result = CliRunner().invoke(
        main,
        [
            "report",
            "benchmark",
            "--out-dir",
            str(tmp_path / "report"),
            "--input-json",
            str(bad_json),
        ],
    )

    assert result.exit_code != 0
    assert "unsupported benchmark JSON" in result.output


def _samples(item: dict[str, Any]) -> list[dict[str, Any]]:
    payload = item["payload"]
    if item["kind"] == "rref":
        return [
            sample
            for policy in payload["policies"].values()
            for sample in policy["samples"]
        ]
    return list(payload["samples"])
