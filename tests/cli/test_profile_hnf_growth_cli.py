import json
from pathlib import Path

from click.testing import CliRunner

from nf_agent.cli import main


def _assert_no_raw_fields(value: object) -> None:
    if isinstance(value, dict):
        forbidden = {"input", "final", "matrix", "ops", "input_matrix", "final_matrix"}
        assert forbidden.isdisjoint(value)
        for item in value.values():
            _assert_no_raw_fields(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_raw_fields(item)


def test_profile_hnf_growth_writes_compact_summary_and_report(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    out_dir = tmp_path / "report"

    result = CliRunner().invoke(
        main,
        [
            "profile",
            "hnf-growth",
            "--config",
            "configs/v6e1/hnf_growth_search.yaml",
            "--work-dir",
            str(work_dir),
            "--out-dir",
            str(out_dir),
            "--family",
            "sparse_8x8",
            "--count",
            "8",
            "--candidate-limit",
            "64",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["schema_version"] == "hnf-growth-profile-v1"
    assert payload["family"] == "sparse_8x8"
    assert Path(payload["summary_json"]) == out_dir / "summary.json"
    assert Path(payload["report_md"]) == out_dir / "report.md"
    assert Path(payload["shard_path"]).suffix == ".zarr"

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["source_schema_version"] == "hnf-backward-trace-zarr-v1"
    assert summary["profile_version"] == "v1.1-beta"
    assert summary["search"]["mode"] == "exact_row_preconditioned"
    assert summary["search"]["candidate_limit"] == 64
    assert summary["aggregate"]["sample_count"] == 8
    assert summary["aggregate"]["success_rate"] == 1.0
    assert summary["aggregate"]["improved_sample_count"] >= 1
    assert summary["aggregate"]["improvement_rate"] > 0.0
    assert summary["aggregate"]["improved_metric_counts"]["max_bitlength"] >= 1
    assert summary["aggregate"]["improved_metric_counts"]["max_abs_seen"] >= 1
    assert summary["aggregate"]["v1_1_target_met"] is True
    assert summary["aggregate"]["wall_time_seconds"] >= 0.0
    assert len(summary["samples"]) == 8
    for sample in summary["samples"]:
        assert sample["status"] == "success"
        assert sample["replay_ok"] is True
        assert sample["predicate_ok"] is True
        assert sample["best_policy"] in {"row_hnf", "row_preconditioned_row_hnf"}
        assert isinstance(sample["best_candidate"], int)
        assert sample["candidate_count"] == 64
        assert sample["rejected_candidate_count"] == 0
        assert set(sample["improved_metrics"]) <= {
            "max_bitlength",
            "max_abs_seen",
            "fill_in_delta",
            "step_count",
            "certificate_size_entries",
        }
        assert sample["baseline"]["policy"] == "row_hnf"
        assert sample["best"]["policy"] == sample["best_policy"]
        assert "step_count" in sample
        assert "max_bitlength" in sample
        assert "max_abs_seen" in sample
        assert "fill_in_delta" in sample
        assert "certificate_op_count" in sample
        assert "certificate_size_entries" in sample
        assert "input" not in sample
        assert "final" not in sample
        assert "matrix" not in sample
        assert "ops" not in sample
    _assert_no_raw_fields(summary)

    report = (out_dir / "report.md").read_text()
    assert "HNF Growth Profile" in report
    assert "v1.1 beta target: `met`" in report
    assert "row_preconditioned_row_hnf" in report
    assert "sparse_8x8" in report
    assert "Wall time seconds" in report
    assert "[[" not in json.dumps(summary)
