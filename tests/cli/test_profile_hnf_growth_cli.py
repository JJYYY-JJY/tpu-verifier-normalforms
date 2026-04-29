import json
from pathlib import Path

from click.testing import CliRunner

from nf_agent.cli import main


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
            "2",
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
    assert summary["aggregate"]["sample_count"] == 2
    assert summary["aggregate"]["success_rate"] == 1.0
    assert summary["aggregate"]["wall_time_seconds"] >= 0.0
    assert len(summary["samples"]) == 2
    for sample in summary["samples"]:
        assert sample["status"] == "success"
        assert sample["replay_ok"] is True
        assert sample["predicate_ok"] is True
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

    report = (out_dir / "report.md").read_text()
    assert "HNF Growth Profile" in report
    assert "sparse_8x8" in report
    assert "Wall time seconds" in report
    assert "[[" not in json.dumps(summary)
