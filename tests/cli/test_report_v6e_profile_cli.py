import json
from pathlib import Path

from click.testing import CliRunner

from nf_agent.cli import main


def _write_profile(path: Path, extra: dict[str, object] | None = None) -> None:
    payload: dict[str, object] = {
        "schema_version": "v6e-status-v1",
        "status": "ok",
        "jax": {"backend": "cpu", "local_device_count": 1, "devices": []},
        "versions": {"python": "3.12", "jax": "0", "jaxlib": "0"},
        "host": {"platform": "test", "ram": {"total_bytes": 1, "available_bytes": 1}},
        "hbm": {"status": "unavailable", "total_bytes": None, "used_bytes": None},
        "env": {},
        "no_fallback_statement": "No hidden teacher fallback.",
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_v6e_profile_writes_compact_summary_and_markdown(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    out_dir = tmp_path / "report"
    _write_profile(profile_path)

    result = CliRunner().invoke(
        main,
        ["report", "v6e-profile", "--input", str(profile_path), "--out-dir", str(out_dir)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"
    summary = json.loads(summary_path.read_text())
    report = report_path.read_text()
    assert payload["summary_json"] == str(summary_path)
    assert payload["report_md"] == str(report_path)
    assert summary["schema_version"] == "v6e-profile-report-v1"
    assert summary["backend"] == "cpu"
    assert "No hidden teacher fallback" in report


def test_report_v6e_profile_rejects_raw_matrix_ops_logs_or_checkpoint_paths(tmp_path: Path) -> None:
    forbidden = {
        "matrix": [[1]],
        "ops": [],
        "raw_logs": "full log",
        "checkpoint_path": "/tmp/ckpt",
    }
    for key, value in forbidden.items():
        profile_path = tmp_path / f"{key}.json"
        _write_profile(profile_path, {key: value})

        result = CliRunner().invoke(
            main,
            [
                "report",
                "v6e-profile",
                "--input",
                str(profile_path),
                "--out-dir",
                str(tmp_path / key),
            ],
        )

        assert result.exit_code != 0
        assert "forbidden compact report key" in result.output
