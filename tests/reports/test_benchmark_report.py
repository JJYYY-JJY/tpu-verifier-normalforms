import json
from pathlib import Path
from typing import Any

from nf_agent.reports.benchmark_report import BenchmarkReportConfig, build_benchmark_report


def test_benchmark_report_renderer_writes_required_sections_rows_and_plots(
    tmp_path: Path,
) -> None:
    rref_json = tmp_path / "rref_neural.json"
    hnf_json = tmp_path / "hnf.json"
    snf_json = tmp_path / "snf.json"
    rref_json.write_text(json.dumps(_rref_payload_with_failed_neural()))
    hnf_json.write_text(json.dumps(_hnf_payload()))
    snf_json.write_text(json.dumps(_snf_payload()))

    result = build_benchmark_report(
        BenchmarkReportConfig(
            out_dir=tmp_path / "report",
            input_json_paths=(rref_json, hnf_json, snf_json),
        )
    )

    assert result["status"] == "ok"
    report_text = (tmp_path / "report" / "report.md").read_text()
    assert "# v0.9 Benchmark Report" in report_text
    assert "## Provenance" in report_text
    assert "## Exactness and Fallback Policy" in report_text
    assert "failed neural rollouts remain failures" in report_text
    assert "## Suite" in report_text
    assert "## Correctness" in report_text
    assert "## Timing" in report_text
    assert "## Fill-In" in report_text
    assert "## HNF Coefficient Growth" in report_text
    assert "## Plots" in report_text
    assert "## Limitations" in report_text
    assert "SNF benchmark and report coverage is out of scope" not in report_text
    assert "SNF benchmark uses generated certificates" in report_text

    metrics = json.loads((tmp_path / "report" / "metrics.json").read_text())
    policies = {(row["kind"], row["policy"]) for row in metrics["normalized_rows"]}
    assert ("rref", "leftmost") in policies
    assert ("rref", "neural") in policies
    assert ("hnf", "row_hnf") in policies
    assert ("snf", "certificate_replay") in policies
    neural = next(row for row in metrics["normalized_rows"] if row["policy"] == "neural")
    assert neural["success_count"] == 0
    assert neural["status_counts"] == {"max_steps_exceeded": 2}
    assert metrics["hnf_coefficient_rows"][0]["max_bitlength"] == 4
    assert "neural_invalid_actions" in metrics["artifacts"]["plots"]

    for relative_plot in metrics["artifacts"]["plots"].values():
        plot_path = tmp_path / "report" / relative_plot
        assert plot_path.is_file()
        assert plot_path.stat().st_size > 0


def test_benchmark_report_rejects_non_compact_samples(tmp_path: Path) -> None:
    rref_json = tmp_path / "not_compact.json"
    payload = _rref_payload_with_failed_neural()
    payload["policies"]["leftmost"]["samples"][0]["ops"] = []
    rref_json.write_text(json.dumps(payload))

    try:
        build_benchmark_report(
            BenchmarkReportConfig(
                out_dir=tmp_path / "report",
                input_json_paths=(rref_json,),
            )
        )
    except ValueError as exc:
        assert "compact benchmark JSON must not include sample field: ops" in str(exc)
    else:
        raise AssertionError("expected non-compact benchmark JSON to be rejected")


def test_benchmark_report_rejects_unknown_snf_payload(tmp_path: Path) -> None:
    snf_json = tmp_path / "unknown_snf.json"
    snf_json.write_text(json.dumps({"status": "ok", "family": "snf_certificate"}))

    try:
        build_benchmark_report(
            BenchmarkReportConfig(
                out_dir=tmp_path / "report",
                input_json_paths=(snf_json,),
            )
        )
    except ValueError as exc:
        assert "unsupported benchmark JSON" in str(exc)
    else:
        raise AssertionError("expected unknown SNF benchmark JSON to be rejected")


def _rref_payload_with_failed_neural() -> dict[str, Any]:
    return {
        "status": "ok",
        "source": "generated",
        "count": 2,
        "rows": 3,
        "cols": 3,
        "modulus": 5,
        "policies": {
            "leftmost": {
                "aggregate": {
                    "sample_count": 2,
                    "success_count": 2,
                    "success_rate": 1.0,
                    "status_counts": {"success": 2},
                    "mean_trace_length": 3.5,
                    "mean_rank": 2.0,
                    "mean_initial_density": 0.5,
                    "mean_final_density": 0.3,
                    "mean_max_density": 0.6,
                    "mean_fill_in_delta": 0.1,
                    "mean_wall_time_seconds": 0.01,
                    "mean_teacher_wall_time_seconds": 0.006,
                    "mean_replay_wall_time_seconds": 0.002,
                    "mean_predicate_wall_time_seconds": 0.002,
                },
                "samples": [
                    {
                        "sample_index": 0,
                        "seed": 0,
                        "status": "success",
                        "success": True,
                        "trace_length": 3,
                        "rank": 2,
                        "replay_ok": True,
                        "final_is_rref": True,
                        "initial_density": 0.5,
                        "final_density": 0.3,
                        "max_density": 0.6,
                        "fill_in_delta": 0.1,
                        "wall_time_seconds": 0.01,
                    },
                    {
                        "sample_index": 1,
                        "seed": 1,
                        "status": "success",
                        "success": True,
                        "trace_length": 4,
                        "rank": 2,
                        "replay_ok": True,
                        "final_is_rref": True,
                        "initial_density": 0.4,
                        "final_density": 0.3,
                        "max_density": 0.5,
                        "fill_in_delta": 0.1,
                        "wall_time_seconds": 0.01,
                    },
                ],
            },
            "neural": {
                "aggregate": {
                    "sample_count": 2,
                    "success_count": 0,
                    "success_rate": 0.0,
                    "status_counts": {"max_steps_exceeded": 2},
                    "mean_step_count": 2.0,
                    "mean_invalid_action_count": 1.5,
                    "mean_masked_action_count": 0.0,
                    "mean_initial_density": 0.5,
                    "mean_final_density": 0.5,
                    "mean_max_density": 0.7,
                    "mean_fill_in_delta": 0.2,
                    "mean_wall_time_seconds": 0.02,
                    "mean_rollout_wall_time_seconds": 0.018,
                    "mean_replay_wall_time_seconds": 0.001,
                    "mean_predicate_wall_time_seconds": 0.001,
                },
                "samples": [
                    {
                        "sample_index": 0,
                        "seed": 0,
                        "status": "max_steps_exceeded",
                        "success": False,
                        "step_count": 2,
                        "invalid_action_count": 1,
                        "masked_action_count": 0,
                        "invalid_action_breakdown": {"op_kind": 1},
                        "checkpoint_step": 2,
                        "replay_ok": True,
                        "final_is_rref": False,
                        "initial_density": 0.5,
                        "final_density": 0.5,
                        "max_density": 0.7,
                        "fill_in_delta": 0.2,
                        "wall_time_seconds": 0.02,
                    },
                    {
                        "sample_index": 1,
                        "seed": 1,
                        "status": "max_steps_exceeded",
                        "success": False,
                        "step_count": 2,
                        "invalid_action_count": 2,
                        "masked_action_count": 0,
                        "invalid_action_breakdown": {"op_source": 2},
                        "checkpoint_step": 2,
                        "replay_ok": True,
                        "final_is_rref": False,
                        "initial_density": 0.5,
                        "final_density": 0.5,
                        "max_density": 0.7,
                        "fill_in_delta": 0.2,
                        "wall_time_seconds": 0.02,
                    },
                ],
            },
        },
    }


def _hnf_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "source": "generated",
        "family": "sparse_integer",
        "count": 2,
        "rows": 3,
        "cols": 3,
        "density": 0.4,
        "entry_bound": 5,
        "seed_start": 0,
        "aggregate": {
            "sample_count": 2,
            "success_count": 2,
            "success_rate": 1.0,
            "status_counts": {"success": 2},
            "mean_trace_length": 2.5,
            "mean_initial_density": 0.4,
            "mean_final_density": 0.3,
            "mean_max_density": 0.5,
            "mean_fill_in_delta": 0.1,
            "mean_wall_time_seconds": 0.01,
            "mean_hnf_wall_time_seconds": 0.006,
            "mean_replay_wall_time_seconds": 0.002,
            "mean_predicate_wall_time_seconds": 0.002,
            "max_initial_max_abs": 5,
            "max_max_abs_seen": 9,
            "max_initial_bitlength": 3,
            "max_max_bitlength": 4,
            "max_growth_numerator": 9,
            "max_growth_denominator": 5,
            "max_step_count": 3,
        },
        "samples": [
            {
                "sample_index": 0,
                "seed": 0,
                "status": "success",
                "success": True,
                "trace_length": 2,
                "replay_ok": True,
                "final_is_hnf": True,
                "initial_density": 0.4,
                "final_density": 0.3,
                "max_density": 0.5,
                "fill_in_delta": 0.1,
                "initial_max_abs": 5,
                "max_abs_seen": 9,
                "initial_bitlength": 3,
                "max_bitlength": 4,
                "growth_numerator": 9,
                "growth_denominator": 5,
                "step_count": 2,
                "wall_time_seconds": 0.01,
            },
            {
                "sample_index": 1,
                "seed": 1,
                "status": "success",
                "success": True,
                "trace_length": 3,
                "replay_ok": True,
                "final_is_hnf": True,
                "initial_density": 0.4,
                "final_density": 0.3,
                "max_density": 0.5,
                "fill_in_delta": 0.1,
                "initial_max_abs": 4,
                "max_abs_seen": 8,
                "initial_bitlength": 3,
                "max_bitlength": 4,
                "growth_numerator": 8,
                "growth_denominator": 4,
                "step_count": 3,
                "wall_time_seconds": 0.01,
            },
        ],
    }


def _snf_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "source": "generated",
        "family": "snf_certificate",
        "count": 2,
        "rows": 3,
        "cols": 3,
        "diagonal_factor_bound": 5,
        "row_op_count": 2,
        "col_op_count": 2,
        "op_scalar_bound": 3,
        "seed_start": 0,
        "policies": {
            "certificate_replay": {
                "aggregate": {
                    "sample_count": 2,
                    "success_count": 2,
                    "success_rate": 1.0,
                    "status_counts": {"success": 2},
                    "mean_operation_count": 4.0,
                    "mean_initial_density": 0.5,
                    "mean_final_density": 0.3,
                    "mean_max_density": 0.6,
                    "mean_fill_in_delta": 0.1,
                    "mean_wall_time_seconds": 0.01,
                    "mean_replay_wall_time_seconds": 0.004,
                    "mean_verify_wall_time_seconds": 0.006,
                    "max_initial_max_abs": 9,
                    "max_max_abs_seen": 12,
                    "max_initial_bitlength": 4,
                    "max_max_bitlength": 4,
                },
                "samples": [
                    {
                        "sample_index": 0,
                        "seed": 0,
                        "status": "success",
                        "success": True,
                        "row_op_count": 2,
                        "col_op_count": 2,
                        "operation_count": 4,
                        "replay_ok": True,
                        "verified": True,
                        "initial_density": 0.5,
                        "final_density": 0.3,
                        "max_density": 0.6,
                        "fill_in_delta": 0.1,
                        "initial_max_abs": 9,
                        "max_abs_seen": 12,
                        "initial_bitlength": 4,
                        "max_bitlength": 4,
                        "wall_time_seconds": 0.01,
                    },
                    {
                        "sample_index": 1,
                        "seed": 1,
                        "status": "success",
                        "success": True,
                        "row_op_count": 2,
                        "col_op_count": 2,
                        "operation_count": 4,
                        "replay_ok": True,
                        "verified": True,
                        "initial_density": 0.4,
                        "final_density": 0.3,
                        "max_density": 0.5,
                        "fill_in_delta": 0.1,
                        "initial_max_abs": 8,
                        "max_abs_seen": 11,
                        "initial_bitlength": 4,
                        "max_bitlength": 4,
                        "wall_time_seconds": 0.01,
                    },
                ],
            }
        },
    }
