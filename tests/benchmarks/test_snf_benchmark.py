from __future__ import annotations

import pytest

from nf_agent.benchmarks import snf_benchmark
from nf_agent.benchmarks.snf_benchmark import SNFBenchmarkConfig, run_snf_benchmark
from nf_agent.certificates import SNFCertificate, verify_snf_certificate


def test_generated_snf_certificate_benchmark_runs_exact_verification() -> None:
    result = run_snf_benchmark(
        SNFBenchmarkConfig(
            count=4,
            rows=4,
            cols=3,
            diagonal_factor_bound=5,
            row_op_count=3,
            col_op_count=2,
            op_scalar_bound=3,
            seed_start=11,
        )
    )

    aggregate = result["policies"]["certificate_replay"]["aggregate"]
    samples = result["policies"]["certificate_replay"]["samples"]
    assert result["status"] == "ok"
    assert result["source"] == "generated"
    assert result["family"] == "snf_certificate"
    assert result["count"] == 4
    assert result["rows"] == 4
    assert result["cols"] == 3
    assert aggregate["sample_count"] == 4
    assert aggregate["success_count"] == 4
    assert aggregate["success_rate"] == 1.0
    assert aggregate["status_counts"] == {"success": 4}
    assert len(samples) == 4
    assert all(sample["replay_ok"] for sample in samples)
    assert all(sample["verified"] for sample in samples)
    assert all(sample["row_op_count"] == 3 for sample in samples)
    assert all(sample["col_op_count"] == 2 for sample in samples)
    assert all(sample["operation_count"] == 5 for sample in samples)
    assert "mean_operation_count" in aggregate
    assert "mean_initial_density" in aggregate
    assert "mean_fill_in_delta" in aggregate
    assert "max_max_abs_seen" in aggregate
    assert "max_max_bitlength" in aggregate


def test_snf_benchmark_output_omits_full_matrices_transforms_and_ops() -> None:
    result = run_snf_benchmark(
        SNFBenchmarkConfig(count=1, rows=3, cols=3, row_op_count=2, col_op_count=2)
    )

    sample = result["policies"]["certificate_replay"]["samples"][0]
    forbidden = {
        "input",
        "diagonal",
        "left_transform",
        "right_transform",
        "row_ops",
        "col_ops",
        "ops",
        "matrix",
    }
    assert forbidden.isdisjoint(sample)


def test_generated_snf_certificates_verify_through_existing_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified_shapes: list[tuple[int, int]] = []

    def record_verify(certificate: SNFCertificate) -> None:
        verify_snf_certificate(certificate)
        verified_shapes.append(certificate.shape)

    monkeypatch.setattr(snf_benchmark, "verify_snf_certificate", record_verify)

    result = run_snf_benchmark(
        SNFBenchmarkConfig(count=2, rows=3, cols=4, row_op_count=2, col_op_count=3)
    )

    assert result["policies"]["certificate_replay"]["aggregate"]["success_rate"] == 1.0
    assert verified_shapes == [(3, 4), (3, 4)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"count": 0, "rows": 2, "cols": 2}, "count must be positive"),
        ({"count": 1, "rows": 0, "cols": 2}, "rows must be positive"),
        ({"count": 1, "rows": 2, "cols": 0}, "cols must be positive"),
        (
            {"count": 1, "rows": 2, "cols": 2, "diagonal_factor_bound": 0},
            "diagonal_factor_bound",
        ),
        (
            {"count": 1, "rows": 2, "cols": 2, "row_op_count": -1},
            "row_op_count",
        ),
        (
            {"count": 1, "rows": 2, "cols": 2, "col_op_count": -1},
            "col_op_count",
        ),
        (
            {"count": 1, "rows": 2, "cols": 2, "op_scalar_bound": 0},
            "op_scalar_bound",
        ),
    ],
)
def test_snf_benchmark_rejects_invalid_config(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_snf_benchmark(SNFBenchmarkConfig(**kwargs))
