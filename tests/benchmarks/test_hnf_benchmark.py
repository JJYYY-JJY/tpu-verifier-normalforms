from __future__ import annotations

import pytest

from nf_agent.benchmarks.hnf_benchmark import (
    HNFBenchmarkConfig,
    integer_matrix_density,
    integer_row_op_density_profile,
    run_hnf_benchmark,
)
from nf_agent.env.hnf_int import replay_integer_row_ops, row_hnf


def test_integer_matrix_density_counts_exact_integer_nonzeros() -> None:
    assert integer_matrix_density([]) == 0.0
    assert integer_matrix_density([[], []]) == 0.0
    assert integer_matrix_density([[0, 0], [0, 0]]) == 0.0
    assert integer_matrix_density([[0, 2], [-3, 0]]) == 0.5


def test_integer_row_op_density_profile_replays_every_recorded_state() -> None:
    matrix = [[6, 0], [10, 1]]
    result = row_hnf(matrix)

    profile = integer_row_op_density_profile(matrix, result.ops)

    assert len(profile.states) == len(result.ops) + 1
    assert len(profile.densities) == len(result.ops) + 1

    current = matrix
    for index, op in enumerate(result.ops, start=1):
        current = replay_integer_row_ops(current, [op])
        assert profile.states[index] == current
        assert profile.densities[index] == integer_matrix_density(current)

    assert profile.states[-1] == replay_integer_row_ops(matrix, result.ops)


def test_generated_sparse_hnf_benchmark_runs_exact_verification() -> None:
    result = run_hnf_benchmark(
        HNFBenchmarkConfig(
            count=4,
            rows=4,
            cols=4,
            density=0.3,
            entry_bound=5,
            seed_start=0,
        )
    )

    aggregate = result["aggregate"]
    samples = result["samples"]
    assert result["status"] == "ok"
    assert result["source"] == "generated"
    assert result["family"] == "sparse_integer"
    assert result["count"] == 4
    assert aggregate["sample_count"] == 4
    assert aggregate["success_count"] == 4
    assert aggregate["success_rate"] == 1.0
    assert aggregate["status_counts"] == {"success": 4}
    assert len(samples) == 4
    assert all(sample["replay_ok"] for sample in samples)
    assert all(sample["final_is_hnf"] for sample in samples)
    assert all(isinstance(sample["max_abs_seen"], int) for sample in samples)
    assert "mean_trace_length" in aggregate
    assert "mean_initial_density" in aggregate
    assert "mean_wall_time_seconds" in aggregate
    assert "max_max_abs_seen" in aggregate
    assert "max_growth_numerator" in aggregate


def test_hnf_benchmark_output_omits_full_matrices_and_ops() -> None:
    result = run_hnf_benchmark(
        HNFBenchmarkConfig(count=1, rows=3, cols=3, density=0.4, entry_bound=3)
    )

    sample = result["samples"][0]
    assert "input" not in sample
    assert "initial_matrix" not in sample
    assert "final_matrix" not in sample
    assert "ops" not in sample


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"count": 0, "rows": 2, "cols": 2}, "count must be positive"),
        ({"count": 1, "rows": 0, "cols": 2}, "rows must be positive"),
        ({"count": 1, "rows": 2, "cols": 0}, "cols must be positive"),
        ({"count": 1, "rows": 2, "cols": 2, "density": -0.1}, "density"),
        ({"count": 1, "rows": 2, "cols": 2, "density": 1.1}, "density"),
        ({"count": 1, "rows": 2, "cols": 2, "entry_bound": 0}, "entry_bound"),
    ],
)
def test_hnf_benchmark_rejects_invalid_config(
    kwargs: dict[str, int | float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_hnf_benchmark(HNFBenchmarkConfig(**kwargs))
