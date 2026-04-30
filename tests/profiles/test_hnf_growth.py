from nf_agent.data.matrix_families import sparse_integer_matrix
from nf_agent.env.hnf_int import IntegerRowOp, is_row_hnf, replay_integer_row_ops
from nf_agent.profiles.hnf_growth import _search_row_preconditioned_row_hnf


def test_row_preconditioned_search_replays_best_and_improves_sparse_seed_window() -> None:
    improved_samples = 0

    for seed in range(8):
        matrix = sparse_integer_matrix(
            rows=8,
            cols=8,
            density=0.2,
            seed=seed,
            entry_bound=9,
        )

        result = _search_row_preconditioned_row_hnf(matrix, candidate_limit=64)

        assert result.candidate_count == 64
        assert result.rejected_candidate_count == 0
        assert replay_integer_row_ops(matrix, result.best_ops) == result.best_final_matrix
        assert is_row_hnf(result.best_final_matrix)
        assert result.best.metrics["max_bitlength"] <= result.baseline.metrics["max_bitlength"]
        assert result.best.metrics["max_abs_seen"] <= result.baseline.metrics["max_abs_seen"]
        if result.improved_metrics:
            improved_samples += 1

    assert improved_samples >= 1


def test_row_preconditioned_search_rejects_illegal_candidate_without_fallback() -> None:
    matrix = [[1, 0], [0, 1]]

    result = _search_row_preconditioned_row_hnf(
        matrix,
        candidate_limit=2,
        candidate_preconditioners=[
            (),
            (IntegerRowOp.swap(0, 99),),
        ],
    )

    assert result.candidate_count == 2
    assert result.rejected_candidate_count == 1
    assert result.best_candidate == 0
    assert result.best_policy == "row_hnf"
    assert result.improved_metrics == []
    assert replay_integer_row_ops(matrix, result.best_ops) == result.best_final_matrix
    assert is_row_hnf(result.best_final_matrix)


def test_candidate_limit_one_is_row_hnf_baseline() -> None:
    matrix = sparse_integer_matrix(
        rows=8,
        cols=8,
        density=0.2,
        seed=4,
        entry_bound=9,
    )

    result = _search_row_preconditioned_row_hnf(matrix, candidate_limit=1)

    assert result.candidate_count == 1
    assert result.rejected_candidate_count == 0
    assert result.best_candidate == 0
    assert result.best_policy == "row_hnf"
    assert result.best.metrics == result.baseline.metrics
    assert result.improved_metrics == []
