import pytest

from nf_agent.env.hnf_int import (
    IntegerRowOp,
    apply_integer_row_op,
    is_row_hnf,
    normalize_integer_matrix,
    replay_integer_row_ops,
    row_hnf,
)


def test_normalize_integer_matrix_copies_rectangular_matrix() -> None:
    original = [[1, -2], [3, 4]]

    normalized = normalize_integer_matrix(original)
    original[0][0] = 99

    assert normalized == [[1, -2], [3, 4]]


@pytest.mark.parametrize(
    ("matrix", "error", "message"),
    [
        ([[1], [2, 3]], ValueError, "rectangular"),
        ([[1.5]], TypeError, "integers"),
        ([[True]], TypeError, "integers"),
    ],
)
def test_normalize_integer_matrix_rejects_malformed_input(
    matrix: list[list[object]],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        normalize_integer_matrix(matrix)


def test_integer_row_ops_are_exact_and_non_mutating() -> None:
    matrix = [[1, 2], [3, 4]]

    assert apply_integer_row_op(matrix, IntegerRowOp.swap(0, 1)) == [[3, 4], [1, 2]]
    assert apply_integer_row_op(matrix, IntegerRowOp.negate(0)) == [[-1, -2], [3, 4]]
    assert apply_integer_row_op(matrix, IntegerRowOp.add(1, 0, 2)) == [[1, 2], [5, 8]]
    assert matrix == [[1, 2], [3, 4]]


def test_replay_integer_row_ops_matches_manual_final_matrix() -> None:
    matrix = [[1, 2], [3, 4]]
    ops = [
        IntegerRowOp.swap(0, 1),
        IntegerRowOp.negate(0),
        IntegerRowOp.add(1, 0, 2),
    ]

    assert replay_integer_row_ops(matrix, ops) == [[-3, -4], [-5, -6]]


@pytest.mark.parametrize(
    ("op", "error", "message"),
    [
        (IntegerRowOp.swap(0, 2), IndexError, "row index out of range"),
        (IntegerRowOp(kind="swap", target=0), ValueError, "source row"),
        (IntegerRowOp.add(0, 0, 1), ValueError, "target and source"),
        (IntegerRowOp(kind="add", target=0, source=1), ValueError, "scalar"),
    ],
)
def test_invalid_integer_row_ops_fail_explicitly(
    op: IntegerRowOp,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        apply_integer_row_op([[1], [2]], op)


@pytest.mark.parametrize(
    "matrix",
    [
        [[-2, 4]],
        [[0, 0, 0], [2, 4, 6], [4, 8, 12]],
        [[1, 5], [0, 3]],
        [[6, 0], [10, 0]],
        [[0, 6], [0, 10], [0, -14]],
    ],
)
def test_row_hnf_satisfies_predicate_and_replays(matrix: list[list[int]]) -> None:
    result = row_hnf(matrix)

    assert is_row_hnf(result.final_matrix)
    assert replay_integer_row_ops(matrix, result.ops) == result.final_matrix


def test_row_hnf_reduces_entries_above_pivots_to_residues() -> None:
    result = row_hnf([[1, 5], [0, 3]])

    assert result.final_matrix == [[1, 2], [0, 3]]


def test_rank_deficient_row_hnf_has_zero_rows_below_nonzero_rows() -> None:
    result = row_hnf([[0, 0, 0], [2, 4, 6], [4, 8, 12]])

    assert result.final_matrix == [[2, 4, 6], [0, 0, 0], [0, 0, 0]]


def test_coefficient_growth_metrics_are_exact_integers() -> None:
    result = row_hnf([[6, 0], [10, 1]])
    metrics = result.metrics

    assert metrics.initial_max_abs == 10
    assert metrics.max_abs_seen >= metrics.initial_max_abs
    assert metrics.initial_bitlength == (10).bit_length()
    assert metrics.max_bitlength == metrics.max_abs_seen.bit_length()
    assert metrics.growth_numerator == metrics.max_abs_seen
    assert metrics.growth_denominator == 10
    assert metrics.step_count == len(result.ops)


def test_coefficient_growth_zero_matrix_denominator_is_one() -> None:
    result = row_hnf([[0, 0], [0, 0]])

    assert result.metrics.growth_numerator == 0
    assert result.metrics.growth_denominator == 1
    assert result.metrics.step_count == 0
