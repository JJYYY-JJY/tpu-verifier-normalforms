import pytest

from nf_agent.env.rref_modp import is_rref_modp, replay_row_ops, rref_leftmost


@pytest.mark.parametrize(
    "matrix",
    [
        [[0, 0], [0, 0]],
        [[1, 0], [0, 1]],
        [[1, 2, 3], [2, 4, 6]],
        [[0, 1], [2, 3]],
        [[1, 2, 3], [4, 5, 6]],
    ],
)
def test_rref_result_satisfies_predicate_and_replays(matrix: list[list[int]]) -> None:
    result = rref_leftmost(matrix, 101)

    assert is_rref_modp(result.final_matrix, 101)
    assert replay_row_ops(matrix, result.ops, 101) == result.final_matrix


def test_rank_deficient_matrix_has_expected_rref() -> None:
    result = rref_leftmost([[1, 2, 3], [2, 4, 6]], 101)

    assert result.final_matrix == [[1, 2, 3], [0, 0, 0]]


def test_row_swap_needed_matrix_reduces_to_identity() -> None:
    result = rref_leftmost([[0, 1], [2, 3]], 101)

    assert result.final_matrix == [[1, 0], [0, 1]]
    assert result.ops[0].kind == "swap"


def test_rref_is_idempotent() -> None:
    first = rref_leftmost([[4, 8, 12], [1, 2, 3], [5, 0, 2]], 101)
    second = rref_leftmost(first.final_matrix, 101)

    assert second.final_matrix == first.final_matrix
    assert second.ops == []


def test_invalid_modulus_fails_fast() -> None:
    with pytest.raises(ValueError, match="prime"):
        rref_leftmost([[1]], 100)


def test_malformed_matrix_fails_fast() -> None:
    with pytest.raises(ValueError, match="rectangular"):
        rref_leftmost([[1], [2, 3]], 101)

