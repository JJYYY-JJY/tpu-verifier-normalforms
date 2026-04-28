import pytest

from nf_agent.env.elementary_ops import add_row_multiple, inv_mod, scale_row, swap_rows


def test_inv_mod_finds_multiplicative_inverse() -> None:
    assert inv_mod(3, 101) == 34
    assert inv_mod(-3, 101) == 67


def test_inv_mod_rejects_non_prime_modulus() -> None:
    with pytest.raises(ValueError, match="prime"):
        inv_mod(3, 100)


def test_swap_rows_returns_new_matrix_without_mutating_input() -> None:
    matrix = [[1, 2], [3, 4]]

    swapped = swap_rows(matrix, 0, 1, 101)

    assert swapped == [[3, 4], [1, 2]]
    assert matrix == [[1, 2], [3, 4]]
    assert swapped is not matrix


def test_scale_row_normalizes_entries_mod_p_without_mutating_input() -> None:
    matrix = [[2, 4], [3, 5]]

    scaled = scale_row(matrix, 0, 51, 101)

    assert scaled == [[1, 2], [3, 5]]
    assert matrix == [[2, 4], [3, 5]]


def test_scale_row_rejects_zero_factor() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        scale_row([[1]], 0, 0, 101)


def test_add_row_multiple_is_exact_modular_row_addition() -> None:
    matrix = [[1, 2], [3, 4]]

    updated = add_row_multiple(matrix, target_row=1, source_row=0, scalar=-3, p=101)

    assert updated == [[1, 2], [0, 99]]
    assert matrix == [[1, 2], [3, 4]]

