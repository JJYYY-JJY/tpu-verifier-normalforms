import pytest

from nf_agent.data.matrix_families import (
    dense_random_matrix,
    sparse_integer_matrix,
    sparse_random_matrix,
)


def test_dense_random_matrix_is_seed_deterministic() -> None:
    left = dense_random_matrix(rows=4, cols=5, p=101, seed=17)
    right = dense_random_matrix(rows=4, cols=5, p=101, seed=17)

    assert left == right
    assert all(0 <= entry < 101 for row in left for entry in row)


def test_sparse_random_matrix_density_behavior() -> None:
    matrix = sparse_random_matrix(rows=40, cols=40, p=101, density=0.1, seed=23)
    nonzero = sum(entry != 0 for row in matrix for entry in row)
    observed_density = nonzero / (40 * 40)

    assert 0.06 <= observed_density <= 0.14


def test_sparse_random_matrix_is_seed_deterministic() -> None:
    left = sparse_random_matrix(rows=6, cols=6, p=101, density=0.25, seed=9)
    right = sparse_random_matrix(rows=6, cols=6, p=101, density=0.25, seed=9)

    assert left == right


def test_sparse_integer_matrix_is_seed_deterministic() -> None:
    left = sparse_integer_matrix(rows=6, cols=6, density=0.25, seed=9)
    right = sparse_integer_matrix(rows=6, cols=6, density=0.25, seed=9)

    assert left == right


def test_sparse_integer_matrix_density_behavior() -> None:
    matrix = sparse_integer_matrix(rows=80, cols=80, density=0.1, seed=23, entry_bound=9)
    nonzero = sum(entry != 0 for row in matrix for entry in row)
    observed_density = nonzero / (80 * 80)

    assert 0.075 <= observed_density <= 0.125


def test_sparse_integer_matrix_entries_are_bounded_integers_and_selected_entries_nonzero() -> None:
    matrix = sparse_integer_matrix(rows=20, cols=20, density=1.0, seed=17, entry_bound=3)

    assert all(isinstance(entry, int) for row in matrix for entry in row)
    assert all(-3 <= entry <= 3 for row in matrix for entry in row)
    assert all(entry != 0 for row in matrix for entry in row)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rows": -1, "cols": 2, "density": 0.2, "seed": 0}, "dimensions"),
        ({"rows": 2, "cols": -1, "density": 0.2, "seed": 0}, "dimensions"),
        ({"rows": 2, "cols": 2, "density": -0.1, "seed": 0}, "density"),
        ({"rows": 2, "cols": 2, "density": 1.1, "seed": 0}, "density"),
        ({"rows": 2, "cols": 2, "density": 0.2, "seed": 0, "entry_bound": 0}, "bound"),
    ],
)
def test_sparse_integer_matrix_rejects_invalid_parameters(
    kwargs: dict[str, int | float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sparse_integer_matrix(**kwargs)
