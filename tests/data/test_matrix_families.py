from nf_agent.data.matrix_families import dense_random_matrix, sparse_random_matrix


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

