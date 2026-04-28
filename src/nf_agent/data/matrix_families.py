from random import Random

from nf_agent.env.elementary_ops import Matrix, require_prime


def _require_shape(rows: int, cols: int) -> None:
    if rows < 0 or cols < 0:
        raise ValueError("matrix dimensions must be nonnegative")


def dense_random_matrix(rows: int, cols: int, p: int, seed: int) -> Matrix:
    require_prime(p)
    _require_shape(rows, cols)
    rng = Random(seed)
    return [[rng.randrange(p) for _ in range(cols)] for _ in range(rows)]


def sparse_random_matrix(rows: int, cols: int, p: int, density: float, seed: int) -> Matrix:
    require_prime(p)
    _require_shape(rows, cols)
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must lie in [0, 1]")
    rng = Random(seed)
    matrix: Matrix = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            if rng.random() < density:
                row.append(rng.randrange(1, p))
            else:
                row.append(0)
        matrix.append(row)
    return matrix


def low_rank_random_matrix(rows: int, cols: int, rank: int, p: int, seed: int) -> Matrix:
    require_prime(p)
    _require_shape(rows, cols)
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    rng = Random(seed)
    left = [[rng.randrange(p) for _ in range(rank)] for _ in range(rows)]
    right = [[rng.randrange(p) for _ in range(cols)] for _ in range(rank)]
    return [
        [sum(left[row][k] * right[k][col] for k in range(rank)) % p for col in range(cols)]
        for row in range(rows)
    ]

