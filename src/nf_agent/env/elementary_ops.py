from collections.abc import Sequence

Matrix = list[list[int]]


def _is_prime(p: int) -> bool:
    if p < 2:
        return False
    if p == 2:
        return True
    if p % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= p:
        if p % divisor == 0:
            return False
        divisor += 2
    return True


def require_prime(p: int) -> None:
    if not isinstance(p, int) or not _is_prime(p):
        raise ValueError(f"modulus p must be prime, got {p!r}")


def normalize_matrix(matrix: Sequence[Sequence[int]], p: int) -> Matrix:
    require_prime(p)
    if not isinstance(matrix, Sequence):
        raise TypeError("matrix must be a sequence of rows")

    normalized: Matrix = []
    expected_cols: int | None = None
    for row in matrix:
        if not isinstance(row, Sequence):
            raise TypeError("matrix rows must be sequences")
        values = []
        for entry in row:
            if not isinstance(entry, int):
                raise TypeError("matrix entries must be integers")
            values.append(entry % p)
        if expected_cols is None:
            expected_cols = len(values)
        elif len(values) != expected_cols:
            raise ValueError("matrix must be rectangular")
        normalized.append(values)
    return normalized


def _require_row_index(matrix: Matrix, row: int) -> None:
    if not 0 <= row < len(matrix):
        raise IndexError(f"row index out of range: {row}")


def inv_mod(a: int, p: int) -> int:
    require_prime(p)
    value = a % p
    if value == 0:
        raise ZeroDivisionError("zero has no inverse modulo p")
    return pow(value, -1, p)


def swap_rows(matrix: Sequence[Sequence[int]], row_a: int, row_b: int, p: int) -> Matrix:
    result = normalize_matrix(matrix, p)
    _require_row_index(result, row_a)
    _require_row_index(result, row_b)
    result[row_a], result[row_b] = result[row_b], result[row_a]
    return result


def scale_row(matrix: Sequence[Sequence[int]], row: int, scalar: int, p: int) -> Matrix:
    result = normalize_matrix(matrix, p)
    _require_row_index(result, row)
    factor = scalar % p
    if factor == 0:
        raise ValueError("row scaling factor must be nonzero modulo p")
    result[row] = [(factor * entry) % p for entry in result[row]]
    return result


def add_row_multiple(
    matrix: Sequence[Sequence[int]],
    target_row: int,
    source_row: int,
    scalar: int,
    p: int,
) -> Matrix:
    result = normalize_matrix(matrix, p)
    _require_row_index(result, target_row)
    _require_row_index(result, source_row)
    factor = scalar % p
    source = result[source_row]
    result[target_row] = [
        (target_entry + factor * source_entry) % p
        for target_entry, source_entry in zip(result[target_row], source, strict=True)
    ]
    return result

