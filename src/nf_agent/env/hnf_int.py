from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from nf_agent.env.elementary_ops import Matrix

IntegerRowOpKind = Literal["swap", "negate", "add"]


@dataclass(frozen=True)
class IntegerRowOp:
    kind: IntegerRowOpKind
    target: int
    source: int | None = None
    scalar: int | None = None

    @staticmethod
    def swap(row_a: int, row_b: int) -> "IntegerRowOp":
        return IntegerRowOp(kind="swap", target=row_a, source=row_b)

    @staticmethod
    def negate(row: int) -> "IntegerRowOp":
        return IntegerRowOp(kind="negate", target=row)

    @staticmethod
    def add(target_row: int, source_row: int, scalar: int) -> "IntegerRowOp":
        return IntegerRowOp(kind="add", target=target_row, source=source_row, scalar=scalar)


@dataclass(frozen=True)
class CoefficientGrowthMetrics:
    initial_max_abs: int
    max_abs_seen: int
    initial_bitlength: int
    max_bitlength: int
    growth_numerator: int
    growth_denominator: int
    step_count: int


@dataclass(frozen=True)
class RowHNFResult:
    final_matrix: Matrix
    ops: list[IntegerRowOp]
    metrics: CoefficientGrowthMetrics


def normalize_integer_matrix(matrix: Sequence[Sequence[int]]) -> Matrix:
    if not isinstance(matrix, Sequence):
        raise TypeError("matrix must be a sequence of rows")

    normalized: Matrix = []
    expected_cols: int | None = None
    for row in matrix:
        if not isinstance(row, Sequence):
            raise TypeError("matrix rows must be sequences")
        values = []
        for entry in row:
            if not isinstance(entry, int) or isinstance(entry, bool):
                raise TypeError("matrix entries must be integers")
            values.append(entry)
        if expected_cols is None:
            expected_cols = len(values)
        elif len(values) != expected_cols:
            raise ValueError("matrix must be rectangular")
        normalized.append(values)
    return normalized


def _require_row_index(matrix: Matrix, row: int) -> None:
    if not isinstance(row, int) or isinstance(row, bool):
        raise TypeError("row index must be an integer")
    if not 0 <= row < len(matrix):
        raise IndexError(f"row index out of range: {row}")


def _require_scalar(scalar: int) -> None:
    if not isinstance(scalar, int) or isinstance(scalar, bool):
        raise TypeError("row operation scalar must be an integer")


def apply_integer_row_op(matrix: Sequence[Sequence[int]], op: IntegerRowOp) -> Matrix:
    result = normalize_integer_matrix(matrix)
    _require_row_index(result, op.target)

    if op.kind == "swap":
        if op.source is None:
            raise ValueError("swap op requires source row")
        _require_row_index(result, op.source)
        result[op.target], result[op.source] = result[op.source], result[op.target]
        return result

    if op.kind == "negate":
        result[op.target] = [-entry for entry in result[op.target]]
        return result

    if op.kind == "add":
        if op.source is None:
            raise ValueError("add op requires source row")
        if op.scalar is None:
            raise ValueError("add op requires scalar")
        _require_row_index(result, op.source)
        _require_scalar(op.scalar)
        if op.target == op.source:
            raise ValueError("add op target and source rows must be distinct")
        source = result[op.source]
        result[op.target] = [
            target_entry + op.scalar * source_entry
            for target_entry, source_entry in zip(result[op.target], source, strict=True)
        ]
        return result

    raise ValueError(f"unknown integer row operation kind: {op.kind}")


def replay_integer_row_ops(
    matrix: Sequence[Sequence[int]],
    ops: Sequence[IntegerRowOp],
) -> Matrix:
    current = normalize_integer_matrix(matrix)
    for op in ops:
        current = apply_integer_row_op(current, op)
    return current


def _max_abs(matrix: Sequence[Sequence[int]]) -> int:
    return max((abs(entry) for row in matrix for entry in row), default=0)


def _leading_col(row: Sequence[int]) -> int | None:
    for index, value in enumerate(row):
        if value != 0:
            return index
    return None


def _apply_tracked(current: Matrix, op: IntegerRowOp, ops: list[IntegerRowOp]) -> Matrix:
    updated = apply_integer_row_op(current, op)
    ops.append(op)
    return updated


def _coefficient_growth_metrics(
    initial_max_abs: int,
    max_abs_seen: int,
    step_count: int,
) -> CoefficientGrowthMetrics:
    return CoefficientGrowthMetrics(
        initial_max_abs=initial_max_abs,
        max_abs_seen=max_abs_seen,
        initial_bitlength=initial_max_abs.bit_length(),
        max_bitlength=max_abs_seen.bit_length(),
        growth_numerator=max_abs_seen,
        growth_denominator=max(1, initial_max_abs),
        step_count=step_count,
    )


def row_hnf(matrix: Sequence[Sequence[int]]) -> RowHNFResult:
    current = normalize_integer_matrix(matrix)
    ops: list[IntegerRowOp] = []
    initial_max_abs = _max_abs(current)
    max_abs_seen = initial_max_abs

    if not current:
        return RowHNFResult(
            final_matrix=current,
            ops=ops,
            metrics=_coefficient_growth_metrics(initial_max_abs, max_abs_seen, len(ops)),
        )

    rows = len(current)
    cols = len(current[0])
    pivot_row = 0

    def apply(op: IntegerRowOp) -> None:
        nonlocal current, max_abs_seen
        current = _apply_tracked(current, op, ops)
        max_abs_seen = max(max_abs_seen, _max_abs(current))

    for col in range(cols):
        if pivot_row >= rows:
            break

        while True:
            nonzero_rows = [row for row in range(pivot_row, rows) if current[row][col] != 0]
            if len(nonzero_rows) <= 1:
                break

            reducer = min(nonzero_rows, key=lambda row: abs(current[row][col]))
            for row in nonzero_rows:
                if row == reducer or current[row][col] == 0:
                    continue
                quotient = current[row][col] // current[reducer][col]
                if quotient != 0:
                    apply(IntegerRowOp.add(row, reducer, -quotient))

        selected = next(
            (row for row in range(pivot_row, rows) if current[row][col] != 0),
            None,
        )
        if selected is None:
            continue

        if selected != pivot_row:
            apply(IntegerRowOp.swap(pivot_row, selected))

        if current[pivot_row][col] < 0:
            apply(IntegerRowOp.negate(pivot_row))

        pivot = current[pivot_row][col]
        for row in range(pivot_row):
            value = current[row][col]
            if value == 0:
                continue
            quotient = value // pivot
            apply(IntegerRowOp.add(row, pivot_row, -quotient))

        pivot_row += 1

    return RowHNFResult(
        final_matrix=current,
        ops=ops,
        metrics=_coefficient_growth_metrics(initial_max_abs, max_abs_seen, len(ops)),
    )


def is_row_hnf(matrix: Sequence[Sequence[int]]) -> bool:
    normalized = normalize_integer_matrix(matrix)
    previous_pivot_col = -1
    seen_zero_row = False

    for row_index, row in enumerate(normalized):
        pivot_col = _leading_col(row)
        if pivot_col is None:
            seen_zero_row = True
            continue
        if seen_zero_row:
            return False
        if pivot_col <= previous_pivot_col:
            return False
        pivot = row[pivot_col]
        if pivot <= 0:
            return False
        for lower_row in normalized[row_index + 1 :]:
            if lower_row[pivot_col] != 0:
                return False
        for upper_row in normalized[:row_index]:
            value = upper_row[pivot_col]
            if not 0 <= value < pivot:
                return False
        previous_pivot_col = pivot_col

    return True
