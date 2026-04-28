from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from nf_agent.env.elementary_ops import (
    Matrix,
    add_row_multiple,
    inv_mod,
    normalize_matrix,
    require_prime,
    scale_row,
    swap_rows,
)

RowOpKind = Literal["swap", "scale", "add"]


@dataclass(frozen=True)
class PivotAction:
    row: int
    col: int


@dataclass(frozen=True)
class RowOp:
    kind: RowOpKind
    target: int
    source: int | None = None
    scalar: int | None = None

    @staticmethod
    def swap(row_a: int, row_b: int) -> "RowOp":
        return RowOp(kind="swap", target=row_a, source=row_b)

    @staticmethod
    def scale(row: int, scalar: int) -> "RowOp":
        return RowOp(kind="scale", target=row, scalar=scalar)

    @staticmethod
    def add(target_row: int, source_row: int, scalar: int) -> "RowOp":
        return RowOp(kind="add", target=target_row, source=source_row, scalar=scalar)


@dataclass(frozen=True)
class RREFResult:
    final_matrix: Matrix
    ops: list[RowOp]
    pivots: list[PivotAction]
    modulus: int


def _apply_row_op(matrix: Matrix, op: RowOp, p: int) -> Matrix:
    if op.kind == "swap":
        if op.source is None:
            raise ValueError("swap op requires source row")
        return swap_rows(matrix, op.target, op.source, p)
    if op.kind == "scale":
        if op.scalar is None:
            raise ValueError("scale op requires scalar")
        return scale_row(matrix, op.target, op.scalar, p)
    if op.kind == "add":
        if op.source is None or op.scalar is None:
            raise ValueError("add op requires source row and scalar")
        return add_row_multiple(matrix, op.target, op.source, op.scalar, p)
    raise ValueError(f"unknown row operation kind: {op.kind}")


def replay_row_ops(matrix: Sequence[Sequence[int]], ops: Sequence[RowOp], p: int) -> Matrix:
    current = normalize_matrix(matrix, p)
    for op in ops:
        current = _apply_row_op(current, op, p)
    return current


def rref_leftmost(matrix: Sequence[Sequence[int]], p: int) -> RREFResult:
    require_prime(p)
    current = normalize_matrix(matrix, p)
    ops: list[RowOp] = []
    pivots: list[PivotAction] = []
    if not current:
        return RREFResult(final_matrix=current, ops=ops, pivots=pivots, modulus=p)

    rows = len(current)
    cols = len(current[0])
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break
        selected = next(
            (row for row in range(pivot_row, rows) if current[row][col] % p != 0),
            None,
        )
        if selected is None:
            continue

        if selected != pivot_row:
            op = RowOp.swap(pivot_row, selected)
            current = _apply_row_op(current, op, p)
            ops.append(op)

        pivot_value = current[pivot_row][col] % p
        if pivot_value != 1:
            scalar = inv_mod(pivot_value, p)
            op = RowOp.scale(pivot_row, scalar)
            current = _apply_row_op(current, op, p)
            ops.append(op)

        for row in range(rows):
            if row == pivot_row:
                continue
            value = current[row][col] % p
            if value == 0:
                continue
            op = RowOp.add(row, pivot_row, -value)
            current = _apply_row_op(current, op, p)
            ops.append(op)

        pivots.append(PivotAction(row=pivot_row, col=col))
        pivot_row += 1

    return RREFResult(final_matrix=current, ops=ops, pivots=pivots, modulus=p)


def _leading_col(row: Sequence[int]) -> int | None:
    for index, value in enumerate(row):
        if value != 0:
            return index
    return None


def is_rref_modp(matrix: Sequence[Sequence[int]], p: int) -> bool:
    normalized = normalize_matrix(matrix, p)
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
        if row[pivot_col] != 1:
            return False
        for other_index, other_row in enumerate(normalized):
            if other_index != row_index and other_row[pivot_col] != 0:
                return False
        previous_pivot_col = pivot_col

    return True

