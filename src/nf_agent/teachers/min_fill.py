from collections.abc import Sequence
from dataclasses import dataclass

from nf_agent.env.elementary_ops import (
    Matrix,
    add_row_multiple,
    inv_mod,
    normalize_matrix,
    require_prime,
    scale_row,
    swap_rows,
)
from nf_agent.env.rref_modp import PivotAction, RowOp, RREFResult


@dataclass(frozen=True)
class _CandidateStep:
    candidate_row: int
    matrix: Matrix
    ops: list[RowOp]
    nonzero_count: int


@dataclass(frozen=True)
class MinFillRREFTeacher:
    p: int

    def __post_init__(self) -> None:
        require_prime(self.p)

    def solve(self, matrix: Sequence[Sequence[int]]) -> RREFResult:
        return _rref_min_fill(matrix, self.p)


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


def _nonzero_count(matrix: Matrix, p: int) -> int:
    return sum(1 for row in matrix for value in row if value % p != 0)


def _simulate_pivot_step(
    matrix: Matrix,
    p: int,
    pivot_row: int,
    col: int,
    candidate_row: int,
) -> _CandidateStep:
    current = [row[:] for row in matrix]
    ops: list[RowOp] = []
    rows = len(current)

    if candidate_row != pivot_row:
        op = RowOp.swap(pivot_row, candidate_row)
        current = _apply_row_op(current, op, p)
        ops.append(op)

    pivot_value = current[pivot_row][col] % p
    if pivot_value != 1:
        op = RowOp.scale(pivot_row, inv_mod(pivot_value, p))
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

    return _CandidateStep(
        candidate_row=candidate_row,
        matrix=current,
        ops=ops,
        nonzero_count=_nonzero_count(current, p),
    )


def _rref_min_fill(matrix: Sequence[Sequence[int]], p: int) -> RREFResult:
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

        candidates = [
            row for row in range(pivot_row, rows) if current[row][col] % p != 0
        ]
        if not candidates:
            continue

        selected = min(
            (
                _simulate_pivot_step(current, p, pivot_row, col, candidate)
                for candidate in candidates
            ),
            key=lambda step: (step.nonzero_count, len(step.ops), step.candidate_row),
        )
        current = selected.matrix
        ops.extend(selected.ops)
        pivots.append(PivotAction(row=pivot_row, col=col))
        pivot_row += 1

    return RREFResult(final_matrix=current, ops=ops, pivots=pivots, modulus=p)


def min_fill_teacher_trajectory(matrix: Sequence[Sequence[int]], p: int) -> RREFResult:
    return MinFillRREFTeacher(p=p).solve(matrix)


def final_matrix(matrix: Sequence[Sequence[int]], p: int) -> Matrix:
    return min_fill_teacher_trajectory(matrix, p).final_matrix
