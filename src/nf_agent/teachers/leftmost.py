from collections.abc import Sequence
from dataclasses import dataclass

from nf_agent.env.elementary_ops import Matrix, require_prime
from nf_agent.env.rref_modp import RREFResult, rref_leftmost


@dataclass(frozen=True)
class LeftmostRREFTeacher:
    p: int

    def __post_init__(self) -> None:
        require_prime(self.p)

    def solve(self, matrix: Sequence[Sequence[int]]) -> RREFResult:
        return rref_leftmost(matrix, self.p)


def leftmost_teacher_trajectory(matrix: Sequence[Sequence[int]], p: int) -> RREFResult:
    return LeftmostRREFTeacher(p=p).solve(matrix)


def final_matrix(matrix: Sequence[Sequence[int]], p: int) -> Matrix:
    return leftmost_teacher_trajectory(matrix, p).final_matrix

