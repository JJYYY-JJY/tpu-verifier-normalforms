from nf_agent.data.matrix_families import dense_random_matrix
from nf_agent.env.rref_modp import is_rref_modp, replay_row_ops
from nf_agent.teachers.leftmost import LeftmostRREFTeacher, leftmost_teacher_trajectory


def test_leftmost_teacher_emits_legal_trace() -> None:
    matrix = [[0, 1], [2, 3]]
    result = leftmost_teacher_trajectory(matrix, 101)

    assert result.pivots[0].col == 0
    assert is_rref_modp(result.final_matrix, 101)
    assert replay_row_ops(matrix, result.ops, 101) == result.final_matrix


def test_leftmost_teacher_verifies_random_10000_4x4_matrices_over_f101() -> None:
    teacher = LeftmostRREFTeacher(p=101)

    for seed in range(10_000):
        matrix = dense_random_matrix(rows=4, cols=4, p=101, seed=seed)
        result = teacher.solve(matrix)
        assert is_rref_modp(result.final_matrix, 101)
        assert replay_row_ops(matrix, result.ops, 101) == result.final_matrix

