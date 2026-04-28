from nf_agent.data.matrix_families import dense_random_matrix
from nf_agent.env.rref_modp import RowOp, is_rref_modp, replay_row_ops
from nf_agent.teachers.leftmost import leftmost_teacher_trajectory
from nf_agent.teachers.min_fill import MinFillRREFTeacher, min_fill_teacher_trajectory


def test_min_fill_prefers_pivot_row_with_less_post_step_fill() -> None:
    matrix = [[1, 1, 1], [1, 0, 0], [0, 1, 0]]

    min_fill = min_fill_teacher_trajectory(matrix, 101)
    leftmost = leftmost_teacher_trajectory(matrix, 101)

    assert min_fill.ops[0] == RowOp.swap(0, 1)
    assert leftmost.ops[0] != RowOp.swap(0, 1)


def test_min_fill_teacher_emits_exact_rref_trace_matching_leftmost_final() -> None:
    matrix = [[1, 1, 1], [1, 0, 0], [0, 1, 0]]

    min_fill = min_fill_teacher_trajectory(matrix, 101)
    leftmost = leftmost_teacher_trajectory(matrix, 101)

    assert replay_row_ops(matrix, min_fill.ops, 101) == min_fill.final_matrix
    assert is_rref_modp(min_fill.final_matrix, 101)
    assert min_fill.final_matrix == leftmost.final_matrix


def test_min_fill_teacher_verifies_random_4x4_matrices_over_f101() -> None:
    teacher = MinFillRREFTeacher(p=101)

    for seed in range(256):
        matrix = dense_random_matrix(rows=4, cols=4, p=101, seed=seed)
        result = teacher.solve(matrix)
        assert replay_row_ops(matrix, result.ops, 101) == result.final_matrix
        assert is_rref_modp(result.final_matrix, 101)
