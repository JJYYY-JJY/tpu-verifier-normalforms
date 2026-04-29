from __future__ import annotations

import random
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar

from nf_agent.certificates import (
    SNF_CERTIFICATE_KIND,
    SNF_CERTIFICATE_SCHEMA_VERSION,
    IntegerMatrixOp,
    SNFCertificate,
    replay_snf_certificate,
    verify_snf_certificate,
)

IntegerMatrix: TypeAlias = list[list[int]]
MetricRecord: TypeAlias = dict[str, Any]
T = TypeVar("T")


@dataclass(frozen=True)
class SNFBenchmarkConfig:
    count: int
    rows: int
    cols: int
    diagonal_factor_bound: int = 5
    row_op_count: int = 2
    col_op_count: int = 2
    op_scalar_bound: int = 3
    seed_start: int = 0


@dataclass(frozen=True)
class BenchmarkSample:
    sample_index: int
    seed: int
    certificate: SNFCertificate
    profile: list[IntegerMatrix]


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_nonnegative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _validate_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _identity_matrix(size: int) -> IntegerMatrix:
    return [[1 if row == col else 0 for col in range(size)] for row in range(size)]


def _diagonal_matrix(
    *,
    rows: int,
    cols: int,
    factor_bound: int,
    rng: random.Random,
) -> IntegerMatrix:
    matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    diagonal_length = min(rows, cols)
    value = rng.randint(1, factor_bound)
    for index in range(diagonal_length):
        if index > 0:
            value *= rng.randint(1, factor_bound)
        matrix[index][index] = value
    return matrix


def _generated_ops(
    *,
    dimension: int,
    count: int,
    scalar_bound: int,
    rng: random.Random,
) -> list[IntegerMatrixOp]:
    ops: list[IntegerMatrixOp] = []
    for _ in range(count):
        if dimension == 1:
            ops.append(IntegerMatrixOp(kind="negate", target=0))
            continue

        kind = rng.choice(("swap", "negate", "add"))
        target = rng.randrange(dimension)
        if kind == "negate":
            ops.append(IntegerMatrixOp(kind="negate", target=target))
            continue

        source = rng.randrange(dimension - 1)
        if source >= target:
            source += 1
        if kind == "swap":
            ops.append(IntegerMatrixOp(kind="swap", target=target, source=source))
            continue

        scalar = rng.choice(
            [value for value in range(-scalar_bound, scalar_bound + 1) if value != 0]
        )
        ops.append(IntegerMatrixOp(kind="add", target=target, source=source, scalar=scalar))
    return ops


def _inverse_op(op: IntegerMatrixOp) -> IntegerMatrixOp:
    if op.kind == "swap":
        return IntegerMatrixOp(kind="swap", target=op.target, source=op.source)
    if op.kind == "negate":
        return IntegerMatrixOp(kind="negate", target=op.target)
    if op.scalar is None:
        raise ValueError("add operation requires scalar")
    return IntegerMatrixOp(
        kind="add",
        target=op.target,
        source=op.source,
        scalar=-op.scalar,
    )


def _inverse_ops(ops: Sequence[IntegerMatrixOp]) -> list[IntegerMatrixOp]:
    return [_inverse_op(op) for op in reversed(ops)]


def _replay_row_ops(matrix: IntegerMatrix, ops: Sequence[IntegerMatrixOp]) -> list[IntegerMatrix]:
    states: list[IntegerMatrix] = [[row.copy() for row in matrix]]
    current = states[0]
    for op in ops:
        current = _apply_row_op(current, op)
        states.append(current)
    return states


def _replay_col_ops(
    matrix: IntegerMatrix,
    ops: Sequence[IntegerMatrixOp],
    column_count: int,
) -> list[IntegerMatrix]:
    states: list[IntegerMatrix] = [[row.copy() for row in matrix]]
    current = states[0]
    for op in ops:
        current = _apply_col_op(current, op, column_count)
        states.append(current)
    return states


def _apply_row_op(matrix: IntegerMatrix, op: IntegerMatrixOp) -> IntegerMatrix:
    result = [row.copy() for row in matrix]
    if op.kind == "swap":
        if op.source is None:
            raise ValueError("swap op requires source row")
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
        result[op.target] = [
            target_entry + op.scalar * source_entry
            for target_entry, source_entry in zip(result[op.target], result[op.source], strict=True)
        ]
        return result
    raise ValueError(f"unknown integer row operation kind: {op.kind}")


def _apply_col_op(
    matrix: IntegerMatrix,
    op: IntegerMatrixOp,
    column_count: int,
) -> IntegerMatrix:
    result = [row.copy() for row in matrix]
    if op.kind == "swap":
        if op.source is None:
            raise ValueError("swap op requires source column")
        for row in result:
            row[op.target], row[op.source] = row[op.source], row[op.target]
        return result
    if op.kind == "negate":
        for row in result:
            row[op.target] = -row[op.target]
        return result
    if op.kind == "add":
        if op.source is None:
            raise ValueError("add op requires source column")
        if op.scalar is None:
            raise ValueError("add op requires scalar")
        for row in result:
            row[op.target] += op.scalar * row[op.source]
        return result
    raise ValueError(f"unknown integer column operation kind: {op.kind}")


def _last_state(states: Sequence[IntegerMatrix]) -> IntegerMatrix:
    return [row.copy() for row in states[-1]]


def _generated_sample(
    *,
    sample_index: int,
    seed: int,
    config: SNFBenchmarkConfig,
) -> BenchmarkSample:
    rng = random.Random(seed)
    diagonal = _diagonal_matrix(
        rows=config.rows,
        cols=config.cols,
        factor_bound=config.diagonal_factor_bound,
        rng=rng,
    )
    row_ops = _generated_ops(
        dimension=config.rows,
        count=config.row_op_count,
        scalar_bound=config.op_scalar_bound,
        rng=rng,
    )
    col_ops = _generated_ops(
        dimension=config.cols,
        count=config.col_op_count,
        scalar_bound=config.op_scalar_bound,
        rng=rng,
    )

    inverse_col_states = _replay_col_ops(
        diagonal,
        _inverse_ops(col_ops),
        config.cols,
    )
    input_matrix = _last_state(
        _replay_row_ops(_last_state(inverse_col_states), _inverse_ops(row_ops))
    )
    left_transform = _last_state(_replay_row_ops(_identity_matrix(config.rows), row_ops))
    right_transform = _last_state(
        _replay_col_ops(_identity_matrix(config.cols), col_ops, config.cols)
    )

    row_states = _replay_row_ops(input_matrix, row_ops)
    col_states = _replay_col_ops(_last_state(row_states), col_ops, config.cols)
    profile = [*row_states, *col_states[1:]]

    return BenchmarkSample(
        sample_index=sample_index,
        seed=seed,
        certificate=SNFCertificate(
            kind=SNF_CERTIFICATE_KIND,
            schema_version=SNF_CERTIFICATE_SCHEMA_VERSION,
            shape=(config.rows, config.cols),
            input=input_matrix,
            diagonal=diagonal,
            left_transform=left_transform,
            right_transform=right_transform,
            row_ops=row_ops,
            col_ops=col_ops,
        ),
        profile=profile,
    )


def _generated_samples(config: SNFBenchmarkConfig) -> list[BenchmarkSample]:
    count = _validate_positive_int(config.count, "count")
    _validate_positive_int(config.rows, "rows")
    _validate_positive_int(config.cols, "cols")
    _validate_positive_int(config.diagonal_factor_bound, "diagonal_factor_bound")
    _validate_nonnegative_int(config.row_op_count, "row_op_count")
    _validate_nonnegative_int(config.col_op_count, "col_op_count")
    _validate_positive_int(config.op_scalar_bound, "op_scalar_bound")
    _validate_int(config.seed_start, "seed_start")
    return [
        _generated_sample(sample_index=sample_index, seed=seed, config=config)
        for sample_index, seed in enumerate(range(config.seed_start, config.seed_start + count))
    ]


def _time_call(callback: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    value = callback()
    return value, time.perf_counter() - start


def _matrix_density(matrix: Sequence[Sequence[int]]) -> float:
    total = sum(len(row) for row in matrix)
    if total == 0:
        return 0.0
    nonzero = sum(1 for row in matrix for entry in row if entry != 0)
    return nonzero / total


def _max_abs(matrix: Sequence[Sequence[int]]) -> int:
    return max((abs(entry) for row in matrix for entry in row), default=0)


def _profile_metrics(profile: Sequence[IntegerMatrix]) -> dict[str, float | int]:
    densities = [_matrix_density(state) for state in profile]
    max_abs_values = [_max_abs(state) for state in profile]
    initial_max_abs = max_abs_values[0] if max_abs_values else 0
    max_abs_seen = max(max_abs_values, default=0)
    initial_density = densities[0] if densities else 0.0
    return {
        "initial_density": initial_density,
        "final_density": densities[-1] if densities else 0.0,
        "max_density": max(densities) if densities else 0.0,
        "fill_in_delta": (max(densities) - initial_density) if densities else 0.0,
        "initial_max_abs": initial_max_abs,
        "max_abs_seen": max_abs_seen,
        "initial_bitlength": initial_max_abs.bit_length(),
        "max_bitlength": max_abs_seen.bit_length(),
    }


def _mean(samples: Iterable[Mapping[str, Any]], key: str) -> float:
    values = [float(sample[key]) for sample in samples if key in sample]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _status_counts(samples: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(sample["status"]) for sample in samples))


def _max_exact(samples: Iterable[Mapping[str, Any]], key: str) -> int:
    return max((int(sample[key]) for sample in samples if key in sample), default=0)


def _aggregate_samples(samples: list[MetricRecord]) -> dict[str, Any]:
    count = len(samples)
    success_count = sum(1 for sample in samples if bool(sample.get("success", False)))
    aggregate: dict[str, Any] = {
        "sample_count": count,
        "success_count": success_count,
        "success_rate": success_count / count if count else 0.0,
        "status_counts": _status_counts(samples),
    }
    for key in (
        "row_op_count",
        "col_op_count",
        "operation_count",
        "initial_density",
        "final_density",
        "max_density",
        "fill_in_delta",
        "wall_time_seconds",
        "replay_wall_time_seconds",
        "verify_wall_time_seconds",
    ):
        aggregate[f"mean_{key}"] = _mean(samples, key)
    for key in (
        "initial_max_abs",
        "max_abs_seen",
        "initial_bitlength",
        "max_bitlength",
    ):
        aggregate[f"max_{key}"] = _max_exact(samples, key)
    return aggregate


def _run_sample(sample: BenchmarkSample) -> MetricRecord:
    certificate = sample.certificate
    replayed, replay_seconds = _time_call(lambda: replay_snf_certificate(certificate))
    replay_ok = replayed == certificate.diagonal
    verify_error: str | None = None
    verify_seconds = 0.0
    try:
        _, verify_seconds = _time_call(lambda: verify_snf_certificate(certificate))
        verified = True
    except ValueError as exc:
        verified = False
        verify_error = str(exc)

    success = replay_ok and verified
    record: MetricRecord = {
        "sample_index": sample.sample_index,
        "seed": sample.seed,
        "status": "success" if success else "verification_failed",
        "success": success,
        "row_op_count": len(certificate.row_ops),
        "col_op_count": len(certificate.col_ops),
        "operation_count": len(certificate.row_ops) + len(certificate.col_ops),
        "replay_ok": replay_ok,
        "verified": verified,
        **_profile_metrics(sample.profile),
        "wall_time_seconds": replay_seconds + verify_seconds,
        "replay_wall_time_seconds": replay_seconds,
        "verify_wall_time_seconds": verify_seconds,
    }
    if verify_error is not None:
        record["verify_error"] = verify_error
    return record


def _policy_from_records(records: list[MetricRecord]) -> dict[str, Any]:
    return {"aggregate": _aggregate_samples(records), "samples": records}


def run_snf_benchmark(config: SNFBenchmarkConfig) -> dict[str, Any]:
    samples = _generated_samples(config)
    records = [_run_sample(sample) for sample in samples]
    return {
        "status": "ok",
        "source": "generated",
        "family": "snf_certificate",
        "count": len(samples),
        "rows": config.rows,
        "cols": config.cols,
        "diagonal_factor_bound": config.diagonal_factor_bound,
        "row_op_count": config.row_op_count,
        "col_op_count": config.col_op_count,
        "op_scalar_bound": config.op_scalar_bound,
        "seed_start": config.seed_start,
        "policies": {"certificate_replay": _policy_from_records(records)},
    }
