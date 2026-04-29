from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeVar, cast

import numpy as np

from nf_agent.data.matrix_families import (
    dense_random_matrix,
    low_rank_random_matrix,
    sparse_random_matrix,
)
from nf_agent.data.rref_shards import RREFShardSamples
from nf_agent.env.elementary_ops import Matrix, normalize_matrix, require_prime
from nf_agent.env.rref_modp import RowOp, is_rref_modp, replay_row_ops, rref_leftmost
from nf_agent.rollout import RREFPivotRolloutConfig, rollout_rref_pivot

BenchmarkSource: TypeAlias = Literal["generated", "shard"]
MatrixFamily: TypeAlias = Literal["dense", "sparse", "low_rank"]
MetricRecord: TypeAlias = dict[str, Any]
T = TypeVar("T")


@dataclass(frozen=True)
class RREFBenchmarkConfig:
    source: BenchmarkSource
    count: int | None = None
    rows: int | None = None
    cols: int | None = None
    modulus: int = 101
    family: MatrixFamily = "dense"
    seed_start: int = 0
    density: float | None = None
    rank: int | None = None
    data_path: str | Path | None = None
    model_data_path: str | Path | None = None
    checkpoint_dir: str | Path | None = None
    max_steps: int | None = None
    hidden_sizes: tuple[int, ...] = (256, 256)


@dataclass(frozen=True)
class DensityProfile:
    states: list[Matrix]
    densities: list[float]

    @property
    def initial_density(self) -> float:
        return self.densities[0] if self.densities else 0.0

    @property
    def final_density(self) -> float:
        return self.densities[-1] if self.densities else 0.0

    @property
    def max_density(self) -> float:
        return max(self.densities) if self.densities else 0.0

    @property
    def fill_in_delta(self) -> float:
        return self.max_density - self.initial_density


@dataclass(frozen=True)
class BenchmarkSample:
    sample_index: int
    matrix: Matrix
    seed: int | None = None


def matrix_density_modp(matrix: Sequence[Sequence[int]], p: int) -> float:
    normalized = normalize_matrix(matrix, p)
    total = sum(len(row) for row in normalized)
    if total == 0:
        return 0.0
    nonzero = sum(1 for row in normalized for entry in row if entry != 0)
    return nonzero / total


def row_op_density_profile(
    matrix: Sequence[Sequence[int]],
    ops: Sequence[RowOp],
    p: int,
) -> DensityProfile:
    current = normalize_matrix(matrix, p)
    states = [[row[:] for row in current]]
    densities = [matrix_density_modp(current, p)]
    for op in ops:
        current = replay_row_ops(current, [op], p)
        states.append([row[:] for row in current])
        densities.append(matrix_density_modp(current, p))
    return DensityProfile(states=states, densities=densities)


def _validate_positive_int(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _generated_matrix(config: RREFBenchmarkConfig, seed: int) -> Matrix:
    rows = _validate_positive_int(config.rows, "rows")
    cols = _validate_positive_int(config.cols, "cols")
    if config.family == "dense":
        return dense_random_matrix(rows=rows, cols=cols, p=config.modulus, seed=seed)
    if config.family == "sparse":
        if config.density is None:
            raise ValueError("density is required for sparse generated benchmarks")
        return sparse_random_matrix(
            rows=rows,
            cols=cols,
            p=config.modulus,
            density=config.density,
            seed=seed,
        )
    if config.family == "low_rank":
        if config.rank is None:
            raise ValueError("rank is required for low_rank generated benchmarks")
        return low_rank_random_matrix(
            rows=rows,
            cols=cols,
            rank=config.rank,
            p=config.modulus,
            seed=seed,
        )
    raise ValueError(f"unsupported matrix family: {config.family!r}")


def _generated_samples(config: RREFBenchmarkConfig) -> list[BenchmarkSample]:
    if config.rows is None or config.cols is None or config.count is None:
        raise ValueError("rows, cols, and count are required for generated source")
    count = _validate_positive_int(config.count, "count")
    require_prime(config.modulus)
    return [
        BenchmarkSample(
            sample_index=sample_index,
            seed=seed,
            matrix=_generated_matrix(config, seed),
        )
        for sample_index, seed in enumerate(range(config.seed_start, config.seed_start + count))
    ]


def _raw_shard_inputs(path: str | Path, limit: int) -> list[Matrix]:
    with np.load(path, allow_pickle=False) as shard:
        inputs = np.asarray(shard["inputs"][:limit], dtype=np.int64)
    return cast(list[Matrix], inputs.tolist())


def _shard_samples(config: RREFBenchmarkConfig) -> tuple[list[BenchmarkSample], RREFShardSamples]:
    if config.data_path is None:
        raise ValueError("data_path is required for shard source")
    samples = RREFShardSamples(config.data_path)
    if config.count is None:
        limit = len(samples)
    else:
        if config.count <= 0:
            raise ValueError("count must be positive")
        limit = min(config.count, len(samples))
    matrices = _raw_shard_inputs(config.data_path, limit)
    return (
        [
            BenchmarkSample(sample_index=sample_index, matrix=matrix)
            for sample_index, matrix in enumerate(matrices)
        ],
        samples,
    )


def _input_samples(
    config: RREFBenchmarkConfig,
) -> tuple[list[BenchmarkSample], RREFShardSamples | None]:
    if config.source == "generated":
        return _generated_samples(config), None
    if config.source == "shard":
        return _shard_samples(config)
    raise ValueError(f"unsupported benchmark source: {config.source!r}")


def _metadata_shape_modulus(samples: RREFShardSamples) -> tuple[int, int, int]:
    return samples.rows, samples.cols, samples.modulus


def _benchmark_shape_modulus(
    config: RREFBenchmarkConfig,
    shard_samples: RREFShardSamples | None,
) -> tuple[int, int, int]:
    if config.source == "generated":
        rows = _validate_positive_int(config.rows, "rows")
        cols = _validate_positive_int(config.cols, "cols")
        return rows, cols, config.modulus
    if shard_samples is None:
        raise ValueError("shard metadata is required")
    return _metadata_shape_modulus(shard_samples)


def _neural_model_data_path(config: RREFBenchmarkConfig) -> str | Path | None:
    if config.checkpoint_dir is None:
        return None
    if config.source == "shard":
        return config.model_data_path if config.model_data_path is not None else config.data_path
    if config.model_data_path is None:
        raise ValueError("model_data_path is required when checkpoint_dir is provided")
    return config.model_data_path


def _validate_model_metadata(
    config: RREFBenchmarkConfig,
    shard_samples: RREFShardSamples | None,
) -> str | Path | None:
    model_data_path = _neural_model_data_path(config)
    if model_data_path is None:
        return None
    model_samples = RREFShardSamples(model_data_path)
    benchmark_shape = _benchmark_shape_modulus(config, shard_samples)
    if _metadata_shape_modulus(model_samples) != benchmark_shape:
        raise ValueError("model data shape/modulus must match generated benchmark config")
    return model_data_path


def _mean(samples: Iterable[Mapping[str, Any]], key: str) -> float:
    values = [float(sample[key]) for sample in samples if key in sample]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _status_counts(samples: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(sample["status"]) for sample in samples))


def _aggregate_samples(samples: list[MetricRecord], keys: Sequence[str]) -> dict[str, Any]:
    count = len(samples)
    success_count = sum(1 for sample in samples if bool(sample.get("success", False)))
    aggregate: dict[str, Any] = {
        "sample_count": count,
        "success_count": success_count,
        "success_rate": success_count / count if count else 0.0,
        "status_counts": _status_counts(samples),
    }
    for key in keys:
        aggregate[f"mean_{key}"] = _mean(samples, key)
    return aggregate


def _profile_metrics(profile: DensityProfile) -> dict[str, float]:
    return {
        "initial_density": profile.initial_density,
        "final_density": profile.final_density,
        "max_density": profile.max_density,
        "fill_in_delta": profile.fill_in_delta,
    }


def _time_call(callback: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    value = callback()
    return value, time.perf_counter() - start


def _run_leftmost_sample(sample: BenchmarkSample, modulus: int) -> MetricRecord:
    matrix = sample.matrix
    result, teacher_seconds = _time_call(lambda: rref_leftmost(matrix, modulus))
    replayed, replay_seconds = _time_call(lambda: replay_row_ops(matrix, result.ops, modulus))
    final_is_rref, predicate_seconds = _time_call(
        lambda: is_rref_modp(result.final_matrix, modulus)
    )
    profile = row_op_density_profile(matrix, result.ops, modulus)
    replay_ok = replayed == result.final_matrix
    success = replay_ok and final_is_rref
    return {
        "sample_index": sample.sample_index,
        "seed": sample.seed,
        "status": "success" if success else "verification_failed",
        "success": success,
        "trace_length": len(result.ops),
        "rank": len(result.pivots),
        "replay_ok": replay_ok,
        "final_is_rref": final_is_rref,
        **_profile_metrics(profile),
        "wall_time_seconds": teacher_seconds + replay_seconds + predicate_seconds,
        "teacher_wall_time_seconds": teacher_seconds,
        "replay_wall_time_seconds": replay_seconds,
        "predicate_wall_time_seconds": predicate_seconds,
    }


def _run_neural_sample(
    sample: BenchmarkSample,
    *,
    config: RREFBenchmarkConfig,
    model_data_path: str | Path,
) -> MetricRecord:
    matrix = sample.matrix
    rollout_config = RREFPivotRolloutConfig(
        data_path=model_data_path,
        checkpoint_dir=cast(str | Path, config.checkpoint_dir),
        max_steps=config.max_steps,
        hidden_sizes=config.hidden_sizes,
    )
    result, rollout_seconds = _time_call(lambda: rollout_rref_pivot(rollout_config, matrix))
    replayed, replay_seconds = _time_call(
        lambda: replay_row_ops(result.initial_matrix, result.ops, result.modulus)
    )
    final_is_rref, predicate_seconds = _time_call(
        lambda: is_rref_modp(result.final_matrix, result.modulus)
    )
    profile = row_op_density_profile(result.initial_matrix, result.ops, result.modulus)
    replay_ok = replayed == result.final_matrix
    success = result.success and replay_ok and final_is_rref
    return {
        "sample_index": sample.sample_index,
        "seed": sample.seed,
        "status": result.status,
        "success": success,
        "step_count": result.step_count,
        "invalid_action_count": result.invalid_action_count,
        "masked_action_count": result.masked_action_count,
        "invalid_action_breakdown": dict(result.invalid_action_breakdown),
        "checkpoint_step": result.checkpoint_step,
        "replay_ok": replay_ok,
        "final_is_rref": final_is_rref,
        **_profile_metrics(profile),
        "wall_time_seconds": rollout_seconds + replay_seconds + predicate_seconds,
        "rollout_wall_time_seconds": rollout_seconds,
        "replay_wall_time_seconds": replay_seconds,
        "predicate_wall_time_seconds": predicate_seconds,
    }


def run_rref_benchmark(config: RREFBenchmarkConfig) -> dict[str, Any]:
    samples, shard_samples = _input_samples(config)
    benchmark_rows, benchmark_cols, benchmark_modulus = _benchmark_shape_modulus(
        config,
        shard_samples,
    )
    model_data_path = _validate_model_metadata(config, shard_samples)

    leftmost_samples = [
        _run_leftmost_sample(sample, benchmark_modulus) for sample in samples
    ]
    policies: dict[str, Any] = {
        "leftmost": {
            "aggregate": _aggregate_samples(
                leftmost_samples,
                (
                    "trace_length",
                    "rank",
                    "initial_density",
                    "final_density",
                    "max_density",
                    "fill_in_delta",
                    "wall_time_seconds",
                    "teacher_wall_time_seconds",
                    "replay_wall_time_seconds",
                    "predicate_wall_time_seconds",
                ),
            ),
            "samples": leftmost_samples,
        }
    }

    if model_data_path is not None:
        neural_samples = [
            _run_neural_sample(sample, config=config, model_data_path=model_data_path)
            for sample in samples
        ]
        policies["neural"] = {
            "aggregate": _aggregate_samples(
                neural_samples,
                (
                    "step_count",
                    "invalid_action_count",
                    "masked_action_count",
                    "initial_density",
                    "final_density",
                    "max_density",
                    "fill_in_delta",
                    "wall_time_seconds",
                    "rollout_wall_time_seconds",
                    "replay_wall_time_seconds",
                    "predicate_wall_time_seconds",
                ),
            ),
            "samples": neural_samples,
        }

    return {
        "status": "ok",
        "source": config.source,
        "count": len(samples),
        "rows": benchmark_rows,
        "cols": benchmark_cols,
        "modulus": benchmark_modulus,
        "policies": policies,
    }
