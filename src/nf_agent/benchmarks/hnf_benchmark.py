from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from nf_agent.benchmarks.rref_benchmark import DensityProfile
from nf_agent.data.hnf_shards import HNFShardSamples
from nf_agent.data.matrix_families import sparse_integer_matrix
from nf_agent.env.elementary_ops import Matrix
from nf_agent.env.hnf_int import (
    CoefficientGrowthMetrics,
    IntegerRowOp,
    is_row_hnf,
    normalize_integer_matrix,
    replay_integer_row_ops,
    row_hnf,
)
from nf_agent.rollout import (
    HNFPolicyRuntime,
    HNFRolloutConfig,
    load_hnf_policy_runtime,
    rollout_hnf_beam_with_runtime,
    rollout_hnf_policy_with_runtime,
)

MetricRecord = dict[str, Any]
T = TypeVar("T")


@dataclass(frozen=True)
class HNFBenchmarkConfig:
    count: int
    rows: int
    cols: int
    density: float = 0.2
    entry_bound: int = 9
    seed_start: int = 0
    model_data_path: str | Path | None = None
    supervised_checkpoint_dir: str | Path | None = None
    dagger_checkpoint_dir: str | Path | None = None
    actor_critic_checkpoint_dir: str | Path | None = None
    beam_checkpoint_dir: str | Path | None = None
    max_steps: int | None = None
    hidden_sizes: tuple[int, ...] = (256, 256)
    beam_width: int = 8


@dataclass(frozen=True)
class BenchmarkSample:
    sample_index: int
    seed: int
    matrix: Matrix


def integer_matrix_density(matrix: Sequence[Sequence[int]]) -> float:
    normalized = normalize_integer_matrix(matrix)
    total = sum(len(row) for row in normalized)
    if total == 0:
        return 0.0
    nonzero = sum(1 for row in normalized for entry in row if entry != 0)
    return nonzero / total


def integer_row_op_density_profile(
    matrix: Sequence[Sequence[int]],
    ops: Sequence[IntegerRowOp],
) -> DensityProfile:
    current = normalize_integer_matrix(matrix)
    states = [[entry for entry in row] for row in current]
    densities = [integer_matrix_density(current)]
    profile_states = [states]
    for op in ops:
        current = replay_integer_row_ops(current, [op])
        profile_states.append([[entry for entry in row] for row in current])
        densities.append(integer_matrix_density(current))
    return DensityProfile(states=profile_states, densities=densities)


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_density(value: float) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError("density must be numeric")
    density = float(value)
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must lie in [0, 1]")
    return density


def _generated_samples(config: HNFBenchmarkConfig) -> list[BenchmarkSample]:
    count = _validate_positive_int(config.count, "count")
    rows = _validate_positive_int(config.rows, "rows")
    cols = _validate_positive_int(config.cols, "cols")
    density = _validate_density(config.density)
    entry_bound = _validate_positive_int(config.entry_bound, "entry_bound")
    return [
        BenchmarkSample(
            sample_index=sample_index,
            seed=seed,
            matrix=sparse_integer_matrix(
                rows=rows,
                cols=cols,
                density=density,
                seed=seed,
                entry_bound=entry_bound,
            ),
        )
        for sample_index, seed in enumerate(range(config.seed_start, config.seed_start + count))
    ]


def _time_call(callback: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    value = callback()
    return value, time.perf_counter() - start


def _mean(samples: Iterable[Mapping[str, Any]], key: str) -> float:
    values = [float(sample[key]) for sample in samples if key in sample]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _status_counts(samples: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(sample["status"]) for sample in samples))


def _max_exact(samples: Iterable[Mapping[str, Any]], key: str) -> int:
    return max((int(sample[key]) for sample in samples if key in sample), default=0)


def _profile_metrics(profile: DensityProfile) -> dict[str, float]:
    return {
        "initial_density": profile.initial_density,
        "final_density": profile.final_density,
        "max_density": profile.max_density,
        "fill_in_delta": profile.fill_in_delta,
    }


def _coefficient_metrics(metrics: CoefficientGrowthMetrics) -> dict[str, int]:
    return {
        "initial_max_abs": metrics.initial_max_abs,
        "max_abs_seen": metrics.max_abs_seen,
        "initial_bitlength": metrics.initial_bitlength,
        "max_bitlength": metrics.max_bitlength,
        "growth_numerator": metrics.growth_numerator,
        "growth_denominator": metrics.growth_denominator,
        "step_count": metrics.step_count,
    }


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
        "trace_length",
        "step_count",
        "invalid_action_count",
        "masked_action_count",
        "initial_density",
        "final_density",
        "max_density",
        "fill_in_delta",
        "wall_time_seconds",
        "hnf_wall_time_seconds",
        "replay_wall_time_seconds",
        "predicate_wall_time_seconds",
    ):
        aggregate[f"mean_{key}"] = _mean(samples, key)
    for key in (
        "initial_max_abs",
        "max_abs_seen",
        "initial_bitlength",
        "max_bitlength",
        "growth_numerator",
        "growth_denominator",
        "step_count",
    ):
        aggregate[f"max_{key}"] = _max_exact(samples, key)
    return aggregate


def _run_sample(sample: BenchmarkSample) -> MetricRecord:
    matrix = sample.matrix
    result, hnf_seconds = _time_call(lambda: row_hnf(matrix))
    replayed, replay_seconds = _time_call(lambda: replay_integer_row_ops(matrix, result.ops))
    final_is_hnf, predicate_seconds = _time_call(lambda: is_row_hnf(result.final_matrix))
    profile = integer_row_op_density_profile(matrix, result.ops)
    replay_ok = replayed == result.final_matrix
    success = replay_ok and final_is_hnf
    return {
        "sample_index": sample.sample_index,
        "seed": sample.seed,
        "status": "success" if success else "verification_failed",
        "success": success,
        "trace_length": len(result.ops),
        "replay_ok": replay_ok,
        "final_is_hnf": final_is_hnf,
        **_profile_metrics(profile),
        **_coefficient_metrics(result.metrics),
        "wall_time_seconds": hnf_seconds + replay_seconds + predicate_seconds,
        "hnf_wall_time_seconds": hnf_seconds,
        "replay_wall_time_seconds": replay_seconds,
        "predicate_wall_time_seconds": predicate_seconds,
    }


def _model_data_path(config: HNFBenchmarkConfig) -> str | Path | None:
    checkpoints = (
        config.supervised_checkpoint_dir,
        config.dagger_checkpoint_dir,
        config.actor_critic_checkpoint_dir,
        config.beam_checkpoint_dir,
    )
    if not any(checkpoint is not None for checkpoint in checkpoints):
        return None
    if config.model_data_path is None:
        raise ValueError("model_data_path is required when HNF checkpoint_dir is provided")
    samples = HNFShardSamples(config.model_data_path)
    if (samples.rows, samples.cols) != (config.rows, config.cols):
        raise ValueError("model data shape must match generated HNF benchmark config")
    return config.model_data_path


def _run_neural_sample(
    sample: BenchmarkSample,
    *,
    runtime: HNFPolicyRuntime,
    rollout_config: HNFRolloutConfig,
    beam: bool = False,
) -> MetricRecord:
    if beam:
        result, rollout_seconds = _time_call(
            lambda: rollout_hnf_beam_with_runtime(runtime, rollout_config, sample.matrix)
        )
    else:
        result, rollout_seconds = _time_call(
            lambda: rollout_hnf_policy_with_runtime(runtime, rollout_config, sample.matrix)
        )
    replayed, replay_seconds = _time_call(
        lambda: replay_integer_row_ops(result.initial_matrix, result.ops)
    )
    final_is_hnf, predicate_seconds = _time_call(lambda: is_row_hnf(result.final_matrix))
    profile = integer_row_op_density_profile(result.initial_matrix, result.ops)
    replay_ok = replayed == result.final_matrix
    success = result.success and replay_ok and final_is_hnf
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
        "final_is_hnf": final_is_hnf,
        **_profile_metrics(profile),
        "wall_time_seconds": rollout_seconds + replay_seconds + predicate_seconds,
        "rollout_wall_time_seconds": rollout_seconds,
        "replay_wall_time_seconds": replay_seconds,
        "predicate_wall_time_seconds": predicate_seconds,
    }


def _policy_from_records(records: list[MetricRecord]) -> dict[str, Any]:
    return {"aggregate": _aggregate_samples(records), "samples": records}


def run_hnf_benchmark(config: HNFBenchmarkConfig) -> dict[str, Any]:
    samples = _generated_samples(config)
    row_hnf_records = [_run_sample(sample) for sample in samples]
    policies: dict[str, Any] = {"row_hnf": _policy_from_records(row_hnf_records)}
    model_data_path = _model_data_path(config)
    if model_data_path is not None:
        optional_policies: tuple[tuple[str, str | Path | None, bool], ...] = (
            ("supervised_greedy", config.supervised_checkpoint_dir, False),
            ("dagger_greedy", config.dagger_checkpoint_dir, False),
            ("actor_critic_greedy", config.actor_critic_checkpoint_dir, False),
            (
                "beam",
                config.beam_checkpoint_dir
                or config.actor_critic_checkpoint_dir
                or config.dagger_checkpoint_dir
                or config.supervised_checkpoint_dir,
                True,
            ),
        )
        for policy_name, checkpoint_dir, use_beam in optional_policies:
            if checkpoint_dir is None:
                continue
            rollout_config = HNFRolloutConfig(
                data_path=model_data_path,
                checkpoint_dir=checkpoint_dir,
                max_steps=config.max_steps,
                hidden_sizes=config.hidden_sizes,
                beam_width=config.beam_width,
            )
            runtime = load_hnf_policy_runtime(rollout_config)
            records = [
                _run_neural_sample(
                    sample,
                    runtime=runtime,
                    rollout_config=rollout_config,
                    beam=use_beam,
                )
                for sample in samples
            ]
            policies[policy_name] = _policy_from_records(records)

    return {
        "status": "ok",
        "source": "generated",
        "family": "sparse_integer",
        "count": len(samples),
        "rows": config.rows,
        "cols": config.cols,
        "density": config.density,
        "entry_bound": config.entry_bound,
        "seed_start": config.seed_start,
        "policies": policies,
        "aggregate": policies["row_hnf"]["aggregate"],
        "samples": policies["row_hnf"]["samples"],
    }
