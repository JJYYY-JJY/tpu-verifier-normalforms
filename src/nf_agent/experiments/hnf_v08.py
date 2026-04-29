from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from nf_agent.benchmarks import HNFBenchmarkConfig, run_hnf_benchmark
from nf_agent.data.hnf_shards import write_hnf_shard
from nf_agent.train import (
    HNFActorCriticConfig,
    HNFDaggerConfig,
    HNFTrainConfig,
    train_hnf_actor_critic,
    train_hnf_dagger,
    train_hnf_policy,
)

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class HNFV08ExperimentConfig:
    out_dir: str | Path
    samples_per_size: int = 256
    run_seed_count: int = 5
    sizes: tuple[int, ...] = (4, 6, 8)
    density: float = 0.2
    entry_bound: int = 5
    train_steps: int = 2
    dagger_iterations: int = 1
    actor_critic_steps: int = 1
    batch_size: int = 16
    hidden_sizes: tuple[int, ...] = (64,)
    learning_rate: float = 0.001
    benchmark_max_steps: int | None = 2
    beam_width: int = 8
    allow_threshold_failure: bool = False


def run_hnf_v08_experiment(config: HNFV08ExperimentConfig) -> JsonDict:
    _validate_config(config)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmarks_dir = out_dir / "benchmarks"
    configs_dir = out_dir / "configs"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    benchmark_payloads: list[JsonDict] = []
    size_rows: list[JsonDict] = []
    run_rows: list[JsonDict] = []
    for size in config.sizes:
        size_dir = out_dir / f"size_{size}"
        size_dir.mkdir(parents=True, exist_ok=True)
        config_path = configs_dir / f"hnf_sparse_{size}x{size}.yaml"
        _write_size_config(config_path, size=size, config=config)
        shard_path = size_dir / "teacher.npz"
        write_hnf_shard(
            config_path=config_path,
            count=config.samples_per_size,
            seed_start=0,
            out_path=shard_path,
        )
        supervised_dir = size_dir / "supervised"
        train_hnf_policy(
            HNFTrainConfig(
                data_path=shard_path,
                steps=config.train_steps,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                seed=0,
                out_dir=supervised_dir,
                hidden_sizes=config.hidden_sizes,
            )
        )
        dagger = train_hnf_dagger(
            HNFDaggerConfig(
                data_path=shard_path,
                iterations=config.dagger_iterations,
                train_steps=config.train_steps,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                seed=0,
                out_dir=size_dir / "dagger",
                hidden_sizes=config.hidden_sizes,
                rollout_sample_count=min(16, config.samples_per_size),
                rollout_max_steps=config.benchmark_max_steps,
            )
        )
        actor = train_hnf_actor_critic(
            HNFActorCriticConfig(
                data_path=dagger["aggregate_data_path"],
                checkpoint_dir=dagger["checkpoint_dir"],
                steps=config.actor_critic_steps,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                seed=0,
                out_dir=size_dir / "actor_critic",
                hidden_sizes=config.hidden_sizes,
                rollout_max_steps=config.benchmark_max_steps,
            )
        )
        supervised_aggregate_dir = size_dir / "supervised_aggregate"
        train_hnf_policy(
            HNFTrainConfig(
                data_path=dagger["aggregate_data_path"],
                steps=config.train_steps,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                seed=0,
                out_dir=supervised_aggregate_dir,
                hidden_sizes=config.hidden_sizes,
            )
        )
        for run_seed in range(config.run_seed_count):
            seed_start = run_seed * config.samples_per_size
            benchmark = run_hnf_benchmark(
                HNFBenchmarkConfig(
                    count=config.samples_per_size,
                    rows=size,
                    cols=size,
                    density=config.density,
                    entry_bound=config.entry_bound,
                    seed_start=seed_start,
                    model_data_path=dagger["aggregate_data_path"],
                    supervised_checkpoint_dir=supervised_aggregate_dir,
                    dagger_checkpoint_dir=dagger["checkpoint_dir"],
                    actor_critic_checkpoint_dir=actor["checkpoint_dir"],
                    beam_checkpoint_dir=actor["checkpoint_dir"],
                    max_steps=config.benchmark_max_steps,
                    hidden_sizes=config.hidden_sizes,
                    beam_width=config.beam_width,
                )
            )
            benchmark_path = benchmarks_dir / f"hnf_{size}x{size}_seed_{run_seed}.json"
            benchmark_path.write_text(json.dumps(benchmark, indent=2, sort_keys=True))
            benchmark_payloads.append(benchmark)
            run_rows.append(_run_row(size=size, run_seed=run_seed, benchmark=benchmark))

        size_rows.append(_size_row(size=size, run_rows=run_rows))

    verdict = _threshold_verdict(size_rows)
    metrics: JsonDict = {
        "schema_version": "hnf-v08-experiment-metrics-v1",
        "status": "ok" if verdict["passed"] else "failed_threshold",
        "generated_at_utc": datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "config": _config_json(config),
        "threshold": {
            "success_rate_delta": 0.05,
            "mean_step_count_ratio": 0.90,
            "baseline_policy": "supervised_greedy",
            "candidate_policy": "dagger_actor_critic_beam",
        },
        "threshold_verdict": verdict,
        "size_rows": size_rows,
        "run_rows": run_rows,
        "benchmark_count": len(benchmark_payloads),
    }
    _write_plots(out_dir, size_rows)
    metrics_path = out_dir / "metrics.json"
    report_path = out_dir / "report.md"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    report_path.write_text(_render_report(metrics))
    return {
        "status": metrics["status"],
        "out_dir": str(out_dir),
        "report_md": str(report_path),
        "metrics_json": str(metrics_path),
        "threshold_passed": verdict["passed"],
        "size_count": len(config.sizes),
        "run_seed_count": config.run_seed_count,
    }


def _validate_config(config: HNFV08ExperimentConfig) -> None:
    if config.samples_per_size <= 0:
        raise ValueError("samples_per_size must be positive")
    if config.run_seed_count <= 0:
        raise ValueError("run_seed_count must be positive")
    if not config.sizes:
        raise ValueError("sizes must be non-empty")
    if any(size <= 0 for size in config.sizes):
        raise ValueError("sizes entries must be positive")
    if not 0.0 <= config.density <= 1.0:
        raise ValueError("density must lie in [0, 1]")
    if config.entry_bound <= 0:
        raise ValueError("entry_bound must be positive")
    if config.train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if config.dagger_iterations <= 0:
        raise ValueError("dagger_iterations must be positive")
    if config.actor_critic_steps <= 0:
        raise ValueError("actor_critic_steps must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.beam_width <= 0:
        raise ValueError("beam_width must be positive")


def _write_size_config(path: Path, *, size: int, config: HNFV08ExperimentConfig) -> None:
    path.write_text(
        "task: hnf\n"
        "integer_matrix:\n"
        "  family: sparse\n"
        f"  rows: {size}\n"
        f"  cols: {size}\n"
        f"  density: {config.density}\n"
        f"  entry_bound: {config.entry_bound}\n"
    )


def _aggregate(policy: Mapping[str, Any]) -> Mapping[str, Any]:
    aggregate = policy.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError("benchmark policy is missing aggregate")
    return aggregate


def _run_row(*, size: int, run_seed: int, benchmark: Mapping[str, Any]) -> JsonDict:
    policies = benchmark.get("policies")
    if not isinstance(policies, Mapping):
        raise ValueError("benchmark is missing policies")
    supervised = _aggregate(cast(Mapping[str, Any], policies["supervised_greedy"]))
    beam = _aggregate(cast(Mapping[str, Any], policies["beam"]))
    return {
        "size": size,
        "run_seed": run_seed,
        "supervised_success_rate": float(supervised.get("success_rate", 0.0)),
        "beam_success_rate": float(beam.get("success_rate", 0.0)),
        "supervised_mean_step_count": float(supervised.get("mean_step_count", 0.0)),
        "beam_mean_step_count": float(beam.get("mean_step_count", 0.0)),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _size_row(*, size: int, run_rows: list[JsonDict]) -> JsonDict:
    rows = [row for row in run_rows if row["size"] == size]
    return {
        "size": size,
        "supervised_success_rate": _mean([row["supervised_success_rate"] for row in rows]),
        "beam_success_rate": _mean([row["beam_success_rate"] for row in rows]),
        "supervised_mean_step_count": _mean(
            [row["supervised_mean_step_count"] for row in rows]
        ),
        "beam_mean_step_count": _mean([row["beam_mean_step_count"] for row in rows]),
    }


def _threshold_verdict(size_rows: list[JsonDict]) -> JsonDict:
    per_size: list[JsonDict] = []
    for row in size_rows:
        baseline_success = float(row["supervised_success_rate"])
        beam_success = float(row["beam_success_rate"])
        baseline_steps = float(row["supervised_mean_step_count"])
        beam_steps = float(row["beam_mean_step_count"])
        success_ok = beam_success >= baseline_success + 0.05
        step_ok = baseline_steps > 0.0 and beam_steps <= baseline_steps * 0.90
        per_size.append(
            {
                "size": row["size"],
                "success_ok": success_ok,
                "step_ok": step_ok,
                "passed": success_ok and step_ok,
                "success_delta": beam_success - baseline_success,
                "step_ratio": beam_steps / baseline_steps if baseline_steps > 0.0 else None,
            }
        )
    return {"passed": all(row["passed"] for row in per_size), "per_size": per_size}


def _config_json(config: HNFV08ExperimentConfig) -> JsonDict:
    return {
        "samples_per_size": config.samples_per_size,
        "run_seed_count": config.run_seed_count,
        "sizes": list(config.sizes),
        "density": config.density,
        "entry_bound": config.entry_bound,
        "train_steps": config.train_steps,
        "dagger_iterations": config.dagger_iterations,
        "actor_critic_steps": config.actor_critic_steps,
        "batch_size": config.batch_size,
        "hidden_sizes": list(config.hidden_sizes),
        "learning_rate": config.learning_rate,
        "benchmark_max_steps": config.benchmark_max_steps,
        "beam_width": config.beam_width,
    }


def _write_plots(out_dir: Path, size_rows: list[JsonDict]) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    matplotlib = cast(Any, import_module("matplotlib"))
    matplotlib.use("Agg")
    pyplot = cast(Any, import_module("matplotlib.pyplot"))

    sizes = [str(row["size"]) for row in size_rows]
    positions = list(range(len(sizes)))
    width = 0.35
    fig, ax = pyplot.subplots(figsize=(max(6.0, len(sizes) * 1.5), 4.0))
    ax.bar(
        [position - width / 2 for position in positions],
        [row["supervised_success_rate"] for row in size_rows],
        width,
        label="supervised_greedy",
    )
    ax.bar(
        [position + width / 2 for position in positions],
        [row["beam_success_rate"] for row in size_rows],
        width,
        label="dagger_actor_critic_beam",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("success rate")
    ax.set_title("HNF v0.8 success rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "success_rate.png")
    pyplot.close(fig)

    fig, ax = pyplot.subplots(figsize=(max(6.0, len(sizes) * 1.5), 4.0))
    ax.bar(
        [position - width / 2 for position in positions],
        [row["supervised_mean_step_count"] for row in size_rows],
        width,
        label="supervised_greedy",
    )
    ax.bar(
        [position + width / 2 for position in positions],
        [row["beam_mean_step_count"] for row in size_rows],
        width,
        label="dagger_actor_critic_beam",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("mean step count")
    ax.set_title("HNF v0.8 step count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "mean_step_count.png")
    pyplot.close(fig)


def _render_report(metrics: Mapping[str, Any]) -> str:
    verdict = cast(Mapping[str, Any], metrics["threshold_verdict"])
    size_rows = cast(list[Mapping[str, Any]], metrics["size_rows"])
    lines = [
        "# HNF v0.8 Experiment",
        "",
        f"- Status: `{metrics['status']}`",
        f"- Threshold passed: `{verdict['passed']}`",
        f"- Generated at UTC: `{metrics['generated_at_utc']}`",
        "",
        "## Threshold",
        "",
        "- Candidate: `dagger_actor_critic_beam`",
        "- Baseline: `supervised_greedy`",
        "- Required: success rate >= baseline + 0.05 and mean steps <= baseline * 0.90.",
        "",
        "## Size Aggregates",
        "",
        "| size | supervised_success_rate | beam_success_rate | "
        "supervised_mean_step_count | beam_mean_step_count |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in size_rows:
        lines.append(
            "| "
            f"{row['size']} | "
            f"{float(row['supervised_success_rate']):.6f} | "
            f"{float(row['beam_success_rate']):.6f} | "
            f"{float(row['supervised_mean_step_count']):.6f} | "
            f"{float(row['beam_mean_step_count']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "- success_rate: [plots/success_rate.png](plots/success_rate.png)",
            "- mean_step_count: [plots/mean_step_count.png](plots/mean_step_count.png)",
            "",
            "## Exactness",
            "",
            "HNF row-operation replay and predicates remain exact integer code. "
            "Teacher `row_hnf` is used for shard/oracle data and DAgger aggregation only; "
            "neural greedy and beam benchmark policies report failures directly.",
            "",
        ]
    )
    return "\n".join(lines)
