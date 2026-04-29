import json
from typing import Any, cast

import click

from nf_agent.benchmarks import (
    HNFBenchmarkConfig,
    RREFBenchmarkConfig,
    SNFBenchmarkConfig,
    run_hnf_benchmark,
    run_rref_benchmark,
    run_snf_benchmark,
)
from nf_agent.benchmarks.rref_benchmark import BenchmarkSource, MatrixFamily
from nf_agent.data.hnf_shards import write_hnf_shard
from nf_agent.data.matrix_families import dense_random_matrix, sparse_random_matrix
from nf_agent.data.rref_shards import write_rref_shard
from nf_agent.env.rref_modp import RowOp, is_rref_modp, replay_row_ops, rref_leftmost
from nf_agent.experiments import HNFV08ExperimentConfig, run_hnf_v08_experiment
from nf_agent.reports import BenchmarkReportConfig, build_benchmark_report
from nf_agent.rollout import (
    HNFRolloutConfig,
    RREFPivotRolloutConfig,
    rollout_hnf_beam_sample,
    rollout_hnf_policy_sample,
    rollout_rref_pivot_sample,
)
from nf_agent.train import (
    HNFActorCriticConfig,
    HNFDaggerConfig,
    HNFTrainConfig,
    TrainConfig,
    train_hnf_actor_critic,
    train_hnf_dagger,
    train_hnf_policy,
    train_rref_pivot,
)


def _row_op_to_dict(op: RowOp) -> dict[str, int | str]:
    payload: dict[str, int | str] = {"kind": op.kind, "target": op.target}
    if op.source is not None:
        payload["source"] = op.source
    if op.scalar is not None:
        payload["scalar"] = op.scalar
    return payload


def _emit_json(payload: dict[str, Any]) -> None:
    click.echo(json.dumps(payload, sort_keys=True))


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Verifier-guided normal-form agent tools."""


@main.group()
def data() -> None:
    """Generate exact matrix data."""


@data.command("sample")
@click.option("--rows", type=int, required=True)
@click.option("--cols", type=int, required=True)
@click.option("--p", "modulus", type=int, default=101, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--density", type=float, default=None)
def data_sample(rows: int, cols: int, modulus: int, seed: int, density: float | None) -> None:
    if density is None:
        matrix = dense_random_matrix(rows=rows, cols=cols, p=modulus, seed=seed)
        family = "dense"
    else:
        matrix = sparse_random_matrix(rows=rows, cols=cols, p=modulus, density=density, seed=seed)
        family = "sparse"
    _emit_json({"family": family, "modulus": modulus, "matrix": matrix})


@data.command("make-rref-shard")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option("--count", type=int, required=True)
@click.option("--seed-start", type=int, default=0, show_default=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
def make_rref_shard(config_path: str, count: int, seed_start: int, out_path: str) -> None:
    try:
        write_rref_shard(
            config_path=config_path,
            count=count,
            seed_start=seed_start,
            out_path=out_path,
        )
    except (TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_json({"status": "ok", "out": out_path, "count": count, "seed_start": seed_start})


@data.command("make-hnf-shard")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option("--count", type=int, required=True)
@click.option("--seed-start", type=int, default=0, show_default=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
def make_hnf_shard(config_path: str, count: int, seed_start: int, out_path: str) -> None:
    try:
        write_hnf_shard(
            config_path=config_path,
            count=count,
            seed_start=seed_start,
            out_path=out_path,
        )
    except (TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_json({"status": "ok", "out": out_path, "count": count, "seed_start": seed_start})


@main.group()
def train() -> None:
    """Training commands."""


@train.command("status")
def train_status() -> None:
    _emit_json({"status": "not_implemented", "reason": "v0.2 roadmap"})


@train.command("rref-pivot")
@click.option("--data", "data_path", type=click.Path(dir_okay=False), required=True)
@click.option("--steps", type=int, required=True)
@click.option("--batch-size", type=int, required=True)
@click.option("--learning-rate", type=float, default=0.001, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--out", "out_dir", type=click.Path(file_okay=False), required=True)
@click.option(
    "--hidden-size",
    "hidden_sizes",
    type=int,
    multiple=True,
    help="Hidden layer width. Repeat for multiple layers; default is 256,256.",
)
def train_rref_pivot_cli(
    data_path: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    out_dir: str,
    hidden_sizes: tuple[int, ...],
) -> None:
    try:
        result = train_rref_pivot(
            TrainConfig(
                data_path=data_path,
                steps=steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                out_dir=out_dir,
                hidden_sizes=hidden_sizes or (256, 256),
            )
        )
    except (TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_json(result)


@train.command("hnf-policy")
@click.option("--data", "data_path", type=click.Path(dir_okay=False), required=True)
@click.option("--steps", type=int, required=True)
@click.option("--batch-size", type=int, required=True)
@click.option("--learning-rate", type=float, default=0.001, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--out", "out_dir", type=click.Path(file_okay=False), required=True)
@click.option(
    "--hidden-size",
    "hidden_sizes",
    type=int,
    multiple=True,
    help="Hidden layer width. Repeat for multiple layers; default is 256,256.",
)
def train_hnf_policy_cli(
    data_path: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    out_dir: str,
    hidden_sizes: tuple[int, ...],
) -> None:
    try:
        result = train_hnf_policy(
            HNFTrainConfig(
                data_path=data_path,
                steps=steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                out_dir=out_dir,
                hidden_sizes=hidden_sizes or (256, 256),
            )
        )
    except (TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_json(result)


@train.command("hnf-dagger")
@click.option("--data", "data_path", type=click.Path(dir_okay=False), required=True)
@click.option("--iterations", type=int, required=True)
@click.option("--train-steps", type=int, required=True)
@click.option("--batch-size", type=int, required=True)
@click.option("--learning-rate", type=float, default=0.001, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--out", "out_dir", type=click.Path(file_okay=False), required=True)
@click.option("--rollout-sample-count", type=int, default=16, show_default=True)
@click.option("--rollout-max-steps", type=int, default=None)
@click.option("--hidden-size", "hidden_sizes", type=int, multiple=True)
def train_hnf_dagger_cli(
    data_path: str,
    iterations: int,
    train_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    out_dir: str,
    rollout_sample_count: int,
    rollout_max_steps: int | None,
    hidden_sizes: tuple[int, ...],
) -> None:
    try:
        result = train_hnf_dagger(
            HNFDaggerConfig(
                data_path=data_path,
                iterations=iterations,
                train_steps=train_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                out_dir=out_dir,
                hidden_sizes=hidden_sizes or (256, 256),
                rollout_sample_count=rollout_sample_count,
                rollout_max_steps=rollout_max_steps,
            )
        )
    except (TypeError, ValueError, IndexError) as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_json(result)


@train.command("hnf-actor-critic")
@click.option("--data", "data_path", type=click.Path(dir_okay=False), required=True)
@click.option("--checkpoint", "checkpoint_dir", type=click.Path(file_okay=False), required=True)
@click.option("--steps", type=int, required=True)
@click.option("--batch-size", type=int, required=True)
@click.option("--learning-rate", type=float, default=0.0005, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--out", "out_dir", type=click.Path(file_okay=False), required=True)
@click.option("--rollout-max-steps", type=int, default=None)
@click.option("--hidden-size", "hidden_sizes", type=int, multiple=True)
def train_hnf_actor_critic_cli(
    data_path: str,
    checkpoint_dir: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    out_dir: str,
    rollout_max_steps: int | None,
    hidden_sizes: tuple[int, ...],
) -> None:
    try:
        result = train_hnf_actor_critic(
            HNFActorCriticConfig(
                data_path=data_path,
                checkpoint_dir=checkpoint_dir,
                steps=steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                out_dir=out_dir,
                hidden_sizes=hidden_sizes or (256, 256),
                rollout_max_steps=rollout_max_steps,
            )
        )
    except (TypeError, ValueError, IndexError) as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_json(result)


@main.group()
def rollout() -> None:
    """Run explicit rollout policies."""


@rollout.command("rref")
@click.option("--rows", type=int, required=True)
@click.option("--cols", type=int, required=True)
@click.option("--p", "modulus", type=int, default=101, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--teacher", type=click.Choice(["leftmost"]), required=True)
def rollout_rref(rows: int, cols: int, modulus: int, seed: int, teacher: str) -> None:
    if teacher != "leftmost":
        raise click.ClickException(f"unknown explicit teacher: {teacher}")
    matrix = dense_random_matrix(rows=rows, cols=cols, p=modulus, seed=seed)
    result = rref_leftmost(matrix, modulus)
    _emit_json(
        {
            "input": matrix,
            "final": result.final_matrix,
            "modulus": result.modulus,
            "ops": [_row_op_to_dict(op) for op in result.ops],
            "pivots": [{"row": pivot.row, "col": pivot.col} for pivot in result.pivots],
        }
    )


@rollout.command("rref-neural")
@click.option("--data", "data_path", type=click.Path(dir_okay=False), required=True)
@click.option("--checkpoint", "checkpoint_dir", type=click.Path(file_okay=False), required=True)
@click.option("--sample-index", type=int, required=True)
@click.option("--max-steps", type=int, default=None)
@click.option(
    "--hidden-size",
    "hidden_sizes",
    type=int,
    multiple=True,
    help="Hidden layer width. Repeat for multiple layers; default is 256,256.",
)
def rollout_rref_neural(
    data_path: str,
    checkpoint_dir: str,
    sample_index: int,
    max_steps: int | None,
    hidden_sizes: tuple[int, ...],
) -> None:
    try:
        result = rollout_rref_pivot_sample(
            RREFPivotRolloutConfig(
                data_path=data_path,
                checkpoint_dir=checkpoint_dir,
                max_steps=max_steps,
                hidden_sizes=hidden_sizes or (256, 256),
                sample_index=sample_index,
            )
        )
    except (TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_json(result.as_json_dict())


@rollout.command("hnf-neural")
@click.option("--data", "data_path", type=click.Path(dir_okay=False), required=True)
@click.option("--checkpoint", "checkpoint_dir", type=click.Path(file_okay=False), required=True)
@click.option("--sample-index", type=int, required=True)
@click.option("--max-steps", type=int, default=None)
@click.option("--hidden-size", "hidden_sizes", type=int, multiple=True)
def rollout_hnf_neural(
    data_path: str,
    checkpoint_dir: str,
    sample_index: int,
    max_steps: int | None,
    hidden_sizes: tuple[int, ...],
) -> None:
    try:
        result = rollout_hnf_policy_sample(
            HNFRolloutConfig(
                data_path=data_path,
                checkpoint_dir=checkpoint_dir,
                sample_index=sample_index,
                max_steps=max_steps,
                hidden_sizes=hidden_sizes or (256, 256),
            )
        )
    except (TypeError, ValueError, IndexError) as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_json(result.as_json_dict())


@rollout.command("hnf-beam")
@click.option("--data", "data_path", type=click.Path(dir_okay=False), required=True)
@click.option("--checkpoint", "checkpoint_dir", type=click.Path(file_okay=False), required=True)
@click.option("--sample-index", type=int, required=True)
@click.option("--max-steps", type=int, default=None)
@click.option("--beam-width", type=int, default=8, show_default=True)
@click.option("--hidden-size", "hidden_sizes", type=int, multiple=True)
def rollout_hnf_beam(
    data_path: str,
    checkpoint_dir: str,
    sample_index: int,
    max_steps: int | None,
    beam_width: int,
    hidden_sizes: tuple[int, ...],
) -> None:
    try:
        result = rollout_hnf_beam_sample(
            HNFRolloutConfig(
                data_path=data_path,
                checkpoint_dir=checkpoint_dir,
                sample_index=sample_index,
                max_steps=max_steps,
                hidden_sizes=hidden_sizes or (256, 256),
                beam_width=beam_width,
            )
        )
    except (TypeError, ValueError, IndexError) as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_json(result.as_json_dict())


@main.group()
def benchmark() -> None:
    """Benchmark exact environments and rollout policies."""


@benchmark.command("smoke")
def benchmark_smoke() -> None:
    matrix = dense_random_matrix(rows=4, cols=4, p=101, seed=0)
    result = rref_leftmost(matrix, 101)
    _emit_json({"matrix_count": 1, "trace_length": len(result.ops), "rank": len(result.pivots)})


@benchmark.command("rref")
@click.option("--source", type=click.Choice(["generated", "shard"]), required=True)
@click.option("--count", type=int, default=None)
@click.option("--rows", type=int, default=None)
@click.option("--cols", type=int, default=None)
@click.option("--p", "modulus", type=int, default=101, show_default=True)
@click.option(
    "--family",
    type=click.Choice(["dense", "sparse", "low_rank"]),
    default="dense",
    show_default=True,
)
@click.option("--seed-start", type=int, default=0, show_default=True)
@click.option("--density", type=float, default=None)
@click.option("--rank", type=int, default=None)
@click.option("--data", "data_path", type=click.Path(dir_okay=False), default=None)
@click.option("--model-data", "model_data_path", type=click.Path(dir_okay=False), default=None)
@click.option("--checkpoint", "checkpoint_dir", type=click.Path(file_okay=False), default=None)
@click.option("--max-steps", type=int, default=None)
@click.option(
    "--hidden-size",
    "hidden_sizes",
    type=int,
    multiple=True,
    help="Hidden layer width. Repeat for multiple layers; default is 256,256.",
)
def benchmark_rref(
    source: str,
    count: int | None,
    rows: int | None,
    cols: int | None,
    modulus: int,
    family: str,
    seed_start: int,
    density: float | None,
    rank: int | None,
    data_path: str | None,
    model_data_path: str | None,
    checkpoint_dir: str | None,
    max_steps: int | None,
    hidden_sizes: tuple[int, ...],
) -> None:
    try:
        result = run_rref_benchmark(
            RREFBenchmarkConfig(
                source=cast(BenchmarkSource, source),
                count=count,
                rows=rows,
                cols=cols,
                modulus=modulus,
                family=cast(MatrixFamily, family),
                seed_start=seed_start,
                density=density,
                rank=rank,
                data_path=data_path,
                model_data_path=model_data_path,
                checkpoint_dir=checkpoint_dir,
                max_steps=max_steps,
                hidden_sizes=hidden_sizes or (256, 256),
            )
        )
    except (TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_json(result)


@benchmark.command("hnf")
@click.option("--rows", type=int, required=True)
@click.option("--cols", type=int, required=True)
@click.option("--count", type=int, required=True)
@click.option("--density", type=float, default=0.2, show_default=True)
@click.option("--entry-bound", type=int, default=9, show_default=True)
@click.option("--seed-start", type=int, default=0, show_default=True)
@click.option("--model-data", "model_data_path", type=click.Path(dir_okay=False), default=None)
@click.option(
    "--supervised-checkpoint",
    "supervised_checkpoint_dir",
    type=click.Path(file_okay=False),
    default=None,
)
@click.option(
    "--dagger-checkpoint",
    "dagger_checkpoint_dir",
    type=click.Path(file_okay=False),
    default=None,
)
@click.option(
    "--actor-critic-checkpoint",
    "actor_critic_checkpoint_dir",
    type=click.Path(file_okay=False),
    default=None,
)
@click.option(
    "--beam-checkpoint",
    "beam_checkpoint_dir",
    type=click.Path(file_okay=False),
    default=None,
)
@click.option("--max-steps", type=int, default=None)
@click.option("--beam-width", type=int, default=8, show_default=True)
@click.option("--hidden-size", "hidden_sizes", type=int, multiple=True)
def benchmark_hnf(
    rows: int,
    cols: int,
    count: int,
    density: float,
    entry_bound: int,
    seed_start: int,
    model_data_path: str | None,
    supervised_checkpoint_dir: str | None,
    dagger_checkpoint_dir: str | None,
    actor_critic_checkpoint_dir: str | None,
    beam_checkpoint_dir: str | None,
    max_steps: int | None,
    beam_width: int,
    hidden_sizes: tuple[int, ...],
) -> None:
    try:
        result = run_hnf_benchmark(
            HNFBenchmarkConfig(
                count=count,
                rows=rows,
                cols=cols,
                density=density,
                entry_bound=entry_bound,
                seed_start=seed_start,
                model_data_path=model_data_path,
                supervised_checkpoint_dir=supervised_checkpoint_dir,
                dagger_checkpoint_dir=dagger_checkpoint_dir,
                actor_critic_checkpoint_dir=actor_critic_checkpoint_dir,
                beam_checkpoint_dir=beam_checkpoint_dir,
                max_steps=max_steps,
                hidden_sizes=hidden_sizes or (256, 256),
                beam_width=beam_width,
            )
        )
    except (TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_json(result)


@benchmark.command("snf")
@click.option("--rows", type=int, required=True)
@click.option("--cols", type=int, required=True)
@click.option("--count", type=int, required=True)
@click.option("--diagonal-factor-bound", type=int, default=5, show_default=True)
@click.option("--row-op-count", type=int, default=2, show_default=True)
@click.option("--col-op-count", type=int, default=2, show_default=True)
@click.option("--op-scalar-bound", type=int, default=3, show_default=True)
@click.option("--seed-start", type=int, default=0, show_default=True)
def benchmark_snf(
    rows: int,
    cols: int,
    count: int,
    diagonal_factor_bound: int,
    row_op_count: int,
    col_op_count: int,
    op_scalar_bound: int,
    seed_start: int,
) -> None:
    try:
        result = run_snf_benchmark(
            SNFBenchmarkConfig(
                count=count,
                rows=rows,
                cols=cols,
                diagonal_factor_bound=diagonal_factor_bound,
                row_op_count=row_op_count,
                col_op_count=col_op_count,
                op_scalar_bound=op_scalar_bound,
                seed_start=seed_start,
            )
        )
    except (TypeError, ValueError, IndexError) as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_json(result)


@main.group()
def experiment() -> None:
    """Run reproducible experiment bundles."""


@experiment.command("hnf-v08")
@click.option("--out-dir", type=click.Path(file_okay=False), required=True)
@click.option("--samples-per-size", type=int, default=256, show_default=True)
@click.option("--run-seed-count", type=int, default=5, show_default=True)
@click.option("--sizes", "sizes", type=int, multiple=True, required=True)
@click.option("--density", type=float, default=0.2, show_default=True)
@click.option("--entry-bound", type=int, default=5, show_default=True)
@click.option("--train-steps", type=int, default=2, show_default=True)
@click.option("--dagger-iterations", type=int, default=1, show_default=True)
@click.option("--actor-critic-steps", type=int, default=1, show_default=True)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--learning-rate", type=float, default=0.001, show_default=True)
@click.option("--benchmark-max-steps", type=int, default=2, show_default=True)
@click.option("--beam-width", type=int, default=8, show_default=True)
@click.option("--hidden-size", "hidden_sizes", type=int, multiple=True)
@click.option("--allow-threshold-failure", is_flag=True)
def experiment_hnf_v08(
    out_dir: str,
    samples_per_size: int,
    run_seed_count: int,
    sizes: tuple[int, ...],
    density: float,
    entry_bound: int,
    train_steps: int,
    dagger_iterations: int,
    actor_critic_steps: int,
    batch_size: int,
    learning_rate: float,
    benchmark_max_steps: int | None,
    beam_width: int,
    hidden_sizes: tuple[int, ...],
    allow_threshold_failure: bool,
) -> None:
    try:
        result = run_hnf_v08_experiment(
            HNFV08ExperimentConfig(
                out_dir=out_dir,
                samples_per_size=samples_per_size,
                run_seed_count=run_seed_count,
                sizes=sizes,
                density=density,
                entry_bound=entry_bound,
                train_steps=train_steps,
                dagger_iterations=dagger_iterations,
                actor_critic_steps=actor_critic_steps,
                batch_size=batch_size,
                hidden_sizes=hidden_sizes or (64,),
                learning_rate=learning_rate,
                benchmark_max_steps=benchmark_max_steps,
                beam_width=beam_width,
                allow_threshold_failure=allow_threshold_failure,
            )
        )
    except (OSError, TypeError, ValueError, IndexError) as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_json(result)


@main.group()
def report() -> None:
    """Report generation commands."""


@report.command("status")
def report_status() -> None:
    _emit_json({"status": "implemented", "commands": ["benchmark", "rref-certificate"]})


@report.command("benchmark")
@click.option("--out-dir", type=click.Path(file_okay=False), required=True)
@click.option(
    "--input-json",
    "input_json_paths",
    type=click.Path(exists=True, dir_okay=False),
    multiple=True,
)
@click.option("--suite", type=str, default="paper-smoke", show_default=True)
@click.option("--sample-count", type=int, default=16, show_default=True)
@click.option("--rows", type=int, default=8, show_default=True)
@click.option("--cols", type=int, default=8, show_default=True)
@click.option("--p", "modulus", type=int, default=101, show_default=True)
@click.option("--seed-start", type=int, default=0, show_default=True)
@click.option("--sparse-density", type=float, default=0.2, show_default=True)
@click.option("--low-rank", type=int, default=3, show_default=True)
@click.option("--hnf-entry-bound", type=int, default=9, show_default=True)
@click.option("--rref-checkpoint", type=click.Path(file_okay=False), default=None)
@click.option("--rref-model-data", type=click.Path(dir_okay=False), default=None)
def report_benchmark(
    out_dir: str,
    input_json_paths: tuple[str, ...],
    suite: str,
    sample_count: int,
    rows: int,
    cols: int,
    modulus: int,
    seed_start: int,
    sparse_density: float,
    low_rank: int,
    hnf_entry_bound: int,
    rref_checkpoint: str | None,
    rref_model_data: str | None,
) -> None:
    try:
        result = build_benchmark_report(
            BenchmarkReportConfig(
                out_dir=out_dir,
                input_json_paths=input_json_paths,
                suite=suite,
                sample_count=sample_count,
                rows=rows,
                cols=cols,
                modulus=modulus,
                seed_start=seed_start,
                sparse_density=sparse_density,
                low_rank=low_rank,
                hnf_entry_bound=hnf_entry_bound,
                rref_checkpoint=rref_checkpoint,
                rref_model_data=rref_model_data,
            )
        )
    except (OSError, TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_json(result)


@report.command("rref-certificate")
@click.option("--rows", type=int, required=True)
@click.option("--cols", type=int, required=True)
@click.option("--p", "modulus", type=int, default=101, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--teacher", type=click.Choice(["leftmost"]), required=True)
def report_rref_certificate(rows: int, cols: int, modulus: int, seed: int, teacher: str) -> None:
    """Emit a small RREF JSON certificate for Lean checker tests."""
    try:
        if teacher != "leftmost":
            raise ValueError(f"unknown explicit teacher: {teacher}")
        matrix = dense_random_matrix(rows=rows, cols=cols, p=modulus, seed=seed)
        result = rref_leftmost(matrix, modulus)
        if replay_row_ops(matrix, result.ops, modulus) != result.final_matrix:
            raise ValueError("generated RREF certificate does not replay")
        if not is_rref_modp(result.final_matrix, modulus):
            raise ValueError("generated final matrix is not RREF")
    except (TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        raise click.ClickException(str(exc)) from exc

    _emit_json(
        {
            "kind": "rref_modp",
            "modulus": result.modulus,
            "shape": [rows, cols],
            "input": matrix,
            "ops": [_row_op_to_dict(op) for op in result.ops],
            "final": result.final_matrix,
            "pivots": [{"row": pivot.row, "col": pivot.col} for pivot in result.pivots],
        }
    )
