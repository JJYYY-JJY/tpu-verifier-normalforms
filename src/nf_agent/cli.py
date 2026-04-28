import json
from typing import Any, cast

import click

from nf_agent.benchmarks import RREFBenchmarkConfig, run_rref_benchmark
from nf_agent.benchmarks.rref_benchmark import BenchmarkSource, MatrixFamily
from nf_agent.data.matrix_families import dense_random_matrix, sparse_random_matrix
from nf_agent.data.rref_shards import write_rref_shard
from nf_agent.env.rref_modp import RowOp, rref_leftmost
from nf_agent.rollout import RREFPivotRolloutConfig, rollout_rref_pivot_sample
from nf_agent.train import TrainConfig, train_rref_pivot


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


@main.group()
def report() -> None:
    """Report generation commands."""


@report.command("status")
def report_status() -> None:
    _emit_json({"status": "not_implemented", "reason": "v0.7 roadmap"})
