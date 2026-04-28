import json
from typing import Any

import click

from nf_agent.data.matrix_families import dense_random_matrix, sparse_random_matrix
from nf_agent.env.rref_modp import RowOp, rref_leftmost


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


@main.group()
def train() -> None:
    """Training commands."""


@train.command("status")
def train_status() -> None:
    _emit_json({"status": "not_implemented", "reason": "v0.2 roadmap"})


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


@main.group()
def benchmark() -> None:
    """Benchmark exact environments and rollout policies."""


@benchmark.command("smoke")
def benchmark_smoke() -> None:
    matrix = dense_random_matrix(rows=4, cols=4, p=101, seed=0)
    result = rref_leftmost(matrix, 101)
    _emit_json({"matrix_count": 1, "trace_length": len(result.ops), "rank": len(result.pivots)})


@main.group()
def report() -> None:
    """Report generation commands."""


@report.command("status")
def report_status() -> None:
    _emit_json({"status": "not_implemented", "reason": "v0.7 roadmap"})

