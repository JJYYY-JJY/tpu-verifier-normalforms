from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import yaml  # type: ignore[import-untyped]

from nf_agent.data.matrix_families import (
    dense_random_matrix,
    low_rank_random_matrix,
    sparse_random_matrix,
)
from nf_agent.env.elementary_ops import Matrix, require_prime
from nf_agent.env.rref_modp import RowOp
from nf_agent.teachers.leftmost import LeftmostRREFTeacher

SCHEMA_VERSION = "rref-teacher-trajectory-npz-v0.2"
PADDING_VALUE = -1

RowOpKindCode: TypeAlias = Literal[0, 1, 2, 3]
ShardValue: TypeAlias = np.ndarray[Any, np.dtype[Any]]
ShardArrays: TypeAlias = dict[str, ShardValue]
MatrixFamily: TypeAlias = Literal["dense", "sparse", "low_rank"]

_OP_TO_CODE: Mapping[str, RowOpKindCode] = {"swap": 1, "scale": 2, "add": 3}
_CODE_TO_OP: Mapping[int, str] = {1: "swap", 2: "scale", 3: "add"}


@dataclass(frozen=True)
class RREFShardConfig:
    task: Literal["rref"]
    modulus: int
    family: MatrixFamily
    rows: int
    cols: int
    teacher: Literal["leftmost"]
    density: float | None = None
    rank: int | None = None

    @property
    def max_pivots(self) -> int:
        return min(self.rows, self.cols)

    @property
    def max_ops(self) -> int:
        return self.max_pivots * (self.rows + 1)

    def as_metadata_config(self) -> dict[str, Any]:
        matrix: dict[str, Any] = {
            "family": self.family,
            "rows": self.rows,
            "cols": self.cols,
        }
        if self.density is not None:
            matrix["density"] = self.density
        if self.rank is not None:
            matrix["rank"] = self.rank
        return {
            "task": self.task,
            "field": {"modulus": self.modulus},
            "matrix": matrix,
            "teacher": self.teacher,
        }


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_positive_int(value: object, name: str) -> int:
    integer = _require_int(value, name)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _require_float(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _load_yaml_mapping(config_path: Path) -> Mapping[str, Any]:
    with config_path.open() as handle:
        loaded = yaml.safe_load(handle)
    return _require_mapping(loaded, "config")


def load_rref_shard_config(config_path: str | Path) -> RREFShardConfig:
    path = Path(config_path)
    raw = _load_yaml_mapping(path)

    task = raw.get("task")
    if task != "rref":
        raise ValueError(f"unsupported task for RREF shard: {task!r}")

    teacher = raw.get("teacher")
    if teacher != "leftmost":
        raise ValueError(f"unsupported teacher for RREF shard: {teacher!r}")

    field = _require_mapping(raw.get("field"), "field")
    modulus = _require_int(field.get("modulus"), "field.modulus")
    require_prime(modulus)

    matrix = _require_mapping(raw.get("matrix"), "matrix")
    family = matrix.get("family")
    if family is None:
        raise ValueError("matrix.family is required")
    if family not in {"dense", "sparse", "low_rank"}:
        raise ValueError(f"unsupported matrix family: {family!r}")

    rows = _require_positive_int(matrix.get("rows"), "matrix.rows")
    cols = _require_positive_int(matrix.get("cols"), "matrix.cols")

    density: float | None = None
    rank: int | None = None
    if family == "sparse":
        density = _require_float(matrix.get("density"), "matrix.density")
        if not 0.0 <= density <= 1.0:
            raise ValueError("matrix.density must lie in [0, 1]")
    if family == "low_rank":
        rank = _require_int(matrix.get("rank"), "matrix.rank")
        if rank < 0:
            raise ValueError("matrix.rank must be nonnegative")

    return RREFShardConfig(
        task="rref",
        modulus=modulus,
        family=family,
        rows=rows,
        cols=cols,
        teacher="leftmost",
        density=density,
        rank=rank,
    )


def _generate_matrix(config: RREFShardConfig, seed: int) -> Matrix:
    if config.family == "dense":
        return dense_random_matrix(config.rows, config.cols, config.modulus, seed)
    if config.family == "sparse":
        if config.density is None:
            raise ValueError("matrix.density is required for sparse matrices")
        return sparse_random_matrix(
            config.rows,
            config.cols,
            config.modulus,
            config.density,
            seed,
        )
    if config.rank is None:
        raise ValueError("matrix.rank is required for low_rank matrices")
    return low_rank_random_matrix(
        config.rows,
        config.cols,
        config.rank,
        config.modulus,
        seed,
    )


def _encode_row_op(op: RowOp) -> tuple[RowOpKindCode, int, int, int]:
    kind = _OP_TO_CODE[op.kind]
    source = PADDING_VALUE if op.source is None else op.source
    scalar = PADDING_VALUE if op.scalar is None else op.scalar
    return kind, op.target, source, scalar


def _decode_row_op(kind: int, target: int, source: int, scalar: int) -> RowOp:
    op_name = _CODE_TO_OP.get(kind)
    if op_name == "swap":
        return RowOp.swap(target, source)
    if op_name == "scale":
        return RowOp.scale(target, scalar)
    if op_name == "add":
        return RowOp.add(target, source, scalar)
    raise ValueError(f"unknown encoded row operation kind: {kind}")


def _metadata_json(config: RREFShardConfig, count: int, seed_start: int) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": config.as_metadata_config(),
        "count": count,
        "seed_start": seed_start,
        "seed_stop_exclusive": seed_start + count,
        "shape": {"rows": config.rows, "cols": config.cols},
        "max_pivots": config.max_pivots,
        "max_ops": config.max_ops,
        "op_encoding": {"pad": 0, "swap": 1, "scale": 2, "add": 3},
        "padding_value": PADDING_VALUE,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def generate_rref_shard(config_path: str | Path, count: int, seed_start: int) -> ShardArrays:
    if count <= 0:
        raise ValueError("count must be positive")
    seed_start = _require_int(seed_start, "seed_start")

    config = load_rref_shard_config(config_path)
    teacher = LeftmostRREFTeacher(p=config.modulus)

    inputs = np.empty((count, config.rows, config.cols), dtype=np.int64)
    finals = np.empty((count, config.rows, config.cols), dtype=np.int64)
    pivot_rows = np.full((count, config.max_pivots), PADDING_VALUE, dtype=np.int64)
    pivot_cols = np.full((count, config.max_pivots), PADDING_VALUE, dtype=np.int64)
    pivot_mask = np.zeros((count, config.max_pivots), dtype=np.bool_)
    op_kind = np.zeros((count, config.max_ops), dtype=np.int8)
    op_target = np.full((count, config.max_ops), PADDING_VALUE, dtype=np.int64)
    op_source = np.full((count, config.max_ops), PADDING_VALUE, dtype=np.int64)
    op_scalar = np.full((count, config.max_ops), PADDING_VALUE, dtype=np.int64)
    op_mask = np.zeros((count, config.max_ops), dtype=np.bool_)

    for sample_index, seed in enumerate(range(seed_start, seed_start + count)):
        matrix = _generate_matrix(config, seed)
        result = teacher.solve(matrix)
        if len(result.ops) > config.max_ops:
            raise ValueError(
                f"teacher emitted {len(result.ops)} row ops, exceeding max_ops={config.max_ops}"
            )

        inputs[sample_index] = np.asarray(matrix, dtype=np.int64)
        finals[sample_index] = np.asarray(result.final_matrix, dtype=np.int64)

        for pivot_index, pivot in enumerate(result.pivots):
            pivot_rows[sample_index, pivot_index] = pivot.row
            pivot_cols[sample_index, pivot_index] = pivot.col
            pivot_mask[sample_index, pivot_index] = True

        for op_index, op in enumerate(result.ops):
            kind, target, source, scalar = _encode_row_op(op)
            op_kind[sample_index, op_index] = kind
            op_target[sample_index, op_index] = target
            op_source[sample_index, op_index] = source
            op_scalar[sample_index, op_index] = scalar
            op_mask[sample_index, op_index] = True

    return {
        "inputs": inputs,
        "finals": finals,
        "pivot_rows": pivot_rows,
        "pivot_cols": pivot_cols,
        "pivot_mask": pivot_mask,
        "op_kind": op_kind,
        "op_target": op_target,
        "op_source": op_source,
        "op_scalar": op_scalar,
        "op_mask": op_mask,
        "metadata_json": np.asarray(_metadata_json(config, count, seed_start)),
    }


def write_rref_shard(
    config_path: str | Path,
    count: int,
    seed_start: int,
    out_path: str | Path,
) -> None:
    path = Path(out_path)
    if path.suffix != ".npz":
        raise ValueError("output path must end with .npz")

    shard = generate_rref_shard(config_path=config_path, count=count, seed_start=seed_start)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **shard)  # type: ignore[arg-type]


def row_ops_from_shard_arrays(shard: Mapping[str, Any], sample_index: int) -> list[RowOp]:
    op_mask = np.asarray(shard["op_mask"][sample_index])
    op_kind = np.asarray(shard["op_kind"][sample_index])
    op_target = np.asarray(shard["op_target"][sample_index])
    op_source = np.asarray(shard["op_source"][sample_index])
    op_scalar = np.asarray(shard["op_scalar"][sample_index])

    ops: list[RowOp] = []
    for op_index, active in enumerate(op_mask):
        if not bool(active):
            continue
        ops.append(
            _decode_row_op(
                int(op_kind[op_index]),
                int(op_target[op_index]),
                int(op_source[op_index]),
                int(op_scalar[op_index]),
            )
        )
    return ops
