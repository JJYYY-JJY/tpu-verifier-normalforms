from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import yaml  # type: ignore[import-untyped]
from grain import MapDataset  # type: ignore[import-untyped]

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
TrainingExample: TypeAlias = dict[str, ShardValue]
MatrixFamily: TypeAlias = Literal["dense", "sparse", "low_rank"]

_OP_TO_CODE: Mapping[str, RowOpKindCode] = {"swap": 1, "scale": 2, "add": 3}
_CODE_TO_OP: Mapping[int, str] = {1: "swap", 2: "scale", 3: "add"}
_REQUIRED_SHARD_ARRAYS: Mapping[str, np.dtype[Any]] = {
    "inputs": np.dtype(np.int64),
    "finals": np.dtype(np.int64),
    "pivot_rows": np.dtype(np.int64),
    "pivot_cols": np.dtype(np.int64),
    "pivot_mask": np.dtype(np.bool_),
    "op_kind": np.dtype(np.int8),
    "op_target": np.dtype(np.int64),
    "op_source": np.dtype(np.int64),
    "op_scalar": np.dtype(np.int64),
    "op_mask": np.dtype(np.bool_),
}


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


def _metadata_from_array(value: ShardValue) -> dict[str, Any]:
    if value.shape != ():
        raise ValueError("metadata_json must be a scalar JSON string")
    raw = value.item()
    if not isinstance(raw, str):
        raw = str(raw)
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("metadata_json must decode to a JSON object")
    return loaded


def _require_shape(array: ShardValue, name: str, expected: tuple[int, ...]) -> None:
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")


def _validate_pivot_arrays(arrays: Mapping[str, ShardValue], rows: int, cols: int) -> None:
    pivot_mask = arrays["pivot_mask"]
    pivot_rows = arrays["pivot_rows"]
    pivot_cols = arrays["pivot_cols"]

    inactive = np.logical_not(pivot_mask)
    if not np.all(pivot_rows[inactive] == PADDING_VALUE):
        raise ValueError("pivot_rows padding must be -1 where pivot_mask is false")
    if not np.all(pivot_cols[inactive] == PADDING_VALUE):
        raise ValueError("pivot_cols padding must be -1 where pivot_mask is false")

    if np.any(pivot_rows[pivot_mask] < 0) or np.any(pivot_rows[pivot_mask] >= rows):
        raise ValueError("active pivot_rows entries must be valid row indices")
    if np.any(pivot_cols[pivot_mask] < 0) or np.any(pivot_cols[pivot_mask] >= cols):
        raise ValueError("active pivot_cols entries must be valid column indices")


def _validate_op_arrays(arrays: Mapping[str, ShardValue], rows: int) -> None:
    op_mask = arrays["op_mask"]
    op_kind = arrays["op_kind"]
    op_target = arrays["op_target"]
    op_source = arrays["op_source"]
    op_scalar = arrays["op_scalar"]

    inactive = np.logical_not(op_mask)
    if not np.all(op_kind[inactive] == 0):
        raise ValueError("op_kind padding must be 0 where op_mask is false")
    for name, array in (
        ("op_target", op_target),
        ("op_source", op_source),
        ("op_scalar", op_scalar),
    ):
        if not np.all(array[inactive] == PADDING_VALUE):
            raise ValueError(f"{name} padding must be -1 where op_mask is false")

    active_kinds = op_kind[op_mask]
    if np.any((active_kinds < 1) | (active_kinds > 3)):
        raise ValueError("active op_kind entries must be one of 1, 2, or 3")
    if np.any(op_target[op_mask] < 0) or np.any(op_target[op_mask] >= rows):
        raise ValueError("active op_target entries must be valid row indices")

    source_mask = op_mask & np.isin(op_kind, [1, 3])
    if np.any(op_source[source_mask] < 0) or np.any(op_source[source_mask] >= rows):
        raise ValueError("active swap/add op_source entries must be valid row indices")
    if not np.all(op_source[op_mask & (op_kind == 2)] == PADDING_VALUE):
        raise ValueError("scale op_source entries must be -1")

    if not np.all(op_scalar[op_mask & (op_kind == 1)] == PADDING_VALUE):
        raise ValueError("swap op_scalar entries must be -1")


def _load_validated_rref_shard(path: str | Path) -> tuple[ShardArrays, dict[str, Any]]:
    shard_path = Path(path)
    if shard_path.suffix != ".npz":
        raise ValueError("data path must end with .npz")
    if not shard_path.exists():
        raise ValueError(f"data path does not exist: {shard_path}")

    with np.load(shard_path, allow_pickle=False) as shard:
        missing = sorted(
            key
            for key in [*_REQUIRED_SHARD_ARRAYS.keys(), "metadata_json"]
            if key not in shard.files
        )
        if missing:
            raise ValueError(f"missing required array(s): {', '.join(missing)}")
        arrays = {key: np.asarray(shard[key]) for key in _REQUIRED_SHARD_ARRAYS}
        metadata_json = np.asarray(shard["metadata_json"])

    metadata = _metadata_from_array(metadata_json)
    schema_version = metadata.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported RREF shard schema_version: {schema_version!r}")

    shape = _require_mapping(metadata.get("shape"), "metadata.shape")
    rows = _require_positive_int(shape.get("rows"), "metadata.shape.rows")
    cols = _require_positive_int(shape.get("cols"), "metadata.shape.cols")
    count = _require_positive_int(metadata.get("count"), "metadata.count")
    max_pivots = _require_positive_int(metadata.get("max_pivots"), "metadata.max_pivots")
    max_ops = _require_positive_int(metadata.get("max_ops"), "metadata.max_ops")
    if metadata.get("padding_value") != PADDING_VALUE:
        raise ValueError("metadata.padding_value must be -1")

    config = _require_mapping(metadata.get("config"), "metadata.config")
    field = _require_mapping(config.get("field"), "metadata.config.field")
    modulus = _require_int(field.get("modulus"), "metadata.config.field.modulus")
    require_prime(modulus)

    for key, expected_dtype in _REQUIRED_SHARD_ARRAYS.items():
        if arrays[key].dtype != expected_dtype:
            raise ValueError(f"{key} must have dtype {expected_dtype}, got {arrays[key].dtype}")

    _require_shape(arrays["inputs"], "inputs", (count, rows, cols))
    _require_shape(arrays["finals"], "finals", (count, rows, cols))
    for key in ("pivot_rows", "pivot_cols", "pivot_mask"):
        _require_shape(arrays[key], key, (count, max_pivots))
    for key in ("op_kind", "op_target", "op_source", "op_scalar", "op_mask"):
        _require_shape(arrays[key], key, (count, max_ops))

    _validate_pivot_arrays(arrays, rows, cols)
    _validate_op_arrays(arrays, rows)
    return arrays, metadata


class RREFShardSamples:
    """Random-access training examples backed by a validated RREF trajectory shard."""

    def __init__(self, path: str | Path) -> None:
        self._arrays, self._metadata = _load_validated_rref_shard(path)
        shape = _require_mapping(self._metadata["shape"], "metadata.shape")
        config = _require_mapping(self._metadata["config"], "metadata.config")
        field = _require_mapping(config["field"], "metadata.config.field")
        self.rows = _require_positive_int(shape["rows"], "metadata.shape.rows")
        self.cols = _require_positive_int(shape["cols"], "metadata.shape.cols")
        self.max_pivots = _require_positive_int(
            self._metadata["max_pivots"],
            "metadata.max_pivots",
        )
        self.max_ops = _require_positive_int(self._metadata["max_ops"], "metadata.max_ops")
        self.modulus = _require_int(field["modulus"], "metadata.config.field.modulus")

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def __len__(self) -> int:
        return int(self._arrays["inputs"].shape[0])

    def __getitem__(self, index: int) -> TrainingExample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        inputs = (self._arrays["inputs"][index] % self.modulus).astype(np.float32)
        inputs /= float(self.modulus - 1)

        pivot_mask = self._arrays["pivot_mask"][index].astype(np.bool_)
        op_mask = self._arrays["op_mask"][index].astype(np.bool_)
        op_kind = self._arrays["op_kind"][index].astype(np.int32)
        op_source_mask = op_mask & np.isin(op_kind, [1, 3])
        op_scalar_mask = op_mask & np.isin(op_kind, [2, 3])

        return {
            "inputs": inputs,
            "pivot_active": pivot_mask.astype(np.float32),
            "pivot_cols": np.where(pivot_mask, self._arrays["pivot_cols"][index], 0).astype(
                np.int32
            ),
            "pivot_mask": pivot_mask,
            "op_kind": np.where(op_mask, op_kind, 0).astype(np.int32),
            "op_target": np.where(op_mask, self._arrays["op_target"][index], 0).astype(
                np.int32
            ),
            "op_source": np.where(op_source_mask, self._arrays["op_source"][index], 0).astype(
                np.int32
            ),
            "op_scalar": np.where(
                op_scalar_mask,
                self._arrays["op_scalar"][index] % self.modulus,
                0,
            ).astype(np.int32),
            "op_mask": op_mask,
            "op_source_mask": op_source_mask,
            "op_scalar_mask": op_scalar_mask,
        }


def make_rref_grain_dataset(
    path: str | Path,
    batch_size: int,
    seed: int,
    *,
    drop_remainder: bool = False,
) -> MapDataset:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    samples = RREFShardSamples(path)
    return MapDataset.source(samples).shuffle(seed).batch(
        batch_size,
        drop_remainder=drop_remainder,
    )


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
