from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from nf_agent.data.matrix_families import (
    dense_random_matrix,
    low_rank_random_matrix,
    sparse_random_matrix,
)
from nf_agent.data.shard_storage import load_shard_arrays, shard_format, write_shard_arrays
from nf_agent.env.elementary_ops import Matrix, inv_mod, normalize_matrix, require_prime
from nf_agent.env.rref_modp import PivotAction, RowOp, is_rref_modp, replay_row_ops
from nf_agent.teachers.leftmost import LeftmostRREFTeacher

SCHEMA_VERSION = "rref-backward-trace-npz-v1"
PADDING_VALUE = -1

MatrixFamily: TypeAlias = Literal["dense", "sparse", "low_rank"]
ShardValue: TypeAlias = np.ndarray[Any, np.dtype[Any]]
ShardArrays: TypeAlias = dict[str, ShardValue]
RowOpKindCode: TypeAlias = Literal[0, 1, 2, 3]

_OP_TO_CODE: Mapping[str, RowOpKindCode] = {"swap": 1, "scale": 2, "add": 3}
_CODE_TO_OP: Mapping[int, str] = {1: "swap", 2: "scale", 3: "add"}
_REQUIRED_ARRAYS: Mapping[str, np.dtype[Any]] = {
    "inputs": np.dtype(np.int64),
    "finals": np.dtype(np.int64),
    "pivots": np.dtype(np.int64),
    "ops": np.dtype(np.int64),
    "op_mask": np.dtype(np.bool_),
}


@dataclass(frozen=True)
class RREFBackwardShardConfig:
    task: Literal["rref_backward_state_shards"]
    modulus: int
    family: MatrixFamily
    rows: int
    cols: int
    max_backward_ops: int
    density: float | None = None
    rank: int | None = None

    @property
    def max_pivots(self) -> int:
        return min(self.rows, self.cols)

    def as_metadata_config(self, storage_format: str = "npz") -> dict[str, Any]:
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
            "backward_trace": {
                "schema": SCHEMA_VERSION,
                "format": storage_format,
                "max_backward_ops": self.max_backward_ops,
                "require_exact_replay": True,
            },
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


def load_rref_backward_shard_config(config_path: str | Path) -> RREFBackwardShardConfig:
    raw = _load_yaml_mapping(Path(config_path))

    task = raw.get("task")
    if task != "rref_backward_state_shards":
        raise ValueError(f"unsupported task for RREF backward shard: {task!r}")

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

    backward_trace = _require_mapping(raw.get("backward_trace"), "backward_trace")
    schema = backward_trace.get("schema")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported backward_trace.schema: {schema!r}")
    output_format = backward_trace.get("format", "npz")
    if output_format not in {"npz", "zarr"}:
        raise ValueError("backward_trace.format must be 'npz' or 'zarr'")
    max_backward_ops = _require_positive_int(
        backward_trace.get("max_backward_ops"),
        "backward_trace.max_backward_ops",
    )
    if backward_trace.get("require_exact_replay") is False:
        raise ValueError("backward_trace.require_exact_replay must not be false")

    return RREFBackwardShardConfig(
        task="rref_backward_state_shards",
        modulus=modulus,
        family=cast(MatrixFamily, family),
        rows=rows,
        cols=cols,
        max_backward_ops=max_backward_ops,
        density=density,
        rank=rank,
    )


def _generate_base_matrix(config: RREFBackwardShardConfig, seed: int) -> Matrix:
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


def _random_backward_op(rows: int, p: int, rng: Random) -> RowOp:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if rows == 1:
        return RowOp.scale(0, rng.randrange(1, p))

    kind = rng.choice(("swap", "scale", "add"))
    target = rng.randrange(rows)
    if kind == "scale":
        return RowOp.scale(target, rng.randrange(1, p))

    source = rng.randrange(rows - 1)
    if source >= target:
        source += 1
    if kind == "swap":
        return RowOp.swap(target, source)
    return RowOp.add(target, source, rng.randrange(1, p))


def _inverse_row_op(op: RowOp, p: int) -> RowOp:
    if op.kind == "swap":
        if op.source is None:
            raise ValueError("swap op requires source row")
        return RowOp.swap(op.target, op.source)
    if op.kind == "scale":
        if op.scalar is None:
            raise ValueError("scale op requires scalar")
        return RowOp.scale(op.target, inv_mod(op.scalar, p))
    if op.kind == "add":
        if op.source is None or op.scalar is None:
            raise ValueError("add op requires source row and scalar")
        return RowOp.add(op.target, op.source, -op.scalar)
    raise ValueError(f"unknown row operation kind: {op.kind}")


def _encode_row_op(op: RowOp) -> tuple[int, int, int, int]:
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


def _metadata_json(
    config: RREFBackwardShardConfig,
    count: int,
    seed_start: int,
    storage_format: str = "npz",
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": config.as_metadata_config(storage_format),
        "count": count,
        "seed_start": seed_start,
        "seed_stop_exclusive": seed_start + count,
        "shape": {"rows": config.rows, "cols": config.cols},
        "max_pivots": config.max_pivots,
        "max_ops": config.max_backward_ops,
        "op_encoding": {"pad": 0, "swap": 1, "scale": 2, "add": 3},
        "op_columns": ["kind", "target", "source", "scalar"],
        "padding_value": PADDING_VALUE,
        "generation": "canonical-rref-plus-random-invertible-row-ops",
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _encode_pivots(
    pivots: Sequence[PivotAction],
    max_pivots: int,
) -> np.ndarray[Any, np.dtype[Any]]:
    encoded = np.full((max_pivots, 2), PADDING_VALUE, dtype=np.int64)
    for pivot_index, pivot in enumerate(pivots):
        encoded[pivot_index, 0] = pivot.row
        encoded[pivot_index, 1] = pivot.col
    return encoded


def generate_rref_backward_shard(
    config_path: str | Path,
    count: int,
    seed_start: int,
    storage_format: str = "npz",
) -> ShardArrays:
    if count <= 0:
        raise ValueError("count must be positive")
    seed_start = _require_int(seed_start, "seed_start")

    config = load_rref_backward_shard_config(config_path)
    teacher = LeftmostRREFTeacher(p=config.modulus)

    inputs = np.empty((count, config.rows, config.cols), dtype=np.int64)
    finals = np.empty((count, config.rows, config.cols), dtype=np.int64)
    pivots = np.full((count, config.max_pivots, 2), PADDING_VALUE, dtype=np.int64)
    ops = np.full((count, config.max_backward_ops, 4), PADDING_VALUE, dtype=np.int64)
    ops[:, :, 0] = 0
    op_mask = np.zeros((count, config.max_backward_ops), dtype=np.bool_)

    for sample_index, seed in enumerate(range(seed_start, seed_start + count)):
        base = _generate_base_matrix(config, seed)
        result = teacher.solve(base)
        final_matrix = result.final_matrix
        current = final_matrix
        rng = Random((seed + 1) * 1_000_003 + config.rows * 97 + config.cols * 193)
        backward_ops: list[RowOp] = []

        for _ in range(config.max_backward_ops):
            op = _random_backward_op(config.rows, config.modulus, rng)
            current = replay_row_ops(current, [op], config.modulus)
            backward_ops.append(op)

        forward_ops = [_inverse_row_op(op, config.modulus) for op in reversed(backward_ops)]

        inputs[sample_index] = np.asarray(current, dtype=np.int64)
        finals[sample_index] = np.asarray(final_matrix, dtype=np.int64)
        pivots[sample_index] = _encode_pivots(result.pivots, config.max_pivots)

        for op_index, op in enumerate(forward_ops):
            ops[sample_index, op_index] = np.asarray(_encode_row_op(op), dtype=np.int64)
            op_mask[sample_index, op_index] = True

    return {
        "inputs": inputs,
        "finals": finals,
        "pivots": pivots,
        "ops": ops,
        "op_mask": op_mask,
        "metadata_json": np.asarray(_metadata_json(config, count, seed_start, storage_format)),
    }


def write_rref_backward_shard(
    config_path: str | Path,
    count: int,
    seed_start: int,
    out_path: str | Path,
) -> None:
    if count <= 0:
        raise ValueError("count must be positive")
    path = Path(out_path)
    storage_format = shard_format(path)

    shard = generate_rref_backward_shard(
        config_path=config_path,
        count=count,
        seed_start=seed_start,
        storage_format=storage_format,
    )
    write_shard_arrays(path, shard)


def _metadata_from_array(value: ShardValue) -> dict[str, Any]:
    if value.shape != ():
        raise ValueError("metadata_json must be a scalar JSON string")
    raw = value.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raw = str(raw)
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("metadata_json must decode to a JSON object")
    return loaded


def _require_shape(array: ShardValue, name: str, expected: tuple[int, ...]) -> None:
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")


def _require_prefix_mask(mask: ShardValue, name: str) -> None:
    for sample_index, row in enumerate(mask):
        inactive_seen = False
        for value in row:
            active = bool(value)
            if inactive_seen and active:
                raise ValueError(f"{name} must be true-prefix padded at sample {sample_index}")
            if not active:
                inactive_seen = True


def _pivots_from_rref(matrix: Sequence[Sequence[int]], p: int) -> list[tuple[int, int]]:
    normalized = normalize_matrix(matrix, p)
    pivots: list[tuple[int, int]] = []
    for row_index, row in enumerate(normalized):
        for col_index, value in enumerate(row):
            if value != 0:
                pivots.append((row_index, col_index))
                break
    return pivots


def _active_pivots(pivots: ShardValue, sample_index: int) -> list[tuple[int, int]]:
    encoded = np.asarray(pivots[sample_index])
    active: list[tuple[int, int]] = []
    inactive_seen = False
    for row, col in encoded:
        row_int = int(row)
        col_int = int(col)
        if row_int == PADDING_VALUE and col_int == PADDING_VALUE:
            inactive_seen = True
            continue
        if inactive_seen:
            raise ValueError(f"pivots must be true-prefix padded at sample {sample_index}")
        if row_int == PADDING_VALUE or col_int == PADDING_VALUE:
            raise ValueError(f"pivots padding must use (-1, -1) at sample {sample_index}")
        active.append((row_int, col_int))
    return active


def _validate_pivots(arrays: Mapping[str, ShardValue], rows: int, cols: int, p: int) -> None:
    pivots = arrays["pivots"]
    if np.any((pivots[:, :, 0] == PADDING_VALUE) != (pivots[:, :, 1] == PADDING_VALUE)):
        raise ValueError("pivots padding must use (-1, -1)")
    active = pivots[:, :, 0] != PADDING_VALUE
    if np.any(pivots[:, :, 0][active] < 0) or np.any(pivots[:, :, 0][active] >= rows):
        raise ValueError("active pivot rows must be valid row indices")
    if np.any(pivots[:, :, 1][active] < 0) or np.any(pivots[:, :, 1][active] >= cols):
        raise ValueError("active pivot cols must be valid column indices")

    for sample_index, final in enumerate(arrays["finals"]):
        claimed = _active_pivots(pivots, sample_index)
        derived = _pivots_from_rref(final.tolist(), p)
        if claimed != derived:
            raise ValueError(f"sample {sample_index} pivots do not match final matrix")


def _validate_ops(arrays: Mapping[str, ShardValue], rows: int, p: int) -> None:
    ops = arrays["ops"]
    op_mask = arrays["op_mask"]
    _require_prefix_mask(op_mask, "op_mask")

    inactive = np.logical_not(op_mask)
    inactive_ops = ops[inactive]
    if inactive_ops.size and (
        not np.all(inactive_ops[:, 0] == 0) or not np.all(inactive_ops[:, 1:] == PADDING_VALUE)
    ):
        raise ValueError("ops padding must be [0, -1, -1, -1] where op_mask is false")

    active_ops = ops[op_mask]
    if active_ops.size == 0:
        return

    kinds = active_ops[:, 0]
    targets = active_ops[:, 1]
    sources = active_ops[:, 2]
    scalars = active_ops[:, 3]
    if np.any((kinds < 1) | (kinds > 3)):
        raise ValueError("active op kinds must be one of 1, 2, or 3")
    if np.any(targets < 0) or np.any(targets >= rows):
        raise ValueError("active op targets must be valid row indices")

    source_mask = np.isin(kinds, [1, 3])
    if np.any(sources[source_mask] < 0) or np.any(sources[source_mask] >= rows):
        raise ValueError("active swap/add sources must be valid row indices")
    if np.any(targets[source_mask] == sources[source_mask]):
        raise ValueError("active swap/add target and source rows must be distinct")
    if not np.all(sources[kinds == 2] == PADDING_VALUE):
        raise ValueError("scale op source entries must be -1")

    if not np.all(scalars[kinds == 1] == PADDING_VALUE):
        raise ValueError("swap op scalar entries must be -1")
    scalar_mask = np.isin(kinds, [2, 3])
    if np.any(scalars[scalar_mask] % p == 0):
        raise ValueError("active scale/add scalars must be nonzero modulo p")


def row_ops_from_backward_shard_arrays(shard: Mapping[str, Any], sample_index: int) -> list[RowOp]:
    op_mask = np.asarray(shard["op_mask"][sample_index])
    encoded_ops = np.asarray(shard["ops"][sample_index])

    ops: list[RowOp] = []
    for op_index, active in enumerate(op_mask):
        if not bool(active):
            continue
        kind, target, source, scalar = encoded_ops[op_index]
        ops.append(_decode_row_op(int(kind), int(target), int(source), int(scalar)))
    return ops


def _validate_replay(arrays: Mapping[str, ShardValue], p: int) -> None:
    for sample_index in range(int(arrays["inputs"].shape[0])):
        input_matrix = arrays["inputs"][sample_index].tolist()
        final_matrix = arrays["finals"][sample_index].tolist()
        ops = row_ops_from_backward_shard_arrays(arrays, sample_index)
        replayed = replay_row_ops(input_matrix, ops, p)
        if replayed != final_matrix:
            raise ValueError(f"sample {sample_index} does not replay to claimed final")
        if not is_rref_modp(final_matrix, p):
            raise ValueError(f"sample {sample_index} final matrix is not RREF")


def load_rref_backward_shard(path: str | Path) -> tuple[ShardArrays, dict[str, Any]]:
    shard_path = Path(path)
    arrays, metadata_json = load_shard_arrays(shard_path, _REQUIRED_ARRAYS)

    metadata = _metadata_from_array(metadata_json)
    schema_version = metadata.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported RREF backward shard schema_version: {schema_version!r}")

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

    for key, expected_dtype in _REQUIRED_ARRAYS.items():
        if arrays[key].dtype != expected_dtype:
            raise ValueError(f"{key} must have dtype {expected_dtype}, got {arrays[key].dtype}")

    _require_shape(arrays["inputs"], "inputs", (count, rows, cols))
    _require_shape(arrays["finals"], "finals", (count, rows, cols))
    _require_shape(arrays["pivots"], "pivots", (count, max_pivots, 2))
    _require_shape(arrays["ops"], "ops", (count, max_ops, 4))
    _require_shape(arrays["op_mask"], "op_mask", (count, max_ops))

    if np.any(arrays["inputs"] < 0) or np.any(arrays["inputs"] >= modulus):
        raise ValueError("inputs entries must be normalized modulo p")
    if np.any(arrays["finals"] < 0) or np.any(arrays["finals"] >= modulus):
        raise ValueError("finals entries must be normalized modulo p")

    _validate_ops(arrays, rows, modulus)
    _validate_replay(arrays, modulus)
    _validate_pivots(arrays, rows, cols, modulus)
    return arrays, metadata
