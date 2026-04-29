from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import yaml  # type: ignore[import-untyped]
from grain import MapDataset  # type: ignore[import-untyped]

from nf_agent.data.matrix_families import sparse_integer_matrix
from nf_agent.data.shard_storage import load_shard_arrays, shard_format, write_shard_arrays
from nf_agent.env.elementary_ops import Matrix
from nf_agent.env.hnf_int import (
    IntegerRowOp,
    is_row_hnf,
    normalize_integer_matrix,
    replay_integer_row_ops,
    row_hnf,
)

SCHEMA_VERSION = "hnf-teacher-trajectory-npz-v0.8"
BACKWARD_SCHEMA_VERSION = "hnf-backward-trace-zarr-v1"
PADDING_VALUE = -1

IntegerOpKindCode: TypeAlias = Literal[0, 1, 2, 3]
ShardValue: TypeAlias = np.ndarray[Any, np.dtype[Any]]
ShardArrays: TypeAlias = dict[str, ShardValue]
TrainingExample: TypeAlias = dict[str, ShardValue]

_OP_TO_CODE: Mapping[str, IntegerOpKindCode] = {"swap": 1, "negate": 2, "add": 3}
_CODE_TO_OP: Mapping[int, str] = {1: "swap", 2: "negate", 3: "add"}
_REQUIRED_SHARD_ARRAYS: Mapping[str, np.dtype[Any]] = {
    "inputs": np.dtype(np.int64),
    "finals": np.dtype(np.int64),
    "op_kind": np.dtype(np.int8),
    "op_target": np.dtype(np.int64),
    "op_source": np.dtype(np.int64),
    "op_scalar_id": np.dtype(np.int64),
    "op_scalar_value": np.dtype(np.int64),
    "op_mask": np.dtype(np.bool_),
    "scalar_vocab": np.dtype(np.int64),
}


@dataclass(frozen=True)
class HNFShardConfig:
    task: Literal["hnf"]
    family: Literal["sparse"]
    rows: int
    cols: int
    density: float
    entry_bound: int = 9

    def as_metadata_config(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "integer_matrix": {
                "family": self.family,
                "rows": self.rows,
                "cols": self.cols,
                "density": self.density,
                "entry_bound": self.entry_bound,
            },
            "teacher": "row_hnf",
        }


@dataclass(frozen=True)
class HNFTrajectory:
    input_matrix: Matrix
    final_matrix: Matrix
    ops: tuple[IntegerRowOp, ...]
    seed: int | None = None


@dataclass(frozen=True)
class HNFBackwardFamilyConfig:
    name: str
    rows: int
    cols: int
    density: float
    entry_bound: int = 9

    def as_hnf_shard_config(self) -> HNFShardConfig:
        return HNFShardConfig(
            task="hnf",
            family="sparse",
            rows=self.rows,
            cols=self.cols,
            density=self.density,
            entry_bound=self.entry_bound,
        )

    def as_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": "sparse",
            "rows": self.rows,
            "cols": self.cols,
            "density": self.density,
            "entry_bound": self.entry_bound,
        }


@dataclass(frozen=True)
class HNFBackwardShardConfig:
    task: Literal["hnf_growth_search"]
    families: tuple[HNFBackwardFamilyConfig, ...]
    storage_format: Literal["npz", "zarr"]
    require_unimodular_ops: bool = True
    search_mode: Literal["exact_row_preconditioned"] | None = None
    search_candidate_limit: int | None = None
    search_objective: tuple[str, ...] = ()
    rollout_beam_size: int | None = None

    def family_config(self, name: str) -> HNFBackwardFamilyConfig:
        for family in self.families:
            if family.name == name:
                return family
        names = ", ".join(family.name for family in self.families)
        raise ValueError(f"unknown HNF growth family {name!r}; available: {names}")

    def as_metadata_config(
        self,
        family: HNFBackwardFamilyConfig,
        storage_format: str,
    ) -> dict[str, Any]:
        return {
            "task": self.task,
            "family": family.as_metadata(),
            "backward_trace": {
                "schema": BACKWARD_SCHEMA_VERSION,
                "format": storage_format,
                "require_unimodular_ops": self.require_unimodular_ops,
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


def load_hnf_shard_config(config_path: str | Path) -> HNFShardConfig:
    raw = _load_yaml_mapping(Path(config_path))
    task = raw.get("task")
    if task != "hnf":
        raise ValueError(f"unsupported task for HNF shard: {task!r}")

    matrix = _require_mapping(raw.get("integer_matrix"), "integer_matrix")
    family = matrix.get("family")
    if family != "sparse":
        raise ValueError(f"unsupported integer matrix family: {family!r}")
    rows = _require_positive_int(matrix.get("rows"), "rows")
    cols = _require_positive_int(matrix.get("cols"), "cols")
    density = _require_float(matrix.get("density"), "density")
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must lie in [0, 1]")
    entry_bound = _require_positive_int(matrix.get("entry_bound", 9), "entry_bound")
    return HNFShardConfig(
        task="hnf",
        family="sparse",
        rows=rows,
        cols=cols,
        density=density,
        entry_bound=entry_bound,
    )


def load_hnf_backward_shard_config(config_path: str | Path) -> HNFBackwardShardConfig:
    raw = _load_yaml_mapping(Path(config_path))
    task = raw.get("task")
    if task != "hnf_growth_search":
        raise ValueError(f"unsupported task for HNF backward shard: {task!r}")

    raw_families = raw.get("integer_families")
    if not isinstance(raw_families, Sequence) or isinstance(raw_families, str | bytes):
        raise ValueError("integer_families must be a sequence")
    families: list[HNFBackwardFamilyConfig] = []
    seen_names: set[str] = set()
    for index, value in enumerate(raw_families):
        family = _require_mapping(value, f"integer_families[{index}]")
        name = family.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"integer_families[{index}].name must be a nonempty string")
        if name in seen_names:
            raise ValueError(f"duplicate HNF growth family: {name}")
        seen_names.add(name)
        rows = _require_positive_int(family.get("rows"), f"integer_families[{index}].rows")
        cols = _require_positive_int(family.get("cols"), f"integer_families[{index}].cols")
        density = _require_float(family.get("density"), f"integer_families[{index}].density")
        if not 0.0 <= density <= 1.0:
            raise ValueError(f"integer_families[{index}].density must lie in [0, 1]")
        entry_bound = _require_positive_int(
            family.get("entry_bound", 9),
            f"integer_families[{index}].entry_bound",
        )
        families.append(
            HNFBackwardFamilyConfig(
                name=name,
                rows=rows,
                cols=cols,
                density=density,
                entry_bound=entry_bound,
            )
        )
    if not families:
        raise ValueError("integer_families must contain at least one family")

    backward_trace = _require_mapping(raw.get("backward_trace"), "backward_trace")
    schema = backward_trace.get("schema")
    if schema != BACKWARD_SCHEMA_VERSION:
        raise ValueError(f"unsupported backward_trace.schema: {schema!r}")
    output_format = backward_trace.get("format", "npz")
    if output_format not in {"npz", "zarr"}:
        raise ValueError("backward_trace.format must be 'npz' or 'zarr'")
    if backward_trace.get("require_unimodular_ops") is False:
        raise ValueError("backward_trace.require_unimodular_ops must not be false")
    if backward_trace.get("require_exact_replay") is False:
        raise ValueError("backward_trace.require_exact_replay must not be false")

    search_mode: Literal["exact_row_preconditioned"] | None = None
    search_candidate_limit: int | None = None
    search_objective: tuple[str, ...] = ()
    raw_search = raw.get("search")
    if raw_search is not None:
        search = _require_mapping(raw_search, "search")
        mode = search.get("mode")
        if mode is not None:
            if mode != "exact_row_preconditioned":
                raise ValueError("search.mode must be 'exact_row_preconditioned'")
            search_mode = "exact_row_preconditioned"
        if search.get("candidate_limit") is not None:
            search_candidate_limit = _require_positive_int(
                search.get("candidate_limit"),
                "search.candidate_limit",
            )
        objective = search.get("objective", ())
        if objective is not None:
            if not isinstance(objective, Sequence) or isinstance(objective, str | bytes):
                raise ValueError("search.objective must be a sequence")
            objective_values: list[str] = []
            for index, value in enumerate(objective):
                if not isinstance(value, str) or not value:
                    raise ValueError(f"search.objective[{index}] must be a nonempty string")
                objective_values.append(value)
            search_objective = tuple(objective_values)

    rollout_beam_size: int | None = None
    raw_rollout = raw.get("rollout")
    if raw_rollout is not None:
        rollout = _require_mapping(raw_rollout, "rollout")
        if rollout.get("beam_size") is not None:
            rollout_beam_size = _require_positive_int(
                rollout.get("beam_size"),
                "rollout.beam_size",
            )

    return HNFBackwardShardConfig(
        task="hnf_growth_search",
        families=tuple(families),
        storage_format=cast(Literal["npz", "zarr"], output_format),
        require_unimodular_ops=True,
        search_mode=search_mode,
        search_candidate_limit=search_candidate_limit,
        search_objective=search_objective,
        rollout_beam_size=rollout_beam_size,
    )


def _encode_integer_row_op(
    op: IntegerRowOp,
    scalar_to_id: Mapping[int, int],
) -> tuple[IntegerOpKindCode, int, int, int, int]:
    kind = _OP_TO_CODE[op.kind]
    source = PADDING_VALUE if op.source is None else op.source
    scalar_value = PADDING_VALUE if op.scalar is None else op.scalar
    scalar_id = PADDING_VALUE if op.scalar is None else scalar_to_id[op.scalar]
    return kind, op.target, source, scalar_id, scalar_value


def _decode_integer_row_op(
    kind: int,
    target: int,
    source: int,
    scalar_value: int,
) -> IntegerRowOp:
    op_name = _CODE_TO_OP.get(kind)
    if op_name == "swap":
        return IntegerRowOp.swap(target, source)
    if op_name == "negate":
        return IntegerRowOp.negate(target)
    if op_name == "add":
        return IntegerRowOp.add(target, source, scalar_value)
    raise ValueError(f"unknown encoded integer row operation kind: {kind}")


def _metadata_json(
    config_payload: Mapping[str, Any],
    *,
    count: int,
    seed_start: int,
    rows: int,
    cols: int,
    max_ops: int,
    input_scale: int,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": dict(config_payload),
        "count": count,
        "seed_start": seed_start,
        "seed_stop_exclusive": seed_start + count,
        "shape": {"rows": rows, "cols": cols},
        "max_ops": max_ops,
        "input_scale": input_scale,
        "op_encoding": {"pad": 0, "swap": 1, "negate": 2, "add": 3},
        "padding_value": PADDING_VALUE,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _backward_metadata_json(
    config_payload: Mapping[str, Any],
    *,
    count: int,
    seed_start: int,
    rows: int,
    cols: int,
    max_ops: int,
    input_scale: int,
    storage_format: str,
) -> str:
    config = dict(config_payload)
    backward_trace = dict(_require_mapping(config.get("backward_trace"), "config.backward_trace"))
    backward_trace["schema"] = BACKWARD_SCHEMA_VERSION
    backward_trace["format"] = storage_format
    backward_trace["require_exact_replay"] = True
    backward_trace["require_unimodular_ops"] = True
    config["backward_trace"] = backward_trace
    payload = {
        "schema_version": BACKWARD_SCHEMA_VERSION,
        "config": config,
        "count": count,
        "seed_start": seed_start,
        "seed_stop_exclusive": seed_start + count,
        "shape": {"rows": rows, "cols": cols},
        "max_ops": max_ops,
        "input_scale": input_scale,
        "op_encoding": {"pad": 0, "swap": 1, "negate": 2, "add": 3},
        "padding_value": PADDING_VALUE,
        "generation": "sparse-integer-row-hnf-teacher-trace",
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _generate_trajectories(
    config: HNFShardConfig,
    count: int,
    seed_start: int,
) -> list[HNFTrajectory]:
    trajectories: list[HNFTrajectory] = []
    for seed in range(seed_start, seed_start + count):
        matrix = sparse_integer_matrix(
            rows=config.rows,
            cols=config.cols,
            density=config.density,
            seed=seed,
            entry_bound=config.entry_bound,
        )
        result = row_hnf(matrix)
        trajectories.append(
            HNFTrajectory(
                input_matrix=matrix,
                final_matrix=result.final_matrix,
                ops=tuple(result.ops),
                seed=seed,
            )
        )
    return trajectories


def hnf_shard_arrays_from_trajectories(
    trajectories: Sequence[HNFTrajectory],
    *,
    config_payload: Mapping[str, Any],
    seed_start: int = 0,
) -> ShardArrays:
    if not trajectories:
        raise ValueError("at least one HNF trajectory is required")

    normalized_inputs = [normalize_integer_matrix(item.input_matrix) for item in trajectories]
    normalized_finals = [normalize_integer_matrix(item.final_matrix) for item in trajectories]
    rows = len(normalized_inputs[0])
    cols = len(normalized_inputs[0][0]) if rows else 0
    if rows <= 0 or cols <= 0:
        raise ValueError("HNF shard matrices must have positive shape")

    for index, matrix in enumerate([*normalized_inputs, *normalized_finals]):
        if len(matrix) != rows or (matrix and len(matrix[0]) != cols):
            raise ValueError(f"trajectory matrix shape mismatch at index {index}")

    scalar_values = sorted(
        {
            int(op.scalar)
            for trajectory in trajectories
            for op in trajectory.ops
            if op.kind == "add" and op.scalar is not None
        }
    )
    scalar_vocab = np.asarray(scalar_values, dtype=np.int64)
    scalar_to_id = {scalar: index for index, scalar in enumerate(scalar_values)}
    max_ops = max(1, max(len(item.ops) for item in trajectories))
    count = len(trajectories)
    input_scale = max(
        1,
        max(abs(entry) for matrix in normalized_inputs for row in matrix for entry in row),
    )

    inputs = np.asarray(normalized_inputs, dtype=np.int64)
    finals = np.asarray(normalized_finals, dtype=np.int64)
    op_kind = np.zeros((count, max_ops), dtype=np.int8)
    op_target = np.full((count, max_ops), PADDING_VALUE, dtype=np.int64)
    op_source = np.full((count, max_ops), PADDING_VALUE, dtype=np.int64)
    op_scalar_id = np.full((count, max_ops), PADDING_VALUE, dtype=np.int64)
    op_scalar_value = np.full((count, max_ops), PADDING_VALUE, dtype=np.int64)
    op_mask = np.zeros((count, max_ops), dtype=np.bool_)

    for sample_index, trajectory in enumerate(trajectories):
        replayed = replay_integer_row_ops(trajectory.input_matrix, trajectory.ops)
        if replayed != trajectory.final_matrix:
            raise ValueError(f"HNF trajectory {sample_index} does not replay")
        if not is_row_hnf(trajectory.final_matrix):
            raise ValueError(f"HNF trajectory {sample_index} final matrix is not row HNF")
        for op_index, op in enumerate(trajectory.ops):
            kind, target, source, scalar_id, scalar_value = _encode_integer_row_op(
                op,
                scalar_to_id,
            )
            op_kind[sample_index, op_index] = kind
            op_target[sample_index, op_index] = target
            op_source[sample_index, op_index] = source
            op_scalar_id[sample_index, op_index] = scalar_id
            op_scalar_value[sample_index, op_index] = scalar_value
            op_mask[sample_index, op_index] = True

    return {
        "inputs": inputs,
        "finals": finals,
        "op_kind": op_kind,
        "op_target": op_target,
        "op_source": op_source,
        "op_scalar_id": op_scalar_id,
        "op_scalar_value": op_scalar_value,
        "op_mask": op_mask,
        "scalar_vocab": scalar_vocab,
        "metadata_json": np.asarray(
            _metadata_json(
                config_payload,
                count=count,
                seed_start=seed_start,
                rows=rows,
                cols=cols,
                max_ops=max_ops,
                input_scale=input_scale,
            )
        ),
    }


def generate_hnf_shard(config_path: str | Path, count: int, seed_start: int) -> ShardArrays:
    if count <= 0:
        raise ValueError("count must be positive")
    seed_start = _require_int(seed_start, "seed_start")
    config = load_hnf_shard_config(config_path)
    trajectories = _generate_trajectories(config, count, seed_start)
    return hnf_shard_arrays_from_trajectories(
        trajectories,
        config_payload=config.as_metadata_config(),
        seed_start=seed_start,
    )


def hnf_backward_shard_arrays_from_trajectories(
    trajectories: Sequence[HNFTrajectory],
    *,
    config_payload: Mapping[str, Any],
    seed_start: int = 0,
    storage_format: str = "npz",
) -> ShardArrays:
    if storage_format not in {"npz", "zarr"}:
        raise ValueError("storage_format must be 'npz' or 'zarr'")
    shard = hnf_shard_arrays_from_trajectories(
        trajectories,
        config_payload=config_payload,
        seed_start=seed_start,
    )
    metadata = _metadata_from_array(np.asarray(shard["metadata_json"]))
    shape = _require_mapping(metadata.get("shape"), "metadata.shape")
    shard["metadata_json"] = np.asarray(
        _backward_metadata_json(
            config_payload,
            count=_require_positive_int(metadata.get("count"), "metadata.count"),
            seed_start=seed_start,
            rows=_require_positive_int(shape.get("rows"), "metadata.shape.rows"),
            cols=_require_positive_int(shape.get("cols"), "metadata.shape.cols"),
            max_ops=_require_positive_int(metadata.get("max_ops"), "metadata.max_ops"),
            input_scale=_require_positive_int(
                metadata.get("input_scale"),
                "metadata.input_scale",
            ),
            storage_format=storage_format,
        )
    )
    return shard


def generate_hnf_backward_shard(
    config_path: str | Path,
    family: str,
    count: int,
    seed_start: int,
    storage_format: str = "npz",
) -> ShardArrays:
    if count <= 0:
        raise ValueError("count must be positive")
    seed_start = _require_int(seed_start, "seed_start")
    config = load_hnf_backward_shard_config(config_path)
    family_config = config.family_config(family)
    trajectories = _generate_trajectories(family_config.as_hnf_shard_config(), count, seed_start)
    return hnf_backward_shard_arrays_from_trajectories(
        trajectories,
        config_payload=config.as_metadata_config(family_config, storage_format),
        seed_start=seed_start,
        storage_format=storage_format,
    )


def write_hnf_shard(
    config_path: str | Path,
    count: int,
    seed_start: int,
    out_path: str | Path,
) -> None:
    path = Path(out_path)
    if path.suffix != ".npz":
        raise ValueError("output path must end with .npz")
    shard = generate_hnf_shard(config_path=config_path, count=count, seed_start=seed_start)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **shard)  # type: ignore[arg-type]


def write_hnf_shard_from_trajectories(
    trajectories: Sequence[HNFTrajectory],
    *,
    config_payload: Mapping[str, Any],
    out_path: str | Path,
    seed_start: int = 0,
) -> None:
    path = Path(out_path)
    if path.suffix != ".npz":
        raise ValueError("output path must end with .npz")
    shard = hnf_shard_arrays_from_trajectories(
        trajectories,
        config_payload=config_payload,
        seed_start=seed_start,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **shard)  # type: ignore[arg-type]


def write_hnf_backward_shard(
    config_path: str | Path,
    family: str,
    count: int,
    seed_start: int,
    out_path: str | Path,
) -> None:
    if count <= 0:
        raise ValueError("count must be positive")
    path = Path(out_path)
    storage_format = shard_format(path)
    shard = generate_hnf_backward_shard(
        config_path=config_path,
        family=family,
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


def _validate_op_arrays(
    arrays: Mapping[str, ShardValue],
    *,
    rows: int,
    scalar_vocab: ShardValue,
) -> None:
    op_mask = arrays["op_mask"]
    op_kind = arrays["op_kind"]
    op_target = arrays["op_target"]
    op_source = arrays["op_source"]
    op_scalar_id = arrays["op_scalar_id"]
    op_scalar_value = arrays["op_scalar_value"]
    _require_prefix_mask(op_mask, "op_mask")
    inactive = np.logical_not(op_mask)
    if not np.all(op_kind[inactive] == 0):
        raise ValueError("op_kind padding must be 0 where op_mask is false")
    for name, array in (
        ("op_target", op_target),
        ("op_source", op_source),
        ("op_scalar_id", op_scalar_id),
        ("op_scalar_value", op_scalar_value),
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
    if np.any(op_target[source_mask] == op_source[source_mask]):
        raise ValueError("active swap/add target and source rows must be distinct")
    if not np.all(op_source[op_mask & (op_kind == 2)] == PADDING_VALUE):
        raise ValueError("negate op_source entries must be -1")

    scalar_mask = op_mask & (op_kind == 3)
    if np.any(op_scalar_id[scalar_mask] < 0) or np.any(
        op_scalar_id[scalar_mask] >= len(scalar_vocab)
    ):
        raise ValueError("active add op_scalar_id entries must index scalar_vocab")
    if not np.all(op_scalar_id[op_mask & (op_kind != 3)] == PADDING_VALUE):
        raise ValueError("non-add op_scalar_id entries must be -1")
    if not np.all(op_scalar_value[op_mask & (op_kind != 3)] == PADDING_VALUE):
        raise ValueError("non-add op_scalar_value entries must be -1")
    if np.any(scalar_mask):
        expected_values = scalar_vocab[op_scalar_id[scalar_mask]]
        if not np.array_equal(op_scalar_value[scalar_mask], expected_values):
            raise ValueError("op_scalar_value entries must match scalar_vocab")


def _validate_hnf_replay(arrays: Mapping[str, ShardValue]) -> None:
    for sample_index in range(int(arrays["inputs"].shape[0])):
        input_matrix = arrays["inputs"][sample_index].tolist()
        final_matrix = arrays["finals"][sample_index].tolist()
        ops = integer_row_ops_from_hnf_shard_arrays(arrays, sample_index)
        replayed = replay_integer_row_ops(input_matrix, ops)
        if replayed != final_matrix:
            raise ValueError(f"sample {sample_index} does not replay to claimed final")
        if not is_row_hnf(final_matrix):
            raise ValueError(f"sample {sample_index} final matrix is not row HNF")


def _load_validated_hnf_shard(path: str | Path) -> tuple[ShardArrays, dict[str, Any]]:
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
        raise ValueError(f"unsupported HNF shard schema_version: {schema_version!r}")
    shape = _require_mapping(metadata.get("shape"), "metadata.shape")
    rows = _require_positive_int(shape.get("rows"), "metadata.shape.rows")
    cols = _require_positive_int(shape.get("cols"), "metadata.shape.cols")
    count = _require_positive_int(metadata.get("count"), "metadata.count")
    max_ops = _require_positive_int(metadata.get("max_ops"), "metadata.max_ops")
    _require_positive_int(metadata.get("input_scale"), "metadata.input_scale")
    if metadata.get("padding_value") != PADDING_VALUE:
        raise ValueError("metadata.padding_value must be -1")

    for key, expected_dtype in _REQUIRED_SHARD_ARRAYS.items():
        if arrays[key].dtype != expected_dtype:
            raise ValueError(f"{key} must have dtype {expected_dtype}, got {arrays[key].dtype}")
    _require_shape(arrays["inputs"], "inputs", (count, rows, cols))
    _require_shape(arrays["finals"], "finals", (count, rows, cols))
    for key in ("op_kind", "op_target", "op_source", "op_scalar_id", "op_scalar_value", "op_mask"):
        _require_shape(arrays[key], key, (count, max_ops))
    if arrays["scalar_vocab"].ndim != 1:
        raise ValueError("scalar_vocab must be one-dimensional")
    if not np.array_equal(arrays["scalar_vocab"], np.unique(arrays["scalar_vocab"])):
        raise ValueError("scalar_vocab must be sorted and unique")
    _validate_op_arrays(arrays, rows=rows, scalar_vocab=arrays["scalar_vocab"])
    return arrays, metadata


def load_hnf_backward_shard(path: str | Path) -> tuple[ShardArrays, dict[str, Any]]:
    shard_path = Path(path)
    arrays, metadata_json = load_shard_arrays(shard_path, _REQUIRED_SHARD_ARRAYS)

    metadata = _metadata_from_array(metadata_json)
    schema_version = metadata.get("schema_version")
    if schema_version != BACKWARD_SCHEMA_VERSION:
        raise ValueError(f"unsupported HNF backward shard schema_version: {schema_version!r}")
    shape = _require_mapping(metadata.get("shape"), "metadata.shape")
    rows = _require_positive_int(shape.get("rows"), "metadata.shape.rows")
    cols = _require_positive_int(shape.get("cols"), "metadata.shape.cols")
    count = _require_positive_int(metadata.get("count"), "metadata.count")
    max_ops = _require_positive_int(metadata.get("max_ops"), "metadata.max_ops")
    _require_positive_int(metadata.get("input_scale"), "metadata.input_scale")
    if metadata.get("padding_value") != PADDING_VALUE:
        raise ValueError("metadata.padding_value must be -1")

    config = _require_mapping(metadata.get("config"), "metadata.config")
    backward_trace = _require_mapping(
        config.get("backward_trace"),
        "metadata.config.backward_trace",
    )
    if backward_trace.get("schema") != BACKWARD_SCHEMA_VERSION:
        raise ValueError("metadata.config.backward_trace.schema does not match schema_version")
    if backward_trace.get("format") != shard_format(shard_path):
        raise ValueError("metadata.config.backward_trace.format must match shard path format")
    if backward_trace.get("require_unimodular_ops") is False:
        raise ValueError("metadata.config.backward_trace.require_unimodular_ops must not be false")
    if backward_trace.get("require_exact_replay") is False:
        raise ValueError("metadata.config.backward_trace.require_exact_replay must not be false")

    for key, expected_dtype in _REQUIRED_SHARD_ARRAYS.items():
        if arrays[key].dtype != expected_dtype:
            raise ValueError(f"{key} must have dtype {expected_dtype}, got {arrays[key].dtype}")
    _require_shape(arrays["inputs"], "inputs", (count, rows, cols))
    _require_shape(arrays["finals"], "finals", (count, rows, cols))
    for key in ("op_kind", "op_target", "op_source", "op_scalar_id", "op_scalar_value", "op_mask"):
        _require_shape(arrays[key], key, (count, max_ops))
    if arrays["scalar_vocab"].ndim != 1:
        raise ValueError("scalar_vocab must be one-dimensional")
    if not np.array_equal(arrays["scalar_vocab"], np.unique(arrays["scalar_vocab"])):
        raise ValueError("scalar_vocab must be sorted and unique")
    _validate_op_arrays(arrays, rows=rows, scalar_vocab=arrays["scalar_vocab"])
    _validate_hnf_replay(arrays)
    return arrays, metadata


class HNFShardSamples:
    """Random-access training examples backed by a validated HNF trajectory shard."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._arrays, self._metadata = _load_validated_hnf_shard(path)
        shape = _require_mapping(self._metadata["shape"], "metadata.shape")
        self.rows = _require_positive_int(shape["rows"], "metadata.shape.rows")
        self.cols = _require_positive_int(shape["cols"], "metadata.shape.cols")
        self.max_ops = _require_positive_int(self._metadata["max_ops"], "metadata.max_ops")
        self.input_scale = _require_positive_int(
            self._metadata["input_scale"],
            "metadata.input_scale",
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def scalar_vocab(self) -> np.ndarray[Any, np.dtype[np.int64]]:
        return cast(np.ndarray[Any, np.dtype[np.int64]], self._arrays["scalar_vocab"])

    @property
    def scalar_vocab_size(self) -> int:
        return int(self._arrays["scalar_vocab"].shape[0])

    def __len__(self) -> int:
        return int(self._arrays["inputs"].shape[0])

    def __getitem__(self, index: int) -> TrainingExample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        inputs = self._arrays["inputs"][index].astype(np.float32)
        inputs /= float(self.input_scale)
        op_mask = self._arrays["op_mask"][index].astype(np.bool_)
        op_kind = self._arrays["op_kind"][index].astype(np.int32)
        op_source_mask = op_mask & np.isin(op_kind, [1, 3])
        op_scalar_mask = op_mask & (op_kind == 3)
        return {
            "inputs": inputs,
            "op_kind": np.where(op_mask, op_kind, 0).astype(np.int32),
            "op_target": np.where(op_mask, self._arrays["op_target"][index], 0).astype(
                np.int32
            ),
            "op_source": np.where(op_source_mask, self._arrays["op_source"][index], 0).astype(
                np.int32
            ),
            "op_scalar": np.where(
                op_scalar_mask,
                self._arrays["op_scalar_id"][index],
                0,
            ).astype(np.int32),
            "op_mask": op_mask,
            "op_source_mask": op_source_mask,
            "op_scalar_mask": op_scalar_mask,
            "value_target": np.asarray(float(np.any(op_mask)), dtype=np.float32),
        }


def make_hnf_grain_dataset(
    path: str | Path,
    batch_size: int,
    seed: int,
    *,
    drop_remainder: bool = False,
) -> MapDataset:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    samples = HNFShardSamples(path)
    return MapDataset.source(samples).shuffle(seed).batch(
        batch_size,
        drop_remainder=drop_remainder,
    )


def integer_row_ops_from_hnf_shard_arrays(
    shard: Mapping[str, Any],
    sample_index: int,
) -> list[IntegerRowOp]:
    op_mask = np.asarray(shard["op_mask"][sample_index])
    op_kind = np.asarray(shard["op_kind"][sample_index])
    op_target = np.asarray(shard["op_target"][sample_index])
    op_source = np.asarray(shard["op_source"][sample_index])
    op_scalar_value = np.asarray(shard["op_scalar_value"][sample_index])
    ops: list[IntegerRowOp] = []
    for op_index, active in enumerate(op_mask):
        if not bool(active):
            continue
        ops.append(
            _decode_integer_row_op(
                int(op_kind[op_index]),
                int(op_target[op_index]),
                int(op_source[op_index]),
                int(op_scalar_value[op_index]),
            )
        )
    return ops


def hnf_trajectories_from_shard(path: str | Path) -> list[HNFTrajectory]:
    arrays, _metadata = _load_validated_hnf_shard(path)
    trajectories: list[HNFTrajectory] = []
    for sample_index in range(arrays["inputs"].shape[0]):
        ops = integer_row_ops_from_hnf_shard_arrays(arrays, sample_index)
        trajectories.append(
            HNFTrajectory(
                input_matrix=cast(Matrix, arrays["inputs"][sample_index].tolist()),
                final_matrix=cast(Matrix, arrays["finals"][sample_index].tolist()),
                ops=tuple(ops),
            )
        )
    return trajectories
