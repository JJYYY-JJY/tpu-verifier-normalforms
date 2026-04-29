from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from grain import MapDataset  # type: ignore[import-untyped]

from nf_agent.data.rref_backward_shards import (
    SCHEMA_VERSION as BACKWARD_SCHEMA_VERSION,
)
from nf_agent.data.rref_backward_shards import (
    load_rref_backward_shard,
    row_ops_from_backward_shard_arrays,
)
from nf_agent.data.shard_storage import load_shard_arrays, shard_format, write_shard_arrays
from nf_agent.env.elementary_ops import require_prime
from nf_agent.env.rref_modp import RowOp, is_rref_modp, replay_row_ops

SCHEMA_VERSION = "rref-state-action-npz-v1"
PADDING_VALUE = -1

ShardValue: TypeAlias = np.ndarray[Any, np.dtype[Any]]
ShardArrays: TypeAlias = dict[str, ShardValue]
TrainingExample: TypeAlias = dict[str, ShardValue]

_REQUIRED_ARRAYS: Mapping[str, np.dtype[Any]] = {
    "states": np.dtype(np.int64),
    "action_kind": np.dtype(np.int8),
    "action_target": np.dtype(np.int64),
    "action_source": np.dtype(np.int64),
    "action_scalar": np.dtype(np.int64),
    "stop_label": np.dtype(np.bool_),
    "legal_kind_mask": np.dtype(np.bool_),
    "legal_target_mask": np.dtype(np.bool_),
    "legal_source_mask": np.dtype(np.bool_),
    "legal_target_source_mask": np.dtype(np.bool_),
    "legal_scalar_mask": np.dtype(np.bool_),
    "trace_states": np.dtype(np.int64),
    "trace_action_kind": np.dtype(np.int8),
    "trace_action_target": np.dtype(np.int64),
    "trace_action_source": np.dtype(np.int64),
    "trace_action_scalar": np.dtype(np.int64),
    "trace_stop_label": np.dtype(np.bool_),
    "trace_step_mask": np.dtype(np.bool_),
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
    for row_index, row in enumerate(mask):
        inactive_seen = False
        for value in row:
            active = bool(value)
            if inactive_seen and active:
                raise ValueError(f"{name} must be true-prefix padded at row {row_index}")
            if not active:
                inactive_seen = True


def _field_modulus_from_backward_metadata(metadata: Mapping[str, Any]) -> int:
    config = _require_mapping(metadata.get("config"), "source.metadata.config")
    field = _require_mapping(config.get("field"), "source.metadata.config.field")
    modulus = _require_int(field.get("modulus"), "source.metadata.config.field.modulus")
    require_prime(modulus)
    return modulus


def _metadata_json(
    *,
    source_metadata: Mapping[str, Any],
    trace_shard_path: Path,
    rows: int,
    cols: int,
    modulus: int,
    trace_count: int,
    flat_count: int,
    max_ops: int,
    storage_format: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "format": storage_format,
        "source_schema_version": source_metadata.get("schema_version"),
        "source_path": str(trace_shard_path),
        "source_count": source_metadata.get("count"),
        "source_config": source_metadata.get("config"),
        "shape": {"rows": rows, "cols": cols},
        "modulus": modulus,
        "trace_count": trace_count,
        "flat_count": flat_count,
        "max_ops": max_ops,
        "max_steps": max_ops + 1,
        "op_encoding": {"stop": 0, "swap": 1, "scale": 2, "add": 3},
        "action_columns": ["kind", "target", "source", "scalar"],
        "padding_value": PADDING_VALUE,
        "includes_trace_tensors": True,
        "generation": "exact-rref-backward-trace-state-action-expansion",
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _encode_row_op(op: RowOp, p: int) -> tuple[int, int, int, int]:
    if op.kind == "swap":
        if op.source is None:
            raise ValueError("swap op requires source row")
        return 1, op.target, op.source, PADDING_VALUE
    if op.kind == "scale":
        if op.scalar is None:
            raise ValueError("scale op requires scalar")
        scalar = op.scalar % p
        if scalar == 0:
            raise ValueError("scale action scalar must be nonzero modulo p")
        return 2, op.target, PADDING_VALUE, scalar
    if op.kind == "add":
        if op.source is None or op.scalar is None:
            raise ValueError("add op requires source row and scalar")
        scalar = op.scalar % p
        if scalar == 0:
            raise ValueError("add action scalar must be nonzero modulo p")
        return 3, op.target, op.source, scalar
    raise ValueError(f"unknown row operation kind: {op.kind}")


def _decode_action(kind: int, target: int, source: int, scalar: int) -> RowOp:
    if kind == 1:
        return RowOp.swap(target, source)
    if kind == 2:
        return RowOp.scale(target, scalar)
    if kind == 3:
        return RowOp.add(target, source, scalar)
    raise ValueError(f"action kind {kind} is not a row operation")


def _base_legal_target_source_mask(rows: int) -> ShardValue:
    row_ids = np.arange(rows)
    return cast(ShardValue, (row_ids[:, None] != row_ids[None, :]).astype(np.bool_))


def _base_legal_scalar_mask(p: int) -> ShardValue:
    mask = np.ones((p,), dtype=np.bool_)
    mask[0] = False
    return mask


def _legal_kind_mask_for_state(state: Sequence[Sequence[int]], *, rows: int, p: int) -> ShardValue:
    mask = np.zeros((4,), dtype=np.bool_)
    mask[0] = is_rref_modp(state, p)
    mask[1] = rows > 1
    mask[2] = rows > 0
    mask[3] = rows > 1
    return mask


def _set_flat_sample(
    arrays: ShardArrays,
    index: int,
    *,
    state: Sequence[Sequence[int]],
    action: tuple[int, int, int, int],
    stop: bool,
    rows: int,
    p: int,
    target_source_mask: ShardValue,
    scalar_mask: ShardValue,
) -> None:
    kind, target, source, scalar = action
    arrays["states"][index] = np.asarray(state, dtype=np.int64)
    arrays["action_kind"][index] = kind
    arrays["action_target"][index] = target
    arrays["action_source"][index] = source
    arrays["action_scalar"][index] = scalar
    arrays["stop_label"][index] = stop
    arrays["legal_kind_mask"][index] = _legal_kind_mask_for_state(state, rows=rows, p=p)
    arrays["legal_target_mask"][index] = True
    arrays["legal_source_mask"][index] = True
    arrays["legal_target_source_mask"][index] = target_source_mask
    arrays["legal_scalar_mask"][index] = scalar_mask


def _set_trace_sample(
    arrays: ShardArrays,
    trace_index: int,
    step_index: int,
    *,
    state: Sequence[Sequence[int]],
    action: tuple[int, int, int, int],
    stop: bool,
) -> None:
    kind, target, source, scalar = action
    arrays["trace_states"][trace_index, step_index] = np.asarray(state, dtype=np.int64)
    arrays["trace_action_kind"][trace_index, step_index] = kind
    arrays["trace_action_target"][trace_index, step_index] = target
    arrays["trace_action_source"][trace_index, step_index] = source
    arrays["trace_action_scalar"][trace_index, step_index] = scalar
    arrays["trace_stop_label"][trace_index, step_index] = stop
    arrays["trace_step_mask"][trace_index, step_index] = True


def generate_rref_state_shard(trace_shard_path: str | Path) -> ShardArrays:
    trace_path = Path(trace_shard_path)
    shard_format(trace_path)
    if not trace_path.exists():
        raise ValueError(f"trace shard path does not exist: {trace_path}")

    source_arrays, source_metadata = load_rref_backward_shard(trace_path)
    modulus = _field_modulus_from_backward_metadata(source_metadata)
    shape = _require_mapping(source_metadata.get("shape"), "source.metadata.shape")
    rows = _require_positive_int(shape.get("rows"), "source.metadata.shape.rows")
    cols = _require_positive_int(shape.get("cols"), "source.metadata.shape.cols")
    trace_count = int(source_arrays["inputs"].shape[0])
    max_ops = int(source_arrays["ops"].shape[1])
    op_counts = source_arrays["op_mask"].sum(axis=1).astype(np.int64)
    flat_count = int(op_counts.sum()) + trace_count
    max_steps = max_ops + 1

    arrays: ShardArrays = {
        "states": np.empty((flat_count, rows, cols), dtype=np.int64),
        "action_kind": np.zeros((flat_count,), dtype=np.int8),
        "action_target": np.full((flat_count,), PADDING_VALUE, dtype=np.int64),
        "action_source": np.full((flat_count,), PADDING_VALUE, dtype=np.int64),
        "action_scalar": np.full((flat_count,), PADDING_VALUE, dtype=np.int64),
        "stop_label": np.zeros((flat_count,), dtype=np.bool_),
        "legal_kind_mask": np.zeros((flat_count, 4), dtype=np.bool_),
        "legal_target_mask": np.zeros((flat_count, rows), dtype=np.bool_),
        "legal_source_mask": np.zeros((flat_count, rows), dtype=np.bool_),
        "legal_target_source_mask": np.zeros((flat_count, rows, rows), dtype=np.bool_),
        "legal_scalar_mask": np.zeros((flat_count, modulus), dtype=np.bool_),
        "trace_states": np.full(
            (trace_count, max_steps, rows, cols),
            PADDING_VALUE,
            dtype=np.int64,
        ),
        "trace_action_kind": np.zeros((trace_count, max_steps), dtype=np.int8),
        "trace_action_target": np.full(
            (trace_count, max_steps),
            PADDING_VALUE,
            dtype=np.int64,
        ),
        "trace_action_source": np.full(
            (trace_count, max_steps),
            PADDING_VALUE,
            dtype=np.int64,
        ),
        "trace_action_scalar": np.full(
            (trace_count, max_steps),
            PADDING_VALUE,
            dtype=np.int64,
        ),
        "trace_stop_label": np.zeros((trace_count, max_steps), dtype=np.bool_),
        "trace_step_mask": np.zeros((trace_count, max_steps), dtype=np.bool_),
    }
    target_source_mask = _base_legal_target_source_mask(rows)
    scalar_mask = _base_legal_scalar_mask(modulus)

    flat_index = 0
    for trace_index in range(trace_count):
        current = source_arrays["inputs"][trace_index].tolist()
        ops = row_ops_from_backward_shard_arrays(source_arrays, trace_index)

        for step_index, op in enumerate(ops):
            action = _encode_row_op(op, modulus)
            _set_flat_sample(
                arrays,
                flat_index,
                state=current,
                action=action,
                stop=False,
                rows=rows,
                p=modulus,
                target_source_mask=target_source_mask,
                scalar_mask=scalar_mask,
            )
            _set_trace_sample(
                arrays,
                trace_index,
                step_index,
                state=current,
                action=action,
                stop=False,
            )
            current = replay_row_ops(current, [op], modulus)
            flat_index += 1

        stop_action = (0, PADDING_VALUE, PADDING_VALUE, PADDING_VALUE)
        _set_flat_sample(
            arrays,
            flat_index,
            state=current,
            action=stop_action,
            stop=True,
            rows=rows,
            p=modulus,
            target_source_mask=target_source_mask,
            scalar_mask=scalar_mask,
        )
        _set_trace_sample(
            arrays,
            trace_index,
            len(ops),
            state=current,
            action=stop_action,
            stop=True,
        )
        flat_index += 1

    if flat_index != flat_count:
        raise ValueError(f"generated flat_count mismatch: expected {flat_count}, got {flat_index}")

    arrays["metadata_json"] = np.asarray(
        _metadata_json(
            source_metadata=source_metadata,
            trace_shard_path=trace_path,
            rows=rows,
            cols=cols,
            modulus=modulus,
            trace_count=trace_count,
            flat_count=flat_count,
            max_ops=max_ops,
            storage_format="npz",
        )
    )
    metadata = _metadata_from_array(arrays["metadata_json"])
    _validate_state_action_arrays(arrays, metadata)
    return arrays


def write_rref_state_shard(trace_shard_path: str | Path, out_path: str | Path) -> None:
    path = Path(out_path)
    storage_format = shard_format(path)
    shard = generate_rref_state_shard(trace_shard_path)
    metadata = _metadata_from_array(shard["metadata_json"])
    metadata["format"] = storage_format
    metadata["source_path"] = str(Path(trace_shard_path))
    shard["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    _validate_state_action_arrays(shard, metadata)
    write_shard_arrays(path, shard)


def _validate_normalized_matrix_entries(array: ShardValue, name: str, p: int) -> None:
    if np.any(array < 0) or np.any(array >= p):
        raise ValueError(f"{name} entries must be normalized modulo p")


def _validate_action_values(
    kind: int,
    target: int,
    source: int,
    scalar: int,
    *,
    rows: int,
    p: int,
    label: str,
) -> None:
    if kind == 0:
        if target != PADDING_VALUE or source != PADDING_VALUE or scalar != PADDING_VALUE:
            raise ValueError(f"{label} stop action must use -1 target/source/scalar")
        return
    if kind not in {1, 2, 3}:
        raise ValueError(f"{label} action_kind must be one of 0, 1, 2, or 3")
    if target < 0 or target >= rows:
        raise ValueError(f"{label} action_target must be a valid row index")
    if kind in {1, 3}:
        if source < 0 or source >= rows:
            raise ValueError(f"{label} action_source must be a valid row index")
        if source == target:
            raise ValueError(f"{label} action target/source rows must be distinct")
    elif source != PADDING_VALUE:
        raise ValueError(f"{label} scale action_source must be -1")
    if kind == 1:
        if scalar != PADDING_VALUE:
            raise ValueError(f"{label} swap action_scalar must be -1")
    elif scalar <= 0 or scalar >= p:
        raise ValueError(f"{label} scale/add action_scalar must be in [1, p)")


def _validate_flat_actions(arrays: Mapping[str, ShardValue], rows: int, p: int) -> None:
    for index in range(int(arrays["states"].shape[0])):
        kind = int(arrays["action_kind"][index])
        target = int(arrays["action_target"][index])
        source = int(arrays["action_source"][index])
        scalar = int(arrays["action_scalar"][index])
        stop = bool(arrays["stop_label"][index])
        if stop != (kind == 0):
            raise ValueError(f"flat sample {index} stop_label must match action_kind == 0")
        _validate_action_values(
            kind,
            target,
            source,
            scalar,
            rows=rows,
            p=p,
            label=f"flat sample {index}",
        )


def _validate_legal_masks(arrays: Mapping[str, ShardValue], rows: int, p: int) -> None:
    expected_target = np.ones((rows,), dtype=np.bool_)
    expected_source = np.ones((rows,), dtype=np.bool_)
    expected_target_source = _base_legal_target_source_mask(rows)
    expected_scalar = _base_legal_scalar_mask(p)
    for index, state in enumerate(arrays["states"]):
        expected_kind = _legal_kind_mask_for_state(state.tolist(), rows=rows, p=p)
        if not np.array_equal(arrays["legal_kind_mask"][index], expected_kind):
            raise ValueError(f"flat sample {index} legal_kind_mask is invalid")
        if not np.array_equal(arrays["legal_target_mask"][index], expected_target):
            raise ValueError(f"flat sample {index} legal_target_mask is invalid")
        if not np.array_equal(arrays["legal_source_mask"][index], expected_source):
            raise ValueError(f"flat sample {index} legal_source_mask is invalid")
        if not np.array_equal(arrays["legal_target_source_mask"][index], expected_target_source):
            raise ValueError(f"flat sample {index} legal_target_source_mask is invalid")
        if not np.array_equal(arrays["legal_scalar_mask"][index], expected_scalar):
            raise ValueError(f"flat sample {index} legal_scalar_mask is invalid")


def _validate_trace_padding(arrays: Mapping[str, ShardValue]) -> None:
    inactive = np.logical_not(arrays["trace_step_mask"])
    if np.any(arrays["trace_states"][inactive] != PADDING_VALUE):
        raise ValueError("trace_states padding must be -1 where trace_step_mask is false")
    if np.any(arrays["trace_action_kind"][inactive] != 0):
        raise ValueError("trace_action_kind padding must be 0 where trace_step_mask is false")
    for name in ("trace_action_target", "trace_action_source", "trace_action_scalar"):
        if np.any(arrays[name][inactive] != PADDING_VALUE):
            raise ValueError(f"{name} padding must be -1 where trace_step_mask is false")
    if np.any(arrays["trace_stop_label"][inactive]):
        raise ValueError("trace_stop_label padding must be false where trace_step_mask is false")


def _validate_trace_actions_and_replay(
    arrays: Mapping[str, ShardValue],
    *,
    trace_count: int,
    max_steps: int,
    rows: int,
    p: int,
) -> None:
    _require_prefix_mask(arrays["trace_step_mask"], "trace_step_mask")
    for trace_index in range(trace_count):
        active_steps = int(arrays["trace_step_mask"][trace_index].sum())
        if active_steps <= 0:
            raise ValueError(f"trace {trace_index} must include at least one stop step")
        stop_step = active_steps - 1
        for step_index in range(max_steps):
            kind = int(arrays["trace_action_kind"][trace_index, step_index])
            target = int(arrays["trace_action_target"][trace_index, step_index])
            source = int(arrays["trace_action_source"][trace_index, step_index])
            scalar = int(arrays["trace_action_scalar"][trace_index, step_index])
            active = bool(arrays["trace_step_mask"][trace_index, step_index])
            stop = bool(arrays["trace_stop_label"][trace_index, step_index])
            if not active:
                continue
            if stop != (step_index == stop_step):
                raise ValueError(f"trace {trace_index} must stop exactly at its final active step")
            if stop != (kind == 0):
                raise ValueError(
                    f"trace {trace_index} step {step_index} stop_label must match action_kind == 0"
                )
            _validate_action_values(
                kind,
                target,
                source,
                scalar,
                rows=rows,
                p=p,
                label=f"trace {trace_index} step {step_index}",
            )
            state = arrays["trace_states"][trace_index, step_index]
            _validate_normalized_matrix_entries(
                state,
                f"trace {trace_index} step {step_index} state",
                p,
            )
            if step_index < stop_step:
                op = _decode_action(kind, target, source, scalar)
                replayed = replay_row_ops(state.tolist(), [op], p)
                expected = arrays["trace_states"][trace_index, step_index + 1].tolist()
                if replayed != expected:
                    raise ValueError(
                        f"trace {trace_index} step {step_index} does not replay to next state"
                    )
            else:
                final = state.tolist()
                if not is_rref_modp(final, p):
                    raise ValueError(f"trace {trace_index} final stop state is not RREF")


def _validate_flat_trace_consistency(arrays: Mapping[str, ShardValue]) -> None:
    flat_index = 0
    trace_count = int(arrays["trace_step_mask"].shape[0])
    for trace_index in range(trace_count):
        active_steps = int(arrays["trace_step_mask"][trace_index].sum())
        for step_index in range(active_steps):
            if not np.array_equal(
                arrays["states"][flat_index],
                arrays["trace_states"][trace_index, step_index],
            ):
                raise ValueError(
                    f"flat sample {flat_index} state does not match trace "
                    f"{trace_index} step {step_index}"
                )
            for flat_name, trace_name in (
                ("action_kind", "trace_action_kind"),
                ("action_target", "trace_action_target"),
                ("action_source", "trace_action_source"),
                ("action_scalar", "trace_action_scalar"),
                ("stop_label", "trace_stop_label"),
            ):
                if arrays[flat_name][flat_index] != arrays[trace_name][trace_index, step_index]:
                    raise ValueError(
                        f"flat sample {flat_index} {flat_name} does not match trace "
                        f"{trace_index} step {step_index}"
                    )
            flat_index += 1
    if flat_index != int(arrays["states"].shape[0]):
        raise ValueError("flat samples do not match active trace steps")


def _validate_state_action_arrays(
    arrays: Mapping[str, ShardValue],
    metadata: Mapping[str, Any],
) -> None:
    schema_version = metadata.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported RREF state-action shard schema_version: {schema_version!r}")
    if metadata.get("source_schema_version") != BACKWARD_SCHEMA_VERSION:
        raise ValueError("metadata.source_schema_version must be rref-backward-trace-npz-v1")
    shape = _require_mapping(metadata.get("shape"), "metadata.shape")
    rows = _require_positive_int(shape.get("rows"), "metadata.shape.rows")
    cols = _require_positive_int(shape.get("cols"), "metadata.shape.cols")
    modulus = _require_int(metadata.get("modulus"), "metadata.modulus")
    require_prime(modulus)
    trace_count = _require_positive_int(metadata.get("trace_count"), "metadata.trace_count")
    flat_count = _require_positive_int(metadata.get("flat_count"), "metadata.flat_count")
    max_ops = _require_positive_int(metadata.get("max_ops"), "metadata.max_ops")
    max_steps = _require_positive_int(metadata.get("max_steps"), "metadata.max_steps")
    if max_steps != max_ops + 1:
        raise ValueError("metadata.max_steps must equal metadata.max_ops + 1")
    if metadata.get("padding_value") != PADDING_VALUE:
        raise ValueError("metadata.padding_value must be -1")
    if metadata.get("includes_trace_tensors") is not True:
        raise ValueError("metadata.includes_trace_tensors must be true")
    if metadata.get("op_encoding") != {"stop": 0, "swap": 1, "scale": 2, "add": 3}:
        raise ValueError("metadata.op_encoding is invalid")

    for key, expected_dtype in _REQUIRED_ARRAYS.items():
        if arrays[key].dtype != expected_dtype:
            raise ValueError(f"{key} must have dtype {expected_dtype}, got {arrays[key].dtype}")

    _require_shape(arrays["states"], "states", (flat_count, rows, cols))
    for key in ("action_kind", "action_target", "action_source", "action_scalar", "stop_label"):
        _require_shape(arrays[key], key, (flat_count,))
    _require_shape(arrays["legal_kind_mask"], "legal_kind_mask", (flat_count, 4))
    _require_shape(arrays["legal_target_mask"], "legal_target_mask", (flat_count, rows))
    _require_shape(arrays["legal_source_mask"], "legal_source_mask", (flat_count, rows))
    _require_shape(
        arrays["legal_target_source_mask"],
        "legal_target_source_mask",
        (flat_count, rows, rows),
    )
    _require_shape(arrays["legal_scalar_mask"], "legal_scalar_mask", (flat_count, modulus))
    _require_shape(
        arrays["trace_states"],
        "trace_states",
        (trace_count, max_steps, rows, cols),
    )
    for key in (
        "trace_action_kind",
        "trace_action_target",
        "trace_action_source",
        "trace_action_scalar",
        "trace_stop_label",
        "trace_step_mask",
    ):
        _require_shape(arrays[key], key, (trace_count, max_steps))

    if int(arrays["trace_step_mask"].sum()) != flat_count:
        raise ValueError("metadata.flat_count must equal active trace step count")

    _validate_normalized_matrix_entries(arrays["states"], "states", modulus)
    _validate_flat_actions(arrays, rows, modulus)
    _validate_legal_masks(arrays, rows, modulus)
    _validate_trace_padding(arrays)
    _validate_trace_actions_and_replay(
        arrays,
        trace_count=trace_count,
        max_steps=max_steps,
        rows=rows,
        p=modulus,
    )
    _validate_flat_trace_consistency(arrays)


def load_rref_state_shard(path: str | Path) -> tuple[ShardArrays, dict[str, Any]]:
    shard_path = Path(path)
    arrays, metadata_json = load_shard_arrays(shard_path, _REQUIRED_ARRAYS)

    metadata = _metadata_from_array(metadata_json)
    _validate_state_action_arrays(arrays, metadata)
    return arrays, metadata


class RREFStateActionSamples:
    """Random-access flat state/action examples backed by a validated RREF shard."""

    def __init__(self, path: str | Path) -> None:
        self._arrays, self._metadata = load_rref_state_shard(path)
        shape = _require_mapping(self._metadata["shape"], "metadata.shape")
        self.rows = _require_positive_int(shape["rows"], "metadata.shape.rows")
        self.cols = _require_positive_int(shape["cols"], "metadata.shape.cols")
        self.modulus = _require_int(self._metadata["modulus"], "metadata.modulus")

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def __len__(self) -> int:
        return int(self._arrays["states"].shape[0])

    def __getitem__(self, index: int) -> TrainingExample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        state = (self._arrays["states"][index] % self.modulus).astype(np.float32)
        state /= float(self.modulus - 1)
        return {
            "state": state,
            "action_kind": np.asarray(self._arrays["action_kind"][index], dtype=np.int32),
            "action_target": np.asarray(self._arrays["action_target"][index], dtype=np.int32),
            "action_source": np.asarray(self._arrays["action_source"][index], dtype=np.int32),
            "action_scalar": np.asarray(self._arrays["action_scalar"][index], dtype=np.int32),
            "stop_label": np.asarray(self._arrays["stop_label"][index], dtype=np.bool_),
            "legal_kind_mask": self._arrays["legal_kind_mask"][index].astype(np.bool_),
            "legal_target_mask": self._arrays["legal_target_mask"][index].astype(np.bool_),
            "legal_source_mask": self._arrays["legal_source_mask"][index].astype(np.bool_),
            "legal_target_source_mask": self._arrays["legal_target_source_mask"][index].astype(
                np.bool_
            ),
            "legal_scalar_mask": self._arrays["legal_scalar_mask"][index].astype(np.bool_),
        }


def make_rref_state_action_grain_dataset(
    path: str | Path,
    batch_size: int,
    seed: int,
    *,
    drop_remainder: bool = False,
) -> MapDataset:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    samples = RREFStateActionSamples(path)
    return MapDataset.source(samples).shuffle(seed).batch(
        batch_size,
        drop_remainder=drop_remainder,
    )
