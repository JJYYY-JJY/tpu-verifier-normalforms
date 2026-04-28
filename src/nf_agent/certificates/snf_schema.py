from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, TypeGuard, cast

SNF_CERTIFICATE_SCHEMA_VERSION: Literal["snf-certificate-json-v0.1"] = (
    "snf-certificate-json-v0.1"
)
SNF_CERTIFICATE_KIND: Literal["snf_int"] = "snf_int"

IntegerMatrix: TypeAlias = list[list[int]]
IntegerMatrixOpKind = Literal["swap", "negate", "add"]

_CERTIFICATE_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "shape",
        "input",
        "diagonal",
        "left_transform",
        "right_transform",
        "row_ops",
        "col_ops",
    }
)


@dataclass(frozen=True)
class IntegerMatrixOp:
    kind: IntegerMatrixOpKind
    target: int
    source: int | None = None
    scalar: int | None = None


@dataclass(frozen=True)
class SNFCertificate:
    kind: Literal["snf_int"]
    schema_version: Literal["snf-certificate-json-v0.1"]
    shape: tuple[int, int]
    input: IntegerMatrix
    diagonal: IntegerMatrix
    left_transform: IntegerMatrix
    right_transform: IntegerMatrix
    row_ops: list[IntegerMatrixOp]
    col_ops: list[IntegerMatrixOp]


def snf_certificate_json_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Integer Smith normal form certificate",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "kind",
            "schema_version",
            "shape",
            "input",
            "diagonal",
            "left_transform",
            "right_transform",
            "row_ops",
            "col_ops",
        ],
        "properties": {
            "kind": {"const": SNF_CERTIFICATE_KIND},
            "schema_version": {"const": SNF_CERTIFICATE_SCHEMA_VERSION},
            "shape": {
                "type": "array",
                "prefixItems": [
                    {"type": "integer", "minimum": 0},
                    {"type": "integer", "minimum": 0},
                ],
                "minItems": 2,
                "maxItems": 2,
            },
            "input": {"$ref": "#/$defs/integerMatrix"},
            "diagonal": {"$ref": "#/$defs/integerMatrix"},
            "left_transform": {"$ref": "#/$defs/integerMatrix"},
            "right_transform": {"$ref": "#/$defs/integerMatrix"},
            "row_ops": {
                "type": "array",
                "items": {"$ref": "#/$defs/integerMatrixOp"},
            },
            "col_ops": {
                "type": "array",
                "items": {"$ref": "#/$defs/integerMatrixOp"},
            },
        },
        "$defs": {
            "integerMatrix": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
            "integerMatrixOp": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "target"],
                "properties": {
                    "kind": {"enum": ["swap", "negate", "add"]},
                    "target": {"type": "integer", "minimum": 0},
                    "source": {"type": "integer", "minimum": 0},
                    "scalar": {"type": "integer"},
                },
                "oneOf": [
                    {
                        "properties": {"kind": {"const": "swap"}},
                        "required": ["source"],
                        "not": {"required": ["scalar"]},
                    },
                    {
                        "properties": {"kind": {"const": "negate"}},
                        "not": {
                            "anyOf": [
                                {"required": ["source"]},
                                {"required": ["scalar"]},
                            ]
                        },
                    },
                    {
                        "properties": {"kind": {"const": "add"}},
                        "required": ["source", "scalar"],
                    },
                ],
            },
        },
    }


def validate_snf_certificate_record(payload: Mapping[str, object]) -> SNFCertificate:
    keys = set(payload)
    missing = _CERTIFICATE_FIELDS - keys
    if missing:
        raise ValueError(f"missing SNF certificate field: {sorted(missing)[0]}")
    unexpected = keys - _CERTIFICATE_FIELDS
    if unexpected:
        raise ValueError(f"unexpected SNF certificate field: {sorted(unexpected)[0]}")

    kind = payload["kind"]
    if kind != SNF_CERTIFICATE_KIND:
        raise ValueError(f"SNF certificate kind must be {SNF_CERTIFICATE_KIND!r}")

    schema_version = payload["schema_version"]
    if schema_version != SNF_CERTIFICATE_SCHEMA_VERSION:
        raise ValueError(
            f"SNF certificate schema_version must be {SNF_CERTIFICATE_SCHEMA_VERSION!r}"
        )

    rows, cols = _require_shape(payload["shape"])
    input_matrix = _require_integer_matrix(payload["input"], "input", rows, cols)
    diagonal = _require_integer_matrix(payload["diagonal"], "diagonal", rows, cols)
    left_transform = _require_integer_matrix(
        payload["left_transform"],
        "left_transform",
        rows,
        rows,
    )
    right_transform = _require_integer_matrix(
        payload["right_transform"],
        "right_transform",
        cols,
        cols,
    )
    _require_snf_diagonal(diagonal)
    row_ops = _require_ops(payload["row_ops"], "row_ops", rows)
    col_ops = _require_ops(payload["col_ops"], "col_ops", cols)

    return SNFCertificate(
        kind=SNF_CERTIFICATE_KIND,
        schema_version=SNF_CERTIFICATE_SCHEMA_VERSION,
        shape=(rows, cols),
        input=input_matrix,
        diagonal=diagonal,
        left_transform=left_transform,
        right_transform=right_transform,
        row_ops=row_ops,
        col_ops=col_ops,
    )


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _require_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _require_shape(value: object) -> tuple[int, int]:
    if not _is_sequence(value):
        raise TypeError("shape must be a two-entry sequence")
    entries = list(value)
    if len(entries) != 2:
        raise ValueError("shape must contain exactly two dimensions")
    rows = _require_int(entries[0], "shape row count")
    cols = _require_int(entries[1], "shape column count")
    if rows < 0 or cols < 0:
        raise ValueError("shape dimensions must be nonnegative")
    return rows, cols


def _require_integer_matrix(
    value: object,
    name: str,
    expected_rows: int,
    expected_cols: int,
) -> IntegerMatrix:
    if not _is_sequence(value):
        raise TypeError(f"{name} must be a sequence of rows")
    rows = list(value)
    if len(rows) != expected_rows:
        raise ValueError(f"{name} must have {expected_rows} rows")

    normalized: IntegerMatrix = []
    for row_index, row in enumerate(rows):
        if not _is_sequence(row):
            raise TypeError(f"{name} row {row_index} must be a sequence")
        values = list(row)
        if len(values) != expected_cols:
            raise ValueError(f"{name} row {row_index} must have {expected_cols} entries")
        normalized.append(
            [
                _require_int(entry, f"{name} entry ({row_index},{col_index})")
                for col_index, entry in enumerate(values)
            ]
        )
    return normalized


def _require_snf_diagonal(matrix: IntegerMatrix) -> None:
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    diagonal_length = min(rows, cols)
    previous_nonzero: int | None = None
    zero_seen = False

    for row_index, row in enumerate(matrix):
        for col_index, entry in enumerate(row):
            if row_index != col_index and entry != 0:
                raise ValueError("diagonal off-diagonal entries must be zero")

    for index in range(diagonal_length):
        entry = matrix[index][index]
        if entry < 0:
            raise ValueError("diagonal entries must be nonnegative")
        if entry == 0:
            zero_seen = True
            continue
        if zero_seen:
            raise ValueError("diagonal entries must remain zero after zero appears")
        if previous_nonzero is not None and entry % previous_nonzero != 0:
            raise ValueError("each nonzero diagonal entry must divide the next one")
        previous_nonzero = entry


def _require_ops(value: object, name: str, bound: int) -> list[IntegerMatrixOp]:
    if not _is_sequence(value):
        raise TypeError(f"{name} must be a sequence of operations")
    return [_require_op(op, f"{name}[{index}]", bound) for index, op in enumerate(value)]


def _require_op(value: object, name: str, bound: int) -> IntegerMatrixOp:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an operation object")
    op = cast(Mapping[str, object], value)
    keys = set(op)
    allowed_keys = {"kind", "target", "source", "scalar"}
    unexpected = keys - allowed_keys
    if unexpected:
        unexpected_field = next(iter(unexpected))
        raise ValueError(f"{name} has unexpected field: {unexpected_field!r}")
    if "kind" not in op:
        raise ValueError(f"{name} missing operation kind")
    if "target" not in op:
        raise ValueError(f"{name} missing target")

    kind = op["kind"]
    if kind not in {"swap", "negate", "add"}:
        raise ValueError(f"{name} has invalid operation kind")

    target = _require_index(op["target"], f"{name} target", bound)

    if kind == "swap":
        if "source" not in op:
            raise ValueError(f"{name} swap operation requires source")
        if "scalar" in op:
            raise ValueError(f"{name} swap operation must not include scalar")
        source = _require_index(op["source"], f"{name} source", bound)
        return IntegerMatrixOp(kind="swap", target=target, source=source)

    if kind == "negate":
        if "source" in op or "scalar" in op:
            raise ValueError(f"{name} negate operation only allows target")
        return IntegerMatrixOp(kind="negate", target=target)

    if "source" not in op:
        raise ValueError(f"{name} add operation requires source")
    if "scalar" not in op:
        raise ValueError(f"{name} add operation requires scalar")
    source = _require_index(op["source"], f"{name} source", bound)
    scalar = _require_int(op["scalar"], f"{name} scalar")
    if target == source:
        raise ValueError(f"{name} add operation target and source must be distinct")
    return IntegerMatrixOp(kind="add", target=target, source=source, scalar=scalar)


def _require_index(value: object, name: str, bound: int) -> int:
    index = _require_int(value, name)
    if not 0 <= index < bound:
        raise IndexError(f"{name} index out of range for bound {bound}")
    return index
