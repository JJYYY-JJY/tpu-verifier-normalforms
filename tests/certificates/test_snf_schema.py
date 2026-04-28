from collections.abc import Callable
from copy import deepcopy

import pytest

from nf_agent.certificates import (
    SNF_CERTIFICATE_KIND,
    SNF_CERTIFICATE_SCHEMA_VERSION,
    IntegerMatrixOp,
    SNFCertificate,
    snf_certificate_json_schema,
    validate_snf_certificate_record,
)


def _valid_payload() -> dict[str, object]:
    return {
        "kind": SNF_CERTIFICATE_KIND,
        "schema_version": SNF_CERTIFICATE_SCHEMA_VERSION,
        "shape": [2, 2],
        "input": [[2, 4], [6, 8]],
        "diagonal": [[2, 0], [0, 4]],
        "left_transform": [[1, 0], [0, 1]],
        "right_transform": [[1, 0], [0, 1]],
        "row_ops": [
            {"kind": "swap", "target": 0, "source": 1},
            {"kind": "negate", "target": 1},
            {"kind": "add", "target": 1, "source": 0, "scalar": -3},
        ],
        "col_ops": [
            {"kind": "add", "target": 0, "source": 1, "scalar": 2},
        ],
    }


def test_valid_snf_certificate_parses_to_normalized_dataclasses() -> None:
    payload = _valid_payload()

    certificate = validate_snf_certificate_record(payload)
    payload["input"] = [[99]]

    assert certificate == SNFCertificate(
        kind=SNF_CERTIFICATE_KIND,
        schema_version=SNF_CERTIFICATE_SCHEMA_VERSION,
        shape=(2, 2),
        input=[[2, 4], [6, 8]],
        diagonal=[[2, 0], [0, 4]],
        left_transform=[[1, 0], [0, 1]],
        right_transform=[[1, 0], [0, 1]],
        row_ops=[
            IntegerMatrixOp(kind="swap", target=0, source=1),
            IntegerMatrixOp(kind="negate", target=1),
            IntegerMatrixOp(kind="add", target=1, source=0, scalar=-3),
        ],
        col_ops=[IntegerMatrixOp(kind="add", target=0, source=1, scalar=2)],
    )


def test_snf_certificate_json_schema_exposes_contract() -> None:
    schema = snf_certificate_json_schema()

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
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
    assert schema["properties"]["kind"]["const"] == SNF_CERTIFICATE_KIND
    assert schema["properties"]["schema_version"]["const"] == SNF_CERTIFICATE_SCHEMA_VERSION
    assert schema["properties"]["shape"]["prefixItems"][0]["minimum"] == 0
    assert schema["properties"]["input"]["$ref"] == "#/$defs/integerMatrix"
    assert schema["properties"]["row_ops"]["items"]["$ref"] == "#/$defs/integerMatrixOp"
    assert schema["$defs"]["integerMatrix"]["items"]["items"]["type"] == "integer"
    assert schema["$defs"]["integerMatrixOp"]["additionalProperties"] is False
    assert schema["$defs"]["integerMatrixOp"]["properties"]["kind"]["enum"] == [
        "swap",
        "negate",
        "add",
    ]


def _mutated_payload(mutator: Callable[[dict[str, object]], None]) -> dict[str, object]:
    payload = deepcopy(_valid_payload())
    mutator(payload)
    return payload


@pytest.mark.parametrize(
    ("mutator", "error", "message"),
    [
        (lambda payload: payload.__setitem__("kind", "rref_modp"), ValueError, "kind"),
        (lambda payload: payload.__setitem__("schema_version", "v0"), ValueError, "schema_version"),
        (lambda payload: payload.pop("input"), ValueError, "missing"),
        (lambda payload: payload.__setitem__("extra", 1), ValueError, "unexpected"),
        (lambda payload: payload.__setitem__("shape", [2]), ValueError, "shape"),
        (lambda payload: payload.__setitem__("shape", [-1, 2]), ValueError, "nonnegative"),
        (lambda payload: payload.__setitem__("shape", [True, 2]), TypeError, "shape"),
        (lambda payload: payload.__setitem__("input", [[1], [2, 3]]), ValueError, "input"),
        (lambda payload: payload.__setitem__("input", [[1.5, 0], [0, 1]]), TypeError, "input"),
        (lambda payload: payload.__setitem__("input", [[True, 0], [0, 1]]), TypeError, "input"),
    ],
)
def test_validate_snf_certificate_rejects_malformed_top_level_and_matrix_fields(
    mutator: Callable[[dict[str, object]], None],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        validate_snf_certificate_record(_mutated_payload(mutator))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input", [[1, 2, 3], [4, 5, 6]], "input"),
        ("diagonal", [[2, 0]], "diagonal"),
        ("left_transform", [[1, 0]], "left_transform"),
        ("right_transform", [[1, 0]], "right_transform"),
    ],
)
def test_validate_snf_certificate_rejects_mismatched_matrix_dimensions(
    field: str,
    value: list[list[int]],
    message: str,
) -> None:
    payload = _valid_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        validate_snf_certificate_record(payload)


@pytest.mark.parametrize(
    ("diagonal", "message"),
    [
        ([[2, 1], [0, 4]], "off-diagonal"),
        ([[-2, 0], [0, 4]], "nonnegative"),
        ([[4, 0], [0, 6]], "divide"),
        ([[0, 0], [0, 3]], "after zero"),
    ],
)
def test_validate_snf_certificate_rejects_invalid_diagonal_form(
    diagonal: list[list[int]],
    message: str,
) -> None:
    payload = _valid_payload()
    payload["diagonal"] = diagonal

    with pytest.raises(ValueError, match=message):
        validate_snf_certificate_record(payload)


@pytest.mark.parametrize(
    ("field", "ops", "error", "message"),
    [
        ("row_ops", [{"kind": "scale", "target": 0}], ValueError, "operation kind"),
        ("row_ops", [{"kind": "swap", "target": 0}], ValueError, "source"),
        ("row_ops", [{"kind": "add", "target": 0, "source": 1}], ValueError, "scalar"),
        ("row_ops", [{"kind": "negate", "target": 2}], IndexError, "row_ops"),
        (
            "row_ops",
            [{"kind": "add", "target": 0, "source": 0, "scalar": 1}],
            ValueError,
            "distinct",
        ),
        ("col_ops", [{"kind": "swap", "target": 0, "source": 2}], IndexError, "col_ops"),
        (
            "col_ops",
            [{"kind": "add", "target": 0, "source": 1, "scalar": True}],
            TypeError,
            "scalar",
        ),
    ],
)
def test_validate_snf_certificate_rejects_invalid_row_and_column_ops(
    field: str,
    ops: list[dict[str, object]],
    error: type[Exception],
    message: str,
) -> None:
    payload = _valid_payload()
    payload[field] = ops

    with pytest.raises(error, match=message):
        validate_snf_certificate_record(payload)
