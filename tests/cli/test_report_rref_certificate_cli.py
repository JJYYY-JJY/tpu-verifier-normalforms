import json
from pathlib import Path

from click.testing import CliRunner

from nf_agent.cli import main
from nf_agent.env.rref_modp import RowOp, is_rref_modp, replay_row_ops

FIXTURE = Path("tests/fixtures/rref_certificate_2x3_mod5_seed0.json").resolve()


def _row_op_from_dict(payload: dict[str, int | str]) -> RowOp:
    return RowOp(
        kind=payload["kind"],  # type: ignore[arg-type]
        target=payload["target"],  # type: ignore[arg-type]
        source=payload.get("source"),  # type: ignore[arg-type]
        scalar=payload.get("scalar"),  # type: ignore[arg-type]
    )


def test_report_rref_certificate_cli_emits_replayable_schema() -> None:
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "report",
            "rref-certificate",
            "--rows",
            "2",
            "--cols",
            "3",
            "--p",
            "5",
            "--seed",
            "0",
            "--teacher",
            "leftmost",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == json.loads(FIXTURE.read_text())
    assert set(payload) == {"final", "input", "kind", "modulus", "ops", "pivots", "shape"}
    assert payload["kind"] == "rref_modp"
    assert payload["modulus"] == 5
    assert payload["shape"] == [2, 3]
    assert len(payload["input"]) == 2
    assert all(len(row) == 3 for row in payload["input"])

    ops = [_row_op_from_dict(op) for op in payload["ops"]]
    assert replay_row_ops(payload["input"], ops, payload["modulus"]) == payload["final"]
    assert is_rref_modp(payload["final"], payload["modulus"])
    assert payload["pivots"] == [{"row": 0, "col": 0}, {"row": 1, "col": 1}]
