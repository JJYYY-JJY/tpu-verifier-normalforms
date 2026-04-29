import json
from pathlib import Path

from click.testing import CliRunner

from nf_agent.cli import main
from nf_agent.data.rref_backward_shards import write_rref_backward_shard
from nf_agent.data.rref_state_shards import write_rref_state_shard


def _write_backward_config(tmp_path: Path, *, max_backward_ops: int = 4) -> Path:
    config_path = tmp_path / "rref_backward.yaml"
    config_path.write_text(
        "task: rref_backward_state_shards\n"
        "field:\n"
        "  modulus: 101\n"
        "matrix:\n"
        "  family: dense\n"
        "  rows: 4\n"
        "  cols: 4\n"
        "backward_trace:\n"
        "  schema: rref-backward-trace-npz-v1\n"
        "  format: npz\n"
        f"  max_backward_ops: {max_backward_ops}\n"
        "  require_exact_replay: true\n"
    )
    return config_path


def _state_shard(tmp_path: Path) -> Path:
    trace_path = tmp_path / "backward.npz"
    write_rref_backward_shard(
        config_path=_write_backward_config(tmp_path),
        count=4,
        seed_start=0,
        out_path=trace_path,
    )
    state_path = tmp_path / "state.npz"
    write_rref_state_shard(trace_path, state_path)
    return state_path


def test_train_status_lists_rref_matrixformer_command() -> None:
    result = CliRunner().invoke(main, ["train", "status"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "rref-matrixformer" in payload["commands"]


def test_train_rref_matrixformer_cli_runs_and_emits_json(tmp_path: Path) -> None:
    shard_path = _state_shard(tmp_path)
    out_dir = tmp_path / "ckpt"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "train",
            "rref-matrixformer",
            "--data",
            str(shard_path),
            "--steps",
            "2",
            "--batch-size",
            "4",
            "--learning-rate",
            "0.001",
            "--seed",
            "0",
            "--out",
            str(out_dir),
            "--row-embedding-dim",
            "8",
            "--col-embedding-dim",
            "8",
            "--hidden-dim",
            "32",
            "--layers",
            "1",
            "--num-heads",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["model"] == "rref-matrixformer"
    assert payload["final_step"] == 2
    assert payload["latest_step"] == 2
    assert Path(payload["checkpoint_dir"]).exists()
    assert payload["data_schema_version"] == "rref-state-action-npz-v1"


def test_train_rref_matrixformer_cli_rejects_invalid_args(tmp_path: Path) -> None:
    shard_path = _state_shard(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "train",
            "rref-matrixformer",
            "--data",
            str(shard_path),
            "--steps",
            "0",
            "--batch-size",
            "2",
            "--out",
            str(tmp_path / "ckpt"),
        ],
    )

    assert result.exit_code != 0
    assert "steps must be positive" in result.output
