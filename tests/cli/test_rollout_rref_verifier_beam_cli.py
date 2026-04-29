import json
from pathlib import Path

from click.testing import CliRunner

from nf_agent.cli import main
from nf_agent.data.rref_backward_shards import write_rref_backward_shard
from nf_agent.data.rref_state_shards import write_rref_state_shard
from nf_agent.train import RREFMatrixFormerTrainConfig, train_rref_matrixformer


def _write_backward_config(tmp_path: Path) -> Path:
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
        "  max_backward_ops: 4\n"
        "  require_exact_replay: true\n"
    )
    return config_path


def _state_shard(tmp_path: Path) -> Path:
    trace_path = tmp_path / "backward.zarr"
    write_rref_backward_shard(
        _write_backward_config(tmp_path),
        count=4,
        seed_start=0,
        out_path=trace_path,
    )
    state_path = tmp_path / "state.zarr"
    write_rref_state_shard(trace_path, state_path)
    return state_path


def test_rollout_rref_verifier_beam_cli_runs_and_emits_compact_fields(tmp_path: Path) -> None:
    shard_path = _state_shard(tmp_path)
    ckpt_dir = tmp_path / "ckpt"
    train_rref_matrixformer(
        RREFMatrixFormerTrainConfig(
            data_path=shard_path,
            steps=2,
            batch_size=4,
            out_dir=ckpt_dir,
            row_embedding_dim=8,
            col_embedding_dim=8,
            hidden_dim=32,
            layers=1,
            num_heads=1,
        )
    )

    result = CliRunner().invoke(
        main,
        [
            "rollout",
            "rref-verifier-beam",
            "--data",
            str(shard_path),
            "--checkpoint",
            str(ckpt_dir),
            "--sample-index",
            "0",
            "--max-steps",
            "4",
            "--beam-width",
            "4",
            "--batch-size",
            "auto",
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
    assert payload["status"] in {"success", "max_steps_exceeded"}
    assert payload["success"] is (payload["status"] == "success")
    assert payload["beam_width"] == 4
    assert payload["replay_ok"] is True
    assert payload["expanded_count"] >= 0
    assert payload["pruned_count"] >= 0
    assert payload["device_batch_size"] >= 1
    assert payload["checkpoint_step"] == 2
