from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nf_agent.benchmarks.rref_benchmark import (
    RREFBenchmarkConfig,
    matrix_density_modp,
    row_op_density_profile,
    run_rref_benchmark,
)
from nf_agent.data.rref_shards import write_rref_shard
from nf_agent.env.rref_modp import RowOp, replay_row_ops
from nf_agent.train import TrainConfig, train_rref_pivot

CONFIG = Path("configs/rref_8x8_mod101.yaml").resolve()


def test_matrix_density_modp_counts_normalized_nonzeros() -> None:
    assert matrix_density_modp([[0, 0], [0, 0]], 5) == 0.0
    assert matrix_density_modp([[0, 6], [10, -2]], 5) == 0.5


def test_row_op_density_profile_replays_each_state() -> None:
    matrix = [[2, 0], [0, 1]]
    ops = [RowOp.scale(0, 3)]

    profile = row_op_density_profile(matrix, ops, 5)

    assert len(profile.densities) == len(ops) + 1
    assert len(profile.states) == len(ops) + 1
    assert profile.states[-1] == replay_row_ops(matrix, ops, 5)


def test_generated_benchmark_runs_leftmost_with_exact_verification() -> None:
    result = run_rref_benchmark(
        RREFBenchmarkConfig(
            source="generated",
            rows=4,
            cols=4,
            modulus=101,
            family="dense",
            count=4,
            seed_start=0,
        )
    )

    leftmost = result["policies"]["leftmost"]
    assert result["status"] == "ok"
    assert result["source"] == "generated"
    assert result["count"] == 4
    assert leftmost["aggregate"]["success_rate"] == 1.0
    assert leftmost["aggregate"]["status_counts"] == {"success": 4}
    assert len(leftmost["samples"]) == 4
    assert all(sample["replay_ok"] for sample in leftmost["samples"])
    assert all(sample["final_is_rref"] for sample in leftmost["samples"])
    assert "neural" not in result["policies"]


def test_shard_benchmark_reads_inputs_and_respects_count_limit(tmp_path: Path) -> None:
    shard_path = tmp_path / "rref_bench.npz"
    write_rref_shard(config_path=CONFIG, count=5, seed_start=3, out_path=shard_path)

    result = run_rref_benchmark(
        RREFBenchmarkConfig(source="shard", data_path=shard_path, count=2)
    )

    samples = result["policies"]["leftmost"]["samples"]
    assert result["source"] == "shard"
    assert result["count"] == 2
    assert [sample["sample_index"] for sample in samples] == [0, 1]
    assert all(sample["replay_ok"] for sample in samples)
    assert all(sample["final_is_rref"] for sample in samples)


def test_shard_neural_benchmark_reports_status_counts_without_teacher_fallback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    shard_path = tmp_path / "rref_bench_neural.npz"
    ckpt_dir = tmp_path / "ckpt"
    write_rref_shard(config_path=CONFIG, count=8, seed_start=0, out_path=shard_path)
    train_rref_pivot(
        TrainConfig(
            data_path=shard_path,
            steps=2,
            batch_size=4,
            out_dir=ckpt_dir,
            hidden_sizes=(32,),
        )
    )

    def fail_teacher_call(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("neural benchmark must not call the leftmost teacher")

    monkeypatch.setattr("nf_agent.teachers.leftmost.LeftmostRREFTeacher.solve", fail_teacher_call)

    result = run_rref_benchmark(
        RREFBenchmarkConfig(
            source="shard",
            data_path=shard_path,
            count=2,
            checkpoint_dir=ckpt_dir,
            max_steps=2,
            hidden_sizes=(32,),
        )
    )

    neural = result["policies"]["neural"]
    assert sum(neural["aggregate"]["status_counts"].values()) == 2
    assert len(neural["samples"]) == 2
    assert all(
        sample["status"] in {"success", "max_steps_exceeded"} for sample in neural["samples"]
    )
    assert all(sample["success"] is (sample["status"] == "success") for sample in neural["samples"])
    assert all(isinstance(sample["replay_ok"], bool) for sample in neural["samples"])
    assert all("masked_action_count" in sample for sample in neural["samples"])


def test_generated_neural_requires_matching_model_metadata(tmp_path: Path) -> None:
    shard_path = tmp_path / "rref_model_data.npz"
    write_rref_shard(config_path=CONFIG, count=2, seed_start=0, out_path=shard_path)

    try:
        run_rref_benchmark(
            RREFBenchmarkConfig(
                source="generated",
                rows=4,
                cols=4,
                count=1,
                checkpoint_dir=tmp_path / "ckpt",
                model_data_path=shard_path,
            )
        )
    except ValueError as exc:
        assert "model data shape/modulus must match generated benchmark config" in str(exc)
    else:
        raise AssertionError("expected generated/model metadata mismatch to fail")


def test_shard_count_defaults_to_all_inputs(tmp_path: Path) -> None:
    shard_path = tmp_path / "rref_bench_all.npz"
    write_rref_shard(config_path=CONFIG, count=3, seed_start=0, out_path=shard_path)

    result = run_rref_benchmark(RREFBenchmarkConfig(source="shard", data_path=shard_path))

    assert result["count"] == 3
    with np.load(shard_path, allow_pickle=False) as shard:
        assert result["count"] == int(shard["inputs"].shape[0])
