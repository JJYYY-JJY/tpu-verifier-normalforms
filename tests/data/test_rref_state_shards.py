import json
from pathlib import Path

import numpy as np
import pytest

from nf_agent.data.rref_backward_shards import write_rref_backward_shard
from nf_agent.data.rref_state_shards import (
    RREFStateActionSamples,
    generate_rref_state_shard,
    load_rref_state_shard,
    write_rref_state_shard,
)
from nf_agent.env.rref_modp import RowOp, is_rref_modp, replay_row_ops


def _write_backward_config(tmp_path: Path, *, max_backward_ops: int = 5) -> Path:
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


def _write_backward_shard(tmp_path: Path, *, count: int = 3, max_ops: int = 5) -> Path:
    config_path = _write_backward_config(tmp_path, max_backward_ops=max_ops)
    trace_path = tmp_path / "backward.npz"
    write_rref_backward_shard(
        config_path=config_path,
        count=count,
        seed_start=0,
        out_path=trace_path,
    )
    return trace_path


def _row_op_from_state_arrays(shard: dict[str, np.ndarray], index: int) -> RowOp:
    kind = int(shard["action_kind"][index])
    target = int(shard["action_target"][index])
    source = int(shard["action_source"][index])
    scalar = int(shard["action_scalar"][index])
    if kind == 1:
        return RowOp.swap(target, source)
    if kind == 2:
        return RowOp.scale(target, scalar)
    if kind == 3:
        return RowOp.add(target, source, scalar)
    raise AssertionError(f"flat sample {index} is not a row op")


def test_generate_rref_state_shard_has_flat_and_trace_arrays(tmp_path: Path) -> None:
    trace_path = _write_backward_shard(tmp_path, count=3, max_ops=5)

    shard = generate_rref_state_shard(trace_path)
    metadata = json.loads(str(shard["metadata_json"]))

    assert metadata["schema_version"] == "rref-state-action-npz-v1"
    assert metadata["source_schema_version"] == "rref-backward-trace-npz-v1"
    assert metadata["shape"] == {"rows": 4, "cols": 4}
    assert metadata["modulus"] == 101
    assert metadata["trace_count"] == 3
    assert metadata["flat_count"] == int(metadata["trace_count"] * (metadata["max_ops"] + 1))
    assert metadata["includes_trace_tensors"] is True

    flat_count = int(metadata["flat_count"])
    max_steps = int(metadata["max_ops"] + 1)
    assert shard["states"].shape == (flat_count, 4, 4)
    assert shard["states"].dtype == np.int64
    assert shard["action_kind"].shape == (flat_count,)
    assert shard["action_kind"].dtype == np.int8
    assert shard["action_target"].shape == (flat_count,)
    assert shard["action_target"].dtype == np.int64
    assert shard["action_source"].shape == (flat_count,)
    assert shard["action_source"].dtype == np.int64
    assert shard["action_scalar"].shape == (flat_count,)
    assert shard["action_scalar"].dtype == np.int64
    assert shard["stop_label"].shape == (flat_count,)
    assert shard["stop_label"].dtype == np.bool_
    assert shard["legal_kind_mask"].shape == (flat_count, 4)
    assert shard["legal_kind_mask"].dtype == np.bool_
    assert shard["legal_target_mask"].shape == (flat_count, 4)
    assert shard["legal_source_mask"].shape == (flat_count, 4)
    assert shard["legal_target_source_mask"].shape == (flat_count, 4, 4)
    assert shard["legal_scalar_mask"].shape == (flat_count, 101)

    assert shard["trace_states"].shape == (3, max_steps, 4, 4)
    assert shard["trace_states"].dtype == np.int64
    assert shard["trace_action_kind"].shape == (3, max_steps)
    assert shard["trace_action_kind"].dtype == np.int8
    assert shard["trace_action_target"].shape == (3, max_steps)
    assert shard["trace_action_source"].shape == (3, max_steps)
    assert shard["trace_action_scalar"].shape == (3, max_steps)
    assert shard["trace_stop_label"].shape == (3, max_steps)
    assert shard["trace_stop_label"].dtype == np.bool_
    assert shard["trace_step_mask"].shape == (3, max_steps)
    assert shard["trace_step_mask"].dtype == np.bool_


def test_flat_and_trace_counts_match_backward_ops_plus_stop(tmp_path: Path) -> None:
    trace_path = _write_backward_shard(tmp_path, count=4, max_ops=6)

    with np.load(trace_path, allow_pickle=False) as source:
        expected_flat_count = int(source["op_mask"].sum()) + int(source["op_mask"].shape[0])
    shard = generate_rref_state_shard(trace_path)
    metadata = json.loads(str(shard["metadata_json"]))

    assert metadata["flat_count"] == expected_flat_count
    assert len(shard["states"]) == expected_flat_count
    assert int(shard["stop_label"].sum()) == 4
    assert int(shard["trace_stop_label"].sum()) == 4


def test_state_actions_exactly_replay_each_trace(tmp_path: Path) -> None:
    trace_path = _write_backward_shard(tmp_path, count=3, max_ops=5)

    shard = generate_rref_state_shard(trace_path)

    flat_index = 0
    for trace_index in range(shard["trace_states"].shape[0]):
        active_steps = int(shard["trace_step_mask"][trace_index].sum())
        stop_step = active_steps - 1
        for step in range(stop_step):
            op = _row_op_from_state_arrays(shard, flat_index)
            replayed = replay_row_ops(shard["states"][flat_index].tolist(), [op], 101)
            expected = shard["trace_states"][trace_index, step + 1].tolist()

            assert replayed == expected
            assert np.array_equal(
                shard["states"][flat_index],
                shard["trace_states"][trace_index, step],
            )
            assert not bool(shard["stop_label"][flat_index])
            flat_index += 1

        final_state = shard["states"][flat_index].tolist()
        assert bool(shard["stop_label"][flat_index])
        assert int(shard["action_kind"][flat_index]) == 0
        assert final_state == shard["trace_states"][trace_index, stop_step].tolist()
        assert is_rref_modp(final_state, 101)
        flat_index += 1

    assert flat_index == len(shard["states"])


def test_trace_final_stop_state_matches_source_final_and_is_rref(tmp_path: Path) -> None:
    trace_path = _write_backward_shard(tmp_path, count=3, max_ops=5)

    shard = generate_rref_state_shard(trace_path)
    with np.load(trace_path, allow_pickle=False) as source:
        finals = np.asarray(source["finals"])

    for trace_index in range(3):
        stop_step = int(shard["trace_step_mask"][trace_index].sum()) - 1
        final_state = shard["trace_states"][trace_index, stop_step].tolist()

        assert final_state == finals[trace_index].tolist()
        assert is_rref_modp(final_state, 101)


def test_load_rref_state_shard_rejects_corrupted_replay(tmp_path: Path) -> None:
    trace_path = _write_backward_shard(tmp_path, count=2, max_ops=4)
    good_path = tmp_path / "state_good.npz"
    bad_path = tmp_path / "state_bad.npz"
    write_rref_state_shard(trace_path, good_path)

    with np.load(good_path, allow_pickle=False) as shard:
        arrays = {key: np.asarray(shard[key]) for key in shard.files}
    arrays["trace_states"] = arrays["trace_states"].copy()
    arrays["trace_states"][0, 1, 0, 0] = (int(arrays["trace_states"][0, 1, 0, 0]) + 1) % 101
    np.savez(bad_path, **arrays)

    with pytest.raises(ValueError, match="replay|flat|trace"):
        load_rref_state_shard(bad_path)


def test_state_action_samples_returns_normalized_training_example(tmp_path: Path) -> None:
    trace_path = _write_backward_shard(tmp_path, count=2, max_ops=4)
    out_path = tmp_path / "state.npz"
    write_rref_state_shard(trace_path, out_path)

    samples = RREFStateActionSamples(out_path)
    example = samples[0]

    assert len(samples) == samples.metadata["flat_count"]
    assert example["state"].shape == (4, 4)
    assert example["state"].dtype == np.float32
    assert np.all(example["state"] >= 0.0)
    assert np.all(example["state"] <= 1.0)
    assert example["action_kind"].dtype == np.int32
    assert example["action_target"].dtype == np.int32
    assert example["action_source"].dtype == np.int32
    assert example["action_scalar"].dtype == np.int32
    assert example["stop_label"].dtype == np.bool_
    assert example["legal_kind_mask"].dtype == np.bool_
    assert example["legal_target_mask"].shape == (4,)
    assert example["legal_source_mask"].shape == (4,)
    assert example["legal_target_source_mask"].shape == (4, 4)
    assert example["legal_scalar_mask"].shape == (101,)
