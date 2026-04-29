import json
from pathlib import Path

import numpy as np
import pytest

from nf_agent.data.hnf_shards import (
    HNFShardSamples,
    generate_hnf_shard,
    integer_row_ops_from_hnf_shard_arrays,
    load_hnf_shard_config,
    make_hnf_grain_dataset,
    write_hnf_shard,
)
from nf_agent.env.hnf_int import is_row_hnf, replay_integer_row_ops


def _config(tmp_path: Path, *, rows: int = 3, cols: int = 3) -> Path:
    config_path = tmp_path / "hnf.yaml"
    config_path.write_text(
        "task: hnf\n"
        "integer_matrix:\n"
        "  family: sparse\n"
        f"  rows: {rows}\n"
        f"  cols: {cols}\n"
        "  density: 0.5\n"
        "  entry_bound: 4\n"
    )
    return config_path


def test_generate_hnf_shard_has_schema_metadata_and_replayable_traces(tmp_path: Path) -> None:
    shard = generate_hnf_shard(config_path=_config(tmp_path), count=5, seed_start=0)

    assert shard["inputs"].shape == (5, 3, 3)
    assert shard["inputs"].dtype == np.int64
    assert shard["finals"].shape == (5, 3, 3)
    assert shard["finals"].dtype == np.int64
    assert shard["op_kind"].dtype == np.int8
    assert shard["op_target"].dtype == np.int64
    assert shard["op_source"].dtype == np.int64
    assert shard["op_scalar_id"].dtype == np.int64
    assert shard["op_scalar_value"].dtype == np.int64
    assert shard["op_mask"].dtype == np.bool_
    assert shard["op_kind"].shape[0] == 5
    assert shard["op_kind"].shape == shard["op_target"].shape == shard["op_mask"].shape
    assert shard["scalar_vocab"].ndim == 1
    assert np.array_equal(shard["scalar_vocab"], np.unique(shard["scalar_vocab"]))

    metadata = json.loads(str(shard["metadata_json"]))
    assert metadata["schema_version"] == "hnf-teacher-trajectory-npz-v0.8"
    assert metadata["op_encoding"] == {"pad": 0, "swap": 1, "negate": 2, "add": 3}
    assert metadata["padding_value"] == -1
    assert metadata["count"] == 5
    assert metadata["shape"] == {"rows": 3, "cols": 3}

    for sample_index in range(5):
        ops = integer_row_ops_from_hnf_shard_arrays(shard, sample_index)
        replayed = replay_integer_row_ops(shard["inputs"][sample_index].tolist(), ops)
        final = shard["finals"][sample_index].tolist()
        assert replayed == final
        assert is_row_hnf(final)


def test_write_hnf_shard_loads_as_training_samples(tmp_path: Path) -> None:
    shard_path = tmp_path / "hnf.npz"
    write_hnf_shard(_config(tmp_path), count=4, seed_start=7, out_path=shard_path)

    samples = HNFShardSamples(shard_path)
    example = samples[0]

    assert len(samples) == 4
    assert samples.rows == 3
    assert samples.cols == 3
    assert samples.max_ops >= 1
    assert samples.scalar_vocab_size == len(samples.scalar_vocab)
    assert example["inputs"].dtype == np.float32
    assert example["op_kind"].dtype == np.int32
    assert example["op_mask"].dtype == np.bool_
    assert np.max(np.abs(example["inputs"])) <= 1.0

    batches = list(make_hnf_grain_dataset(shard_path, batch_size=2, seed=0))
    assert batches[0]["inputs"].shape == (2, 3, 3)


@pytest.mark.parametrize(
    ("config_text", "message"),
    [
        (
            "task: rref\ninteger_matrix:\n  family: sparse\n  rows: 2\n  cols: 2\n",
            "unsupported task",
        ),
        (
            "task: hnf\ninteger_matrix:\n  family: dense\n  rows: 2\n  cols: 2\n",
            "unsupported integer matrix family",
        ),
        (
            "task: hnf\ninteger_matrix:\n  family: sparse\n  rows: 0\n  cols: 2\n  density: 0.5\n",
            "rows must be positive",
        ),
        (
            "task: hnf\ninteger_matrix:\n  family: sparse\n  rows: 2\n  cols: 2\n  density: 2\n",
            "density must lie in \\[0, 1\\]",
        ),
        (
            "task: hnf\ninteger_matrix:\n  family: sparse\n  rows: 2\n  cols: 2\n"
            "  density: 0.5\n  entry_bound: 0\n",
            "entry_bound must be positive",
        ),
    ],
)
def test_invalid_hnf_configs_fail_explicitly(
    tmp_path: Path,
    config_text: str,
    message: str,
) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(config_text)

    with pytest.raises(ValueError, match=message):
        load_hnf_shard_config(config_path)


def test_hnf_shard_loader_rejects_scalar_vocab_mismatch(tmp_path: Path) -> None:
    shard_path = tmp_path / "hnf.npz"
    write_hnf_shard(_config(tmp_path), count=4, seed_start=0, out_path=shard_path)

    with np.load(shard_path, allow_pickle=False) as shard:
        arrays = {key: np.asarray(shard[key]) for key in shard.files}
    active_add = np.argwhere(arrays["op_kind"] == 3)
    if active_add.size == 0:
        pytest.skip("teacher did not emit an add op for this tiny deterministic shard")
    sample_index, op_index = active_add[0]
    arrays["op_scalar_value"][sample_index, op_index] += 123
    bad_path = tmp_path / "bad_hnf.npz"
    np.savez(bad_path, **arrays)

    with pytest.raises(ValueError, match="op_scalar_value entries must match scalar_vocab"):
        HNFShardSamples(bad_path)
