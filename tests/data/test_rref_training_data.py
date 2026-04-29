import json
from pathlib import Path

import numpy as np
import pytest

from nf_agent.data.rref_shards import (
    RREFShardSamples,
    generate_rref_shard,
    make_rref_grain_dataset,
    write_rref_shard,
)

CONFIG = Path("configs/rref_8x8_mod101.yaml")


def _write_temp_shard(tmp_path: Path, count: int = 5) -> Path:
    out_path = tmp_path / "rref_train.npz"
    write_rref_shard(config_path=CONFIG, count=count, seed_start=0, out_path=out_path)
    return out_path


def test_rref_shard_samples_loads_normalized_training_examples(tmp_path: Path) -> None:
    shard_path = _write_temp_shard(tmp_path)

    samples = RREFShardSamples(shard_path)
    sample = samples[0]

    assert len(samples) == 5
    assert samples.metadata["schema_version"] == "rref-teacher-trajectory-npz-v0.2"
    assert samples.metadata["config"]["field"]["modulus"] == 101
    assert sample["inputs"].shape == (8, 8)
    assert sample["inputs"].dtype == np.float32
    assert np.min(sample["inputs"]) >= 0.0
    assert np.max(sample["inputs"]) <= 1.0
    assert sample["pivot_active"].shape == (8,)
    assert sample["pivot_active"].dtype == np.float32
    assert sample["pivot_cols"].dtype == np.int32
    assert sample["op_kind"].shape == (72,)
    assert sample["op_kind"].dtype == np.int32
    assert sample["op_source_mask"].dtype == np.bool_
    assert sample["op_scalar_mask"].dtype == np.bool_


def test_make_rref_grain_dataset_batches_training_examples(tmp_path: Path) -> None:
    shard_path = _write_temp_shard(tmp_path, count=6)

    dataset = make_rref_grain_dataset(shard_path, batch_size=4, seed=7)
    batch = next(iter(dataset))

    assert batch["inputs"].shape == (4, 8, 8)
    assert batch["pivot_cols"].shape == (4, 8)
    assert batch["op_kind"].shape == (4, 72)
    assert batch["op_scalar"].shape == (4, 72)
    assert batch["inputs"].dtype == np.float32


def test_rref_shard_samples_maps_negative_scalars_to_modulus_classes(tmp_path: Path) -> None:
    shard = generate_rref_shard(config_path=CONFIG, count=1, seed_start=0)
    shard["op_mask"][0, 0] = True
    shard["op_kind"][0, 0] = 3
    shard["op_target"][0, 0] = 0
    shard["op_source"][0, 0] = 1
    shard["op_scalar"][0, 0] = -3
    shard_path = tmp_path / "negative_scalar.npz"
    np.savez(shard_path, **shard)

    sample = RREFShardSamples(shard_path)[0]

    assert sample["op_scalar"][0] == 98
    assert sample["op_scalar_mask"][0]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_array", "missing required array"),
        ("schema_version", "unsupported RREF shard schema_version"),
        ("shape", "inputs must have shape"),
        ("padding", "pivot_cols padding must be -1"),
    ],
)
def test_rref_shard_samples_rejects_malformed_npz(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    shard = generate_rref_shard(config_path=CONFIG, count=2, seed_start=0)
    if mutation == "missing_array":
        shard.pop("op_kind")
    elif mutation == "schema_version":
        metadata = json.loads(str(shard["metadata_json"]))
        metadata["schema_version"] = "old"
        shard["metadata_json"] = np.asarray(json.dumps(metadata))
    elif mutation == "shape":
        shard["inputs"] = shard["inputs"][:, :7, :]
    elif mutation == "padding":
        shard["pivot_mask"][0, -1] = False
        shard["pivot_rows"][0, -1] = -1
        shard["pivot_cols"][0, -1] = 0

    shard_path = tmp_path / "bad.npz"
    np.savez(shard_path, **shard)

    with pytest.raises(ValueError, match=message):
        RREFShardSamples(shard_path)


def test_rref_shard_samples_rejects_non_npz_path(tmp_path: Path) -> None:
    path = tmp_path / "shard.zip"
    path.write_bytes(b"not an npz")

    with pytest.raises(ValueError, match="data path must end with .npz"):
        RREFShardSamples(path)


def test_make_rref_grain_dataset_rejects_invalid_batch_size(tmp_path: Path) -> None:
    shard_path = _write_temp_shard(tmp_path)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        make_rref_grain_dataset(shard_path, batch_size=0, seed=0)
