import json
from pathlib import Path

import numpy as np
import pytest

from nf_agent.data.rref_backward_shards import load_rref_backward_shard, write_rref_backward_shard
from nf_agent.data.rref_state_shards import (
    RREFStateActionSamples,
    load_rref_state_shard,
    write_rref_state_shard,
)


def _write_backward_config(tmp_path: Path, *, modulus: int = 101, max_ops: int = 4) -> Path:
    config_path = tmp_path / "rref_backward.yaml"
    config_path.write_text(
        "task: rref_backward_state_shards\n"
        "field:\n"
        f"  modulus: {modulus}\n"
        "matrix:\n"
        "  family: dense\n"
        "  rows: 4\n"
        "  cols: 4\n"
        "backward_trace:\n"
        "  schema: rref-backward-trace-npz-v1\n"
        "  format: npz\n"
        f"  max_backward_ops: {max_ops}\n"
        "  require_exact_replay: true\n"
    )
    return config_path


def _assert_same_arrays(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    assert set(left) == set(right)
    for key in left:
        assert left[key].dtype == right[key].dtype
        assert np.array_equal(left[key], right[key]), key


def test_rref_backward_zarr_roundtrips_like_npz(tmp_path: Path) -> None:
    config = _write_backward_config(tmp_path)
    npz_path = tmp_path / "backward.npz"
    zarr_path = tmp_path / "backward.zarr"

    write_rref_backward_shard(config, count=3, seed_start=7, out_path=npz_path)
    write_rref_backward_shard(config, count=3, seed_start=7, out_path=zarr_path)

    npz_arrays, npz_metadata = load_rref_backward_shard(npz_path)
    zarr_arrays, zarr_metadata = load_rref_backward_shard(zarr_path)

    _assert_same_arrays(npz_arrays, zarr_arrays)
    assert zarr_metadata["schema_version"] == "rref-backward-trace-npz-v1"
    assert zarr_metadata["config"]["backward_trace"]["format"] == "zarr"
    assert npz_metadata["config"]["backward_trace"]["format"] == "npz"


def test_rref_state_zarr_samples_match_npz_samples(tmp_path: Path) -> None:
    config = _write_backward_config(tmp_path)
    backward_npz = tmp_path / "backward.npz"
    backward_zarr = tmp_path / "backward.zarr"
    state_npz = tmp_path / "state.npz"
    state_zarr = tmp_path / "state.zarr"

    write_rref_backward_shard(config, count=2, seed_start=0, out_path=backward_npz)
    write_rref_backward_shard(config, count=2, seed_start=0, out_path=backward_zarr)
    write_rref_state_shard(backward_npz, state_npz)
    write_rref_state_shard(backward_zarr, state_zarr)

    npz_arrays, npz_metadata = load_rref_state_shard(state_npz)
    zarr_arrays, zarr_metadata = load_rref_state_shard(state_zarr)
    _assert_same_arrays(npz_arrays, zarr_arrays)
    assert zarr_metadata["format"] == "zarr"
    assert zarr_metadata["source_path"].endswith("backward.zarr")

    npz_samples = RREFStateActionSamples(state_npz)
    zarr_samples = RREFStateActionSamples(state_zarr)
    assert len(npz_samples) == len(zarr_samples)
    assert npz_samples.rows == zarr_samples.rows == 4
    assert npz_samples.cols == zarr_samples.cols == 4
    assert npz_samples.modulus == zarr_samples.modulus == 101
    for key, value in npz_samples[0].items():
        assert np.array_equal(value, zarr_samples[0][key]), key


def test_rref_zarr_loader_rejects_non_prime_metadata(tmp_path: Path) -> None:
    config = _write_backward_config(tmp_path)
    zarr_path = tmp_path / "backward.zarr"
    write_rref_backward_shard(config, count=1, seed_start=0, out_path=zarr_path)

    import zarr

    group = zarr.open_group(str(zarr_path), mode="a")
    raw = np.asarray(group["metadata_json"]).item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    metadata = json.loads(str(raw))
    metadata["config"]["field"]["modulus"] = 100
    del group["metadata_json"]
    group.create_array(
        "metadata_json",
        data=np.asarray(json.dumps(metadata, sort_keys=True).encode("utf-8")),
    )

    with pytest.raises(ValueError, match="prime"):
        load_rref_backward_shard(zarr_path)

