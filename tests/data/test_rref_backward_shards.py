import json
from pathlib import Path

import numpy as np
import pytest

from nf_agent.data.rref_backward_shards import (
    generate_rref_backward_shard,
    load_rref_backward_shard,
    load_rref_backward_shard_config,
    row_ops_from_backward_shard_arrays,
    write_rref_backward_shard,
)
from nf_agent.env.rref_modp import is_rref_modp, replay_row_ops

TRACKED_CONFIG = Path("configs/rref_backward_4x4_mod101.yaml")


def _write_config(
    tmp_path: Path,
    *,
    modulus: int = 101,
    rows: int = 4,
    cols: int = 4,
    max_backward_ops: int = 6,
) -> Path:
    config_path = tmp_path / "rref_backward.yaml"
    config_path.write_text(
        "task: rref_backward_state_shards\n"
        "field:\n"
        f"  modulus: {modulus}\n"
        "matrix:\n"
        "  family: dense\n"
        f"  rows: {rows}\n"
        f"  cols: {cols}\n"
        "backward_trace:\n"
        "  schema: rref-backward-trace-npz-v1\n"
        "  format: npz\n"
        f"  max_backward_ops: {max_backward_ops}\n"
        "  require_exact_replay: true\n"
    )
    return config_path


def test_generate_rref_backward_shard_has_schema_shapes_and_dtypes(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, rows=3, cols=5, max_backward_ops=7)

    shard = generate_rref_backward_shard(config_path=config_path, count=4, seed_start=10)

    assert shard["inputs"].shape == (4, 3, 5)
    assert shard["inputs"].dtype == np.int64
    assert shard["finals"].shape == (4, 3, 5)
    assert shard["finals"].dtype == np.int64
    assert shard["pivots"].shape == (4, 3, 2)
    assert shard["pivots"].dtype == np.int64
    assert shard["ops"].shape == (4, 7, 4)
    assert shard["ops"].dtype == np.int64
    assert shard["op_mask"].shape == (4, 7)
    assert shard["op_mask"].dtype == np.bool_

    metadata = json.loads(str(shard["metadata_json"]))
    assert metadata["schema_version"] == "rref-backward-trace-npz-v1"
    assert metadata["count"] == 4
    assert metadata["seed_start"] == 10
    assert metadata["shape"] == {"rows": 3, "cols": 5}
    assert metadata["max_pivots"] == 3
    assert metadata["max_ops"] == 7


def test_tracked_rref_backward_smoke_config_is_loadable() -> None:
    config = load_rref_backward_shard_config(TRACKED_CONFIG)

    assert config.modulus == 101
    assert config.rows == 4
    assert config.cols == 4
    assert config.max_backward_ops == 8


def test_backward_traces_replay_inputs_to_claimed_rref_finals(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, max_backward_ops=8)

    shard = generate_rref_backward_shard(config_path=config_path, count=6, seed_start=0)

    for sample_index in range(6):
        ops = row_ops_from_backward_shard_arrays(shard, sample_index)
        replayed = replay_row_ops(shard["inputs"][sample_index].tolist(), ops, 101)
        final = shard["finals"][sample_index].tolist()

        assert replayed == final
        assert is_rref_modp(final, 101)
        assert len(ops) == int(shard["op_mask"][sample_index].sum())


def test_backward_shard_generation_is_deterministic(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    left = generate_rref_backward_shard(config_path=config_path, count=3, seed_start=5)
    right = generate_rref_backward_shard(config_path=config_path, count=3, seed_start=5)

    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_backward_config_rejects_non_prime_modulus(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, modulus=100)

    with pytest.raises(ValueError, match="modulus p must be prime"):
        load_rref_backward_shard_config(config_path)


def test_backward_config_accepts_zarr_format_and_rejects_unknown_format(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    text = config_path.read_text().replace("  format: npz\n", "  format: zarr\n")
    config_path.write_text(text)

    assert load_rref_backward_shard_config(config_path).modulus == 101

    config_path.write_text(text.replace("  format: zarr\n", "  format: zip\n"))
    with pytest.raises(ValueError, match="backward_trace.format must be 'npz' or 'zarr'"):
        load_rref_backward_shard_config(config_path)


def test_load_rref_backward_shard_rejects_bad_replay(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    good_path = tmp_path / "good.npz"
    bad_path = tmp_path / "bad.npz"

    write_rref_backward_shard(config_path=config_path, count=2, seed_start=0, out_path=good_path)
    with np.load(good_path, allow_pickle=False) as shard:
        arrays = {key: np.asarray(shard[key]) for key in shard.files}
    arrays["finals"] = arrays["finals"].copy()
    arrays["finals"][0, 0, 0] = (int(arrays["finals"][0, 0, 0]) + 1) % 101
    np.savez(bad_path, **arrays)

    with pytest.raises(ValueError, match="does not replay"):
        load_rref_backward_shard(bad_path)


def test_write_rref_backward_shard_creates_valid_npz(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    out_path = tmp_path / "backward.npz"

    write_rref_backward_shard(config_path=config_path, count=3, seed_start=2, out_path=out_path)
    arrays, metadata = load_rref_backward_shard(out_path)

    assert arrays["inputs"].shape == (3, 4, 4)
    assert metadata["schema_version"] == "rref-backward-trace-npz-v1"
    assert metadata["count"] == 3
    assert metadata["seed_start"] == 2
