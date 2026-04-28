import json
from pathlib import Path

import numpy as np
import pytest

from nf_agent.data.rref_shards import (
    generate_rref_shard,
    load_rref_shard_config,
    row_ops_from_shard_arrays,
    write_rref_shard,
)
from nf_agent.env.rref_modp import is_rref_modp, replay_row_ops

CONFIG = Path("configs/rref_8x8_mod101.yaml")


def test_generate_rref_shard_has_fixed_shapes_and_dtypes() -> None:
    shard = generate_rref_shard(config_path=CONFIG, count=5, seed_start=0)

    assert shard["inputs"].shape == (5, 8, 8)
    assert shard["inputs"].dtype == np.int64
    assert shard["finals"].shape == (5, 8, 8)
    assert shard["finals"].dtype == np.int64
    assert shard["pivot_rows"].shape == (5, 8)
    assert shard["pivot_rows"].dtype == np.int64
    assert shard["pivot_cols"].shape == (5, 8)
    assert shard["pivot_cols"].dtype == np.int64
    assert shard["pivot_mask"].shape == (5, 8)
    assert shard["pivot_mask"].dtype == np.bool_
    assert shard["op_kind"].shape == (5, 72)
    assert shard["op_kind"].dtype == np.int8
    assert shard["op_target"].shape == (5, 72)
    assert shard["op_target"].dtype == np.int64
    assert shard["op_source"].shape == (5, 72)
    assert shard["op_source"].dtype == np.int64
    assert shard["op_scalar"].shape == (5, 72)
    assert shard["op_scalar"].dtype == np.int64
    assert shard["op_mask"].shape == (5, 72)
    assert shard["op_mask"].dtype == np.bool_

    metadata = json.loads(str(shard["metadata_json"]))
    assert metadata["schema_version"] == "rref-teacher-trajectory-npz-v0.2"
    assert metadata["count"] == 5
    assert metadata["seed_start"] == 0
    assert metadata["seed_stop_exclusive"] == 5
    assert metadata["config"]["task"] == "rref"
    assert metadata["config"]["field"]["modulus"] == 101


def test_generate_rref_shard_is_deterministic_for_same_seed_range() -> None:
    left = generate_rref_shard(config_path=CONFIG, count=4, seed_start=7)
    right = generate_rref_shard(config_path=CONFIG, count=4, seed_start=7)

    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_generate_rref_shard_changes_when_seed_range_changes() -> None:
    left = generate_rref_shard(config_path=CONFIG, count=4, seed_start=7)
    right = generate_rref_shard(config_path=CONFIG, count=4, seed_start=8)

    assert not np.array_equal(left["inputs"], right["inputs"])


def test_masks_match_counts_and_padding_values() -> None:
    shard = generate_rref_shard(config_path=CONFIG, count=6, seed_start=0)

    for sample_index in range(6):
        pivot_count = int(shard["pivot_mask"][sample_index].sum())
        op_count = int(shard["op_mask"][sample_index].sum())

        assert np.all(shard["pivot_rows"][sample_index, :pivot_count] >= 0)
        assert np.all(shard["pivot_cols"][sample_index, :pivot_count] >= 0)
        assert np.all(shard["pivot_rows"][sample_index, pivot_count:] == -1)
        assert np.all(shard["pivot_cols"][sample_index, pivot_count:] == -1)
        assert np.all(shard["op_kind"][sample_index, :op_count] != 0)
        assert np.all(shard["op_kind"][sample_index, op_count:] == 0)
        assert np.all(shard["op_target"][sample_index, op_count:] == -1)
        assert np.all(shard["op_source"][sample_index, op_count:] == -1)
        assert np.all(shard["op_scalar"][sample_index, op_count:] == -1)


def test_encoded_ops_replay_to_finals_and_finals_are_rref() -> None:
    shard = generate_rref_shard(config_path=CONFIG, count=8, seed_start=0)

    for sample_index in range(8):
        ops = row_ops_from_shard_arrays(shard, sample_index)
        final = shard["finals"][sample_index].tolist()
        replayed = replay_row_ops(shard["inputs"][sample_index].tolist(), ops, 101)

        assert replayed == final
        assert is_rref_modp(final, 101)


@pytest.mark.parametrize(
    "matrix_config",
    [
        "family: sparse\n  rows: 4\n  cols: 5\n  density: 0.25\n",
        "family: low_rank\n  rows: 4\n  cols: 5\n  rank: 2\n",
    ],
)
def test_sparse_and_low_rank_configs_are_supported(tmp_path: Path, matrix_config: str) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "task: rref\n"
        "field:\n"
        "  modulus: 101\n"
        "matrix:\n"
        f"  {matrix_config}"
        "teacher: leftmost\n"
    )

    shard = generate_rref_shard(config_path=config_path, count=3, seed_start=0)

    assert shard["inputs"].shape == (3, 4, 5)
    assert shard["finals"].shape == (3, 4, 5)


@pytest.mark.parametrize(
    ("config_text", "message"),
    [
        (
            "task: hnf\n"
            "field:\n"
            "  modulus: 101\n"
            "matrix:\n"
            "  family: dense\n"
            "  rows: 2\n"
            "  cols: 2\n"
            "teacher: leftmost\n",
            "unsupported task",
        ),
        (
            "task: rref\n"
            "field:\n"
            "  modulus: 100\n"
            "matrix:\n"
            "  family: dense\n"
            "  rows: 2\n"
            "  cols: 2\n"
            "teacher: leftmost\n",
            "modulus p must be prime",
        ),
        (
            "task: rref\n"
            "field:\n"
            "  modulus: 101\n"
            "matrix:\n"
            "  rows: 2\n"
            "  cols: 2\n"
            "teacher: leftmost\n",
            "matrix.family is required",
        ),
        (
            "task: rref\n"
            "field:\n"
            "  modulus: 101\n"
            "matrix:\n"
            "  family: banded\n"
            "  rows: 2\n"
            "  cols: 2\n"
            "teacher: leftmost\n",
            "unsupported matrix family",
        ),
    ],
)
def test_invalid_configs_fail_explicitly(
    tmp_path: Path,
    config_text: str,
    message: str,
) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(config_text)

    with pytest.raises(ValueError, match=message):
        load_rref_shard_config(config_path)


def test_invalid_count_and_output_suffix_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="count must be positive"):
        generate_rref_shard(config_path=CONFIG, count=0, seed_start=0)

    with pytest.raises(ValueError, match="output path must end with .npz"):
        write_rref_shard(config_path=CONFIG, count=1, seed_start=0, out_path=tmp_path / "shard.zip")


def test_write_rref_shard_creates_loadable_npz(tmp_path: Path) -> None:
    out_path = tmp_path / "shard.npz"

    write_rref_shard(config_path=CONFIG, count=3, seed_start=11, out_path=out_path)

    with np.load(out_path, allow_pickle=False) as shard:
        assert shard["inputs"].shape == (3, 8, 8)
        metadata = json.loads(str(shard["metadata_json"]))

    assert metadata["count"] == 3
    assert metadata["seed_start"] == 11
