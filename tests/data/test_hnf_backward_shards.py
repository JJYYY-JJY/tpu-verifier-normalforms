import json
from pathlib import Path

import numpy as np
import pytest

from nf_agent.data.hnf_shards import (
    HNFTrajectory,
    generate_hnf_backward_shard,
    hnf_backward_shard_arrays_from_trajectories,
    integer_row_ops_from_hnf_shard_arrays,
    load_hnf_backward_shard,
    write_hnf_backward_shard,
)
from nf_agent.data.shard_storage import write_shard_arrays
from nf_agent.env.hnf_int import IntegerRowOp, is_row_hnf, replay_integer_row_ops


def _growth_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "hnf_growth.yaml"
    config_path.write_text(
        "task: hnf_growth_search\n"
        "integer_families:\n"
        "  - name: tiny_sparse\n"
        "    rows: 3\n"
        "    cols: 4\n"
        "    density: 0.5\n"
        "    entry_bound: 4\n"
        "backward_trace:\n"
        "  schema: hnf-backward-trace-zarr-v1\n"
        "  format: zarr\n"
        "  require_unimodular_ops: true\n"
    )
    return config_path


def test_generate_hnf_backward_shard_has_schema_metadata_and_exact_replay(
    tmp_path: Path,
) -> None:
    shard = generate_hnf_backward_shard(
        config_path=_growth_config(tmp_path),
        family="tiny_sparse",
        count=4,
        seed_start=10,
        storage_format="npz",
    )

    assert shard["inputs"].shape == (4, 3, 4)
    assert shard["inputs"].dtype == np.int64
    assert shard["finals"].shape == (4, 3, 4)
    assert shard["op_kind"].dtype == np.int8
    assert shard["op_mask"].dtype == np.bool_
    assert shard["scalar_vocab"].ndim == 1

    metadata = json.loads(str(shard["metadata_json"]))
    assert metadata["schema_version"] == "hnf-backward-trace-zarr-v1"
    assert metadata["config"]["family"]["name"] == "tiny_sparse"
    assert metadata["config"]["backward_trace"]["format"] == "npz"
    assert metadata["count"] == 4
    assert metadata["seed_start"] == 10
    assert metadata["shape"] == {"rows": 3, "cols": 4}
    assert metadata["op_encoding"] == {"pad": 0, "swap": 1, "negate": 2, "add": 3}

    for sample_index in range(4):
        ops = integer_row_ops_from_hnf_shard_arrays(shard, sample_index)
        replayed = replay_integer_row_ops(shard["inputs"][sample_index].tolist(), ops)
        final = shard["finals"][sample_index].tolist()
        assert replayed == final
        assert is_row_hnf(final)


@pytest.mark.parametrize("suffix", [".npz", ".zarr"])
def test_hnf_backward_shard_roundtrips_npz_and_zarr(tmp_path: Path, suffix: str) -> None:
    out_path = tmp_path / f"backward{suffix}"

    write_hnf_backward_shard(
        config_path=_growth_config(tmp_path),
        family="tiny_sparse",
        count=3,
        seed_start=0,
        out_path=out_path,
    )
    arrays, metadata = load_hnf_backward_shard(out_path)

    assert arrays["inputs"].shape == (3, 3, 4)
    assert metadata["schema_version"] == "hnf-backward-trace-zarr-v1"
    assert metadata["config"]["backward_trace"]["format"] == suffix.removeprefix(".")
    assert metadata["count"] == 3


def _manual_backward_arrays(storage_format: str = "npz") -> dict[str, np.ndarray]:
    return hnf_backward_shard_arrays_from_trajectories(
        [
            HNFTrajectory(
                input_matrix=[[1, 0], [2, 1]],
                final_matrix=[[1, 0], [0, 1]],
                ops=(IntegerRowOp.add(1, 0, -2),),
                seed=0,
            ),
            HNFTrajectory(
                input_matrix=[[1, 0], [0, 1]],
                final_matrix=[[1, 0], [0, 1]],
                ops=(),
                seed=1,
            ),
        ],
        config_payload={
            "task": "hnf_growth_search",
            "family": {
                "name": "manual",
                "rows": 2,
                "cols": 2,
                "density": 1,
                "entry_bound": 2,
            },
            "backward_trace": {
                "schema": "hnf-backward-trace-zarr-v1",
                "format": storage_format,
                "require_unimodular_ops": True,
                "require_exact_replay": True,
            },
        },
        seed_start=0,
        storage_format=storage_format,
    )


def test_load_hnf_backward_shard_rejects_padding_and_illegal_ops(tmp_path: Path) -> None:
    arrays = _manual_backward_arrays()

    bad_padding = {key: value.copy() for key, value in arrays.items()}
    bad_padding["op_target"][1, 0] = 0
    bad_padding_path = tmp_path / "bad_padding.npz"
    np.savez(bad_padding_path, **bad_padding)
    with pytest.raises(ValueError, match="op_target padding must be -1"):
        load_hnf_backward_shard(bad_padding_path)

    bad_op = {key: value.copy() for key, value in _manual_backward_arrays("zarr").items()}
    bad_op["op_source"][0, 0] = bad_op["op_target"][0, 0]
    bad_op_path = tmp_path / "bad_op.zarr"
    write_shard_arrays(bad_op_path, bad_op)
    with pytest.raises(ValueError, match="target and source rows must be distinct"):
        load_hnf_backward_shard(bad_op_path)


def test_load_hnf_backward_shard_rejects_bad_replay(tmp_path: Path) -> None:
    arrays = _manual_backward_arrays()
    arrays["finals"] = arrays["finals"].copy()
    arrays["finals"][0, 0, 0] += 1
    bad_path = tmp_path / "bad_replay.npz"
    np.savez(bad_path, **arrays)

    with pytest.raises(ValueError, match="does not replay"):
        load_hnf_backward_shard(bad_path)
