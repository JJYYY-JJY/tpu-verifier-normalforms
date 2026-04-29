from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import zarr

ShardValue: TypeAlias = np.ndarray[Any, np.dtype[Any]]
ShardArrays: TypeAlias = dict[str, ShardValue]


def shard_format(path: str | Path) -> str:
    suffix = Path(path).suffix
    if suffix == ".npz":
        return "npz"
    if suffix == ".zarr":
        return "zarr"
    raise ValueError("data path must end with .npz or .zarr")


def _chunks_for(array: ShardValue) -> tuple[int, ...] | None:
    if array.shape == ():
        return None
    first = min(int(array.shape[0]), 128)
    return (first, *tuple(int(axis) for axis in array.shape[1:]))


def write_shard_arrays(path: str | Path, arrays: Mapping[str, ShardValue]) -> None:
    shard_path = Path(path)
    storage_format = shard_format(shard_path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    if storage_format == "npz":
        np.savez(shard_path, **arrays)  # type: ignore[arg-type]
        return

    if shard_path.exists():
        if shard_path.is_dir():
            shutil.rmtree(shard_path)
        else:
            shard_path.unlink()
    group = zarr.open_group(str(shard_path), mode="w", zarr_format=2)
    for key, value in arrays.items():
        array = np.asarray(value)
        chunks = _chunks_for(array)
        if chunks is None:
            group.create_array(key, data=array)
        else:
            group.create_array(key, data=array, chunks=chunks)


def load_shard_arrays(
    path: str | Path,
    required_arrays: Mapping[str, np.dtype[Any]],
) -> tuple[ShardArrays, ShardValue]:
    shard_path = Path(path)
    storage_format = shard_format(shard_path)
    if not shard_path.exists():
        raise ValueError(f"data path does not exist: {shard_path}")

    required = [*required_arrays.keys(), "metadata_json"]
    if storage_format == "npz":
        with np.load(shard_path, allow_pickle=False) as shard:
            missing = sorted(key for key in required if key not in shard.files)
            if missing:
                raise ValueError(f"missing required array(s): {', '.join(missing)}")
            arrays = {key: np.asarray(shard[key]) for key in required_arrays}
            metadata_json = np.asarray(shard["metadata_json"])
        return arrays, metadata_json

    group = zarr.open_group(str(shard_path), mode="r")
    keys = set(group.array_keys())
    missing = sorted(key for key in required if key not in keys)
    if missing:
        raise ValueError(f"missing required array(s): {', '.join(missing)}")
    arrays = {key: np.asarray(group[key]) for key in required_arrays}
    metadata_json = np.asarray(group["metadata_json"])
    return arrays, metadata_json
