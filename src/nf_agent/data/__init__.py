"""Dataset and matrix-family generators."""

from nf_agent.data.rref_shards import (
    RREFShardConfig,
    RREFShardSamples,
    generate_rref_shard,
    load_rref_shard_config,
    make_rref_grain_dataset,
    row_ops_from_shard_arrays,
    write_rref_shard,
)

__all__ = [
    "RREFShardSamples",
    "RREFShardConfig",
    "generate_rref_shard",
    "load_rref_shard_config",
    "make_rref_grain_dataset",
    "row_ops_from_shard_arrays",
    "write_rref_shard",
]
