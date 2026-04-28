"""Dataset and matrix-family generators."""

from nf_agent.data.rref_shards import (
    RREFShardConfig,
    generate_rref_shard,
    load_rref_shard_config,
    row_ops_from_shard_arrays,
    write_rref_shard,
)

__all__ = [
    "RREFShardConfig",
    "generate_rref_shard",
    "load_rref_shard_config",
    "row_ops_from_shard_arrays",
    "write_rref_shard",
]
