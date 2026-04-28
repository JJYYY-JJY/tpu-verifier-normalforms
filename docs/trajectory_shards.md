# Trajectory Shards

`v0.2` RREF teacher shards are fixed-shape, uncompressed `.npz` files generated
offline from exact deterministic RREF teacher traces. They are dataset
artifacts, not verifiers.

## RREF NPZ Schema

Use:

```bash
nf-agent data make-rref-shard \
  --config configs/rref_8x8_mod101.yaml \
  --count 1024 \
  --seed-start 0 \
  --out results/data/rref_8x8_mod101_seed0_count1024.npz
```

Arrays:

- `inputs`: `int64[N, rows, cols]`
- `finals`: `int64[N, rows, cols]`
- `pivot_rows`: `int64[N, min(rows, cols)]`, padded with `-1`
- `pivot_cols`: `int64[N, min(rows, cols)]`, padded with `-1`
- `pivot_mask`: `bool[N, min(rows, cols)]`
- `op_kind`: `int8[N, min(rows, cols) * (rows + 1)]`
- `op_target`: `int64[N, max_ops]`, padded with `-1`
- `op_source`: `int64[N, max_ops]`, padded with `-1`
- `op_scalar`: `int64[N, max_ops]`, padded with `-1`
- `op_mask`: `bool[N, max_ops]`
- `metadata_json`: JSON string

`op_kind` encoding:

- `0`: padding
- `1`: row swap
- `2`: row scale
- `3`: add row multiple

`max_ops = min(rows, cols) * (rows + 1)`. For each pivot, the supported teachers
can emit at most one swap, one scale, and one add for every non-pivot row.

`metadata_json` includes the schema version, normalized source config, count,
seed range, matrix shape, op encoding, and padding value.

## Boundaries

- Supported task: `rref`
- Supported teacher: `leftmost`, `min_fill`
- Supported matrix families: `dense`, `sparse`, `low_rank`
- Supported field: prime `field.modulus`
- Shards are produced with `numpy.savez`, not compressed.
- Generated `.npz` files under `results/data/` are ignored by git.
