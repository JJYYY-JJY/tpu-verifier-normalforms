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

## RREF Backward Trace NPZ Schema

Use:

```bash
nf-agent data make-rref-backward-shard \
  --config configs/rref_backward_4x4_mod101.yaml \
  --count 1024 \
  --seed-start 0 \
  --out results/data/rref_backward_4x4_mod101_seed0_count1024.npz
```

Schema version: `rref-backward-trace-npz-v1`.

Generation:

```text
deterministic source matrix
-> exact leftmost RREF final
-> sampled invertible row ops applied backward from final
-> input matrix
-> inverse row-op trace stored for input -> final replay
```

Arrays:

- `inputs`: `int64[N, rows, cols]`
- `finals`: `int64[N, rows, cols]`
- `pivots`: `int64[N, min(rows, cols), 2]`, padded with `[-1, -1]`
- `ops`: `int64[N, max_ops, 4]`, columns `[kind, target, source, scalar]`
- `op_mask`: `bool[N, max_ops]`
- `metadata_json`: JSON string

`ops[:, :, 0]` uses the same row-op kind encoding as v0.2:

- `0`: padding
- `1`: row swap
- `2`: row scale
- `3`: add row multiple

Inactive op rows must be `[0, -1, -1, -1]`. Active `swap` and `add` operations
must have distinct target/source rows. Active `scale` and `add` scalars must be
nonzero modulo `p`. The loader checks prime modulus, dtypes, shapes, padding,
operation legality, pivots derived from `finals`, exact replay from `inputs` to
`finals`, and `is_rref_modp(finals[i], p)`.

Current executable format is NPZ only. Zarr is reserved for later large v6e
state/action datasets.

## HNF NPZ Schema

Use:

```bash
nf-agent data make-hnf-shard \
  --config configs/hnf_sparse_8x8.yaml \
  --count 1024 \
  --seed-start 0 \
  --out results/data/hnf_8x8_sparse_seed0_count1024.npz
```

Schema version: `hnf-teacher-trajectory-npz-v0.8`.

Arrays:

- `inputs`: `int64[N, rows, cols]`
- `finals`: `int64[N, rows, cols]`
- `op_kind`: `int8[N, max_ops]`
- `op_target`: `int64[N, max_ops]`, padded with `-1`
- `op_source`: `int64[N, max_ops]`, padded with `-1`
- `op_scalar_id`: `int64[N, max_ops]`, padded with `-1`
- `op_scalar_value`: `int64[N, max_ops]`, padded with `-1`
- `op_mask`: `bool[N, max_ops]`
- `scalar_vocab`: sorted `int64[V]` exact integer scalars observed in teacher
  `add` operations
- `metadata_json`: JSON string

`op_kind` encoding:

- `0`: padding
- `1`: row swap
- `2`: row negation
- `3`: add integer row multiple

HNF `max_ops` is derived by a two-pass shard build from the maximum `row_hnf`
trace length in the shard, with a minimum width of `1`. The schema stores
`input_scale` for model-side normalization; exact replay uses raw `int64`
arrays and does not use floating point.

Boundaries:

- Supported task: `hnf`
- Supported integer matrix family: sparse Bernoulli support
- Oracle/dataset source: `row_hnf`
- Supported operations: unimodular integer `swap`, `negate`, and
  `add(target, source, scalar)` with distinct target/source rows
- `row_hnf` is allowed as an oracle/baseline/dataset source only; learned
  rollout and beam search report failures directly.
