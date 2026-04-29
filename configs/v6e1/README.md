# v6e1 Configs

These YAML files are planned CertiNF-v6e profile specs. Most are not consumed
by the current CLI yet.

Current executable configs remain the flat files under `configs/`. The v6e1
files define the intended shape for future commands such as:

- `nf-agent data make-rref-state-shard`
- `nf-agent train rref-matrixformer`
- `nf-agent rollout rref-matrixformer`
- `nf-agent rollout rref-verifier-beam`
- `nf-agent report v6e-profile`

The implemented alpha NPZ smoke commands are:

```bash
nf-agent data make-rref-backward-shard \
  --config configs/rref_backward_4x4_mod101.yaml \
  --count 4 \
  --seed-start 0 \
  --out /tmp/rref_backward_4x4_smoke.npz

nf-agent data make-rref-state-shard \
  --trace-shard /tmp/rref_backward_4x4_smoke.npz \
  --out /tmp/rref_state_4x4_smoke.npz

nf-agent train rref-matrixformer \
  --data /tmp/rref_state_4x4_smoke.npz \
  --steps 2 \
  --batch-size 4 \
  --learning-rate 0.001 \
  --seed 0 \
  --out /tmp/rref_matrixformer_smoke_ckpt \
  --row-embedding-dim 8 \
  --col-embedding-dim 8 \
  --hidden-dim 32 \
  --layers 1 \
  --num-heads 1

nf-agent rollout rref-matrixformer \
  --data /tmp/rref_state_4x4_smoke.npz \
  --checkpoint /tmp/rref_matrixformer_smoke_ckpt \
  --sample-index 0 \
  --max-steps 8 \
  --row-embedding-dim 8 \
  --col-embedding-dim 8 \
  --hidden-dim 32 \
  --layers 1 \
  --num-heads 1
```

They currently support NPZ smoke shards and local greedy checkpoint rollout
only. The large `configs/v6e1/*` files remain protocol/spec inputs until the
Zarr, TPU batched beam/search, and v6e profile runners land.

Large outputs referenced by these configs must stay outside git. Commit only
compact report JSON/Markdown, small fixtures, and sanitized config files.
