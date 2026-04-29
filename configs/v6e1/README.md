# v6e1 Configs

These YAML files are planned CertiNF-v6e profile specs. Most are not consumed
by the current CLI yet.

Current executable configs remain the flat files under `configs/`. The v6e1
files define the intended shape for future commands such as:

- `nf-agent data make-rref-state-shard`
- `nf-agent train rref-matrixformer`
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
```

They currently support NPZ smoke shards only. The large `configs/v6e1/*` files
remain protocol/spec inputs until the Zarr, MatrixFormer, beam/search, and v6e
profile runners land.

Large outputs referenced by these configs must stay outside git. Commit only
compact report JSON/Markdown, small fixtures, and sanitized config files.
