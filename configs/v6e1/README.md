# v6e1 Configs

These YAML files define CertiNF-v6e profile specs. The RREF MatrixFormer smoke,
reduced Colab, and large target profiles are consumed by
`scripts/rref_v6e_profile.py`.

The executable v1.0-beta1 RREF surface is:

- `nf-agent profile v6e-status`
- `nf-agent data make-rref-backward-shard`
- `nf-agent data make-rref-state-shard`
- `nf-agent train rref-matrixformer`
- `nf-agent rollout rref-verifier-beam`
- `nf-agent report v6e-profile`
- `python scripts/rref_v6e_profile.py`

The executable v1.1 HNF exact-search beta surface is:

- `nf-agent data make-hnf-backward-shard`
- `nf-agent profile hnf-growth`

Local Zarr smoke:

```bash
nf-agent profile v6e-status \
  --memory-profile /tmp/nf-v6e1/profile.json

nf-agent data make-rref-backward-shard \
  --config configs/rref_backward_4x4_mod101.yaml \
  --count 4 \
  --seed-start 0 \
  --out /tmp/rref_backward_4x4_smoke.zarr

nf-agent data make-rref-state-shard \
  --trace-shard /tmp/rref_backward_4x4_smoke.zarr \
  --out /tmp/rref_state_4x4_smoke.zarr

nf-agent train rref-matrixformer \
  --data /tmp/rref_state_4x4_smoke.zarr \
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

nf-agent rollout rref-verifier-beam \
  --data /tmp/rref_state_4x4_smoke.zarr \
  --checkpoint /tmp/rref_matrixformer_smoke_ckpt \
  --sample-index 0 \
  --max-steps 8 \
  --beam-width 4 \
  --batch-size auto \
  --row-embedding-dim 8 \
  --col-embedding-dim 8 \
  --hidden-dim 32 \
  --layers 1 \
  --num-heads 1

python scripts/rref_v6e_profile.py \
  --config configs/v6e1/rref_matrixformer_smoke.yaml \
  --work-dir /tmp/nf-v6e1/rref_matrixformer_smoke/work \
  --out-dir /tmp/nf-v6e1/rref_matrixformer_smoke/report
```

HNF growth-search beta. This is exact row-preconditioned search over
unimodular row swaps followed by `row_hnf`; it does not train or run an HNF
MatrixFormer.

```bash
nf-agent profile hnf-growth \
  --config configs/v6e1/hnf_growth_search.yaml \
  --work-dir /tmp/nf-v6e1/hnf_growth/work \
  --out-dir /tmp/nf-v6e1/hnf_growth/report \
  --family sparse_8x8 \
  --count 8 \
  --candidate-limit 64
```

Reduced Colab TPU execution uses `configs/v6e1/rref_colab_reduced_profile.yaml`.
This is the notebook default because it keeps the real 32x32/F_1009
MatrixFormer/Zarr/verifier-beam path while bounding the run to a practical
Colab window:

```bash
python scripts/rref_v6e_profile.py \
  --config configs/v6e1/rref_colab_reduced_profile.yaml \
  --work-dir /tmp/nf-v6e1/rref_reduced/work \
  --out-dir /tmp/nf-v6e1/rref_reduced/report
```

Full Colab TPU acceptance keeps `configs/v6e1/rref_large_profile.yaml` as the
target spec:

```bash
python scripts/rref_v6e_profile.py \
  --config configs/v6e1/rref_large_profile.yaml \
  --work-dir /tmp/nf-v6e1/rref_large/work \
  --out-dir /tmp/nf-v6e1/rref_large/report
```

Large outputs referenced by these configs must stay outside git. Commit only
compact report JSON/Markdown, small fixtures, and sanitized config files.
