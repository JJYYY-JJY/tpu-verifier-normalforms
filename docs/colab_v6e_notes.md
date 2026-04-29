# Colab v6e Notes

The RREF v6e smoke/training notebook is tracked at
`notebooks/rref_v6e_smoke_training.ipynb`. It is a local-and-Colab smoke path
for the current RREF stack: fixed-shape shard generation, short JAX/Flax/Optax
imitation training, verifier-guided neural rollout, and shard-source benchmark
comparison between the exact leftmost baseline and the neural policy.

The notebook does not add TPU-only dependencies and does not change verifier,
teacher, rollout, or benchmark behavior. A Colab TPU v6e runtime may accelerate
the JAX training cell, but the recommended smoke parameters below are small
enough for CPU validation.

Before opening Colab, run the local preflight in
`docs/pre_colab_preflight.md`. Colab should accelerate the RREF training smoke;
it should not be the first place that catches broken shard, rollout, benchmark,
report, or Lean checker plumbing.

Local smoke mirrored by the notebook:

```bash
source .venv/bin/activate
nf-agent --help
nf-agent data make-rref-shard \
  --config configs/rref_8x8_mod101.yaml \
  --count 8 \
  --seed-start 0 \
  --out /tmp/rref_8x8_train_smoke.npz
nf-agent train rref-pivot \
  --data /tmp/rref_8x8_train_smoke.npz \
  --steps 2 \
  --batch-size 4 \
  --learning-rate 0.001 \
  --seed 0 \
  --hidden-size 32 \
  --out /tmp/rref_pivot_ckpt
nf-agent rollout rref-neural \
  --data /tmp/rref_8x8_train_smoke.npz \
  --checkpoint /tmp/rref_pivot_ckpt \
  --sample-index 0 \
  --max-steps 8 \
  --hidden-size 32
nf-agent benchmark rref \
  --source shard \
  --data /tmp/rref_8x8_train_smoke.npz \
  --count 4 \
  --checkpoint /tmp/rref_pivot_ckpt \
  --max-steps 8 \
  --hidden-size 32
```

Implemented constraints:

- Fixed-shape NPZ shards feed accelerator-friendly batches.
- Local preflight, notebook training, rollout, and shard benchmark all use
  `--hidden-size 32` so checkpoint shapes match.
- Grain reads validated random-access shard samples and shuffles by seed.
- `PivotMLP` trains pivot and row-operation heads with masked losses.
- Orbax stores and restores latest checkpoints for explicit resume.
- Neural rollout runs with legal action masking and no teacher fallback.
- RREF benchmark evaluates leftmost and optional neural rollout on the same
  samples, with exact replay, `is_rref_modp`, and fill-in density metrics.
- Teacher generation remains CPU/offline; training consumes serialized traces.
- The notebook writes smoke outputs under `/tmp`, not repo-tracked result paths.

Still out of scope:

- TPU-specific setup libraries or TPU-only notebook cells.
- Hidden deterministic fallback in rollout code remains prohibited.
- HNF/SNF rollout expansion remains separate roadmap work; the Lean checker now
  covers RREF and SNF JSON certificates.
