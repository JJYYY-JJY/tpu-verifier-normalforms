# Colab v6e Notes

The current RREF stack is local-first JAX/Flax/Optax imitation learning over
fixed-shape trajectory shards, with verifier-guided neural rollout and the RREF
benchmark available as local CLI commands. Colab/TPU notebook packaging remains
a follow-up item.

Local smoke:

```bash
source .venv/bin/activate
nf-agent data make-rref-shard \
  --config configs/rref_8x8_mod101.yaml \
  --count 8 \
  --seed-start 0 \
  --out /tmp/rref_8x8_train_smoke.npz
nf-agent train rref-pivot \
  --data /tmp/rref_8x8_train_smoke.npz \
  --steps 2 \
  --batch-size 4 \
  --seed 0 \
  --out /tmp/rref_pivot_ckpt
nf-agent benchmark rref \
  --source shard \
  --data /tmp/rref_8x8_train_smoke.npz \
  --count 8 \
  --checkpoint /tmp/rref_pivot_ckpt \
  --max-steps 8
```

Implemented constraints:

- Fixed-shape NPZ shards feed accelerator-friendly batches.
- Grain reads validated random-access shard samples and shuffles by seed.
- `PivotMLP` trains pivot and row-operation heads with masked losses.
- Orbax stores and restores latest checkpoints for explicit resume.
- Neural rollout runs with legal action masking and no teacher fallback.
- RREF benchmark evaluates leftmost and optional neural rollout on the same
  samples, with exact replay, `is_rref_modp`, and fill-in density metrics.
- Teacher generation remains CPU/offline; training consumes serialized traces.

Still out of scope:

- TPU-specific Colab setup and notebook cells.
- Hidden deterministic fallback in rollout code remains prohibited.
