# Colab v6e Notes

The TPU training target is v6e-compatible JAX/Flax/Optax imitation learning.

Constraints:

- Fixed-shape NPZ shards for accelerator-friendly batching.
- Orbax checkpoints for explicit resume.
- Grain input pipeline once shard format is stable.
- Teacher generation remains CPU/offline; training consumes serialized traces.
- Rollout evaluation must record invalid actions instead of hiding them behind
  oracle fallback.

