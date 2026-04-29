# Colab v6e Measured Run

This runbook records large RREF 8x8/F_101 measured runs without tracking NPZ
shards, checkpoints, raw logs, or Colab PDFs. The tracked artifacts are compact
JSON plus Markdown summaries under `results/measured/`.

The harness uses the same verifier-first path as the smoke notebook:

- Shard generation calls the explicit RREF teacher as a dataset source.
- Training writes Orbax checkpoints at a configurable cadence.
- Benchmarking uses `nf-agent`'s compact RREF benchmark schema.
- Neural failures remain visible in `policies.neural.aggregate.status_counts`.
- No hidden teacher fallback is allowed.

## Local Apple M4 CPU

The Apple M4 profile pins JAX to CPU. Official JAX installation docs say Mac GPU
is not supported by JAX and recommend the standard CPU install for macOS GPU
use; Apple's Metal plug-in exists but is experimental and does not pass all JAX
tests. The default measured run therefore uses CPU and records the actual JAX
backend in the output JSON. References:
[`JAX Mac GPU`](https://docs.jax.dev/en/latest/installation.html#mac-gpu),
[`Apple jax-metal`](https://developer.apple.com/metal/jax/).

```bash
source .venv/bin/activate
python scripts/rref_measured_run.py \
  --profile apple-m4-large \
  --work-dir /tmp/nf-rref-apple-m4-large \
  --out results/measured/rref_8x8_mod101_apple_m4_large.json \
  --summary-md results/measured/rref_8x8_mod101_apple_m4_large.md
```

The profile sets:

- `JAX_PLATFORMS=cpu`
- `OMP_NUM_THREADS=10`
- `VECLIB_MAXIMUM_THREADS=10`
- `XLA_FLAGS=--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=10`

## Colab v6e-1 TPU

Run this only in a Colab TPU v6e runtime. The profile requests TPU first and
asserts that JAX selected backend is exactly `tpu`; if not, it fails before
training. References:
[`Cloud TPU v6e`](https://cloud.google.com/tpu/docs/v6e-training),
[`JAX platforms`](https://docs.jax.dev/en/latest/config_options.html#common-configuration-options).

```bash
python scripts/rref_measured_run.py \
  --profile colab-v6e1-large \
  --work-dir /tmp/nf-rref-colab-v6e1-large \
  --out /tmp/rref_8x8_mod101_colab_v6e1_large.json \
  --summary-md /tmp/rref_8x8_mod101_colab_v6e1_large.md
```

After the Colab run, copy only the compact JSON and Markdown summary into:

- `results/measured/rref_8x8_mod101_colab_v6e1_large.json`
- `results/measured/rref_8x8_mod101_colab_v6e1_large.md`

Do not commit the Colab PDF, `/tmp` work directory, NPZ shard, checkpoints, or
raw stdout/stderr logs.

## Smoke

Use this for local harness validation:

```bash
source .venv/bin/activate
python scripts/rref_measured_run.py \
  --profile local-smoke \
  --work-dir /tmp/nf-rref-smoke \
  --out results/measured/rref_smoke.json \
  --summary-md results/measured/rref_smoke.md
```

The smoke output should contain `schema_version: rref-measured-run-v1`, backend
metadata, batch calibration records, train metrics, compact policy aggregates,
and the no-fallback statement. It must not contain matrices, row operations,
checkpoints, or raw logs.
