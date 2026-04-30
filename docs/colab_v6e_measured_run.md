# Colab v6e Reduced Profile

This runbook records the reduced RREF v6e-1 MatrixFormer/Zarr profile without
tracking Zarr shards, checkpoints, raw logs, or Colab PDFs. The tracked source
artifact is the notebook; generated outputs stay under `/tmp` unless a compact
summary is intentionally imported later.

For Colab, open `notebooks/rref_v6e_measured_run.ipynb`. It clones the repo,
installs the package, probes TPU in a subprocess, calls `run_profile(...)`
through the Python API, displays compact stage progress, and downloads the two
report files from `/tmp/nf-v6e1/rref_reduced/report`.

The reduced profile keeps the same verifier-first path as the full v6e target:

- Backward shard generation uses the explicit RREF teacher as a dataset source.
- State/action shards use the RREF MatrixFormer Zarr schema.
- Training writes Orbax checkpoints at the YAML `train.checkpoint_every`.
- Verifier-beam rollout remains neural/search driven with explicit failure
  statuses.
- CPU exact replay and `is_rref_modp` remain the verifier authority.
- No hidden teacher fallback is allowed.

## Colab v6e-1 TPU

Run this only in a Colab TPU v6e runtime. The notebook sets `JAX_PLATFORMS` to
`tpu,cpu`, probes TPU in a subprocess, then lets `scripts/rref_v6e_profile.py`
assert that the real profile selected backend is exactly `tpu`. References:
[`Cloud TPU v6e`](https://cloud.google.com/tpu/docs/v6e-training),
[`JAX platforms`](https://docs.jax.dev/en/latest/config_options.html#common-configuration-options).

Reduced default:

- config: `configs/v6e1/rref_colab_reduced_profile.yaml`
- task: 32x32 dense RREF over `F_1009`
- data: Zarr, `count: 2048`, `max_backward_ops: 64`
- model: row/col embeddings 64, hidden 256, 4 layers, 4 heads
- train: `steps: 500`, `batch_size: auto`, `checkpoint_every: 100`
- rollout: `beam_width: 8`, `max_steps: 64`, `batch_size: auto`
- work dir: `/tmp/nf-v6e1/rref_reduced/work`
- report dir: `/tmp/nf-v6e1/rref_reduced/report`

Download only:

- `/tmp/nf-v6e1/rref_reduced/report/summary.json`
- `/tmp/nf-v6e1/rref_reduced/report/report.md`

Do not commit the Colab PDF, `/tmp` work directory, Zarr shards, checkpoints, or
raw stdout/stderr logs.

## Full Target Spec

`configs/v6e1/rref_large_profile.yaml` remains the full 32x32/F_1009 target
spec. It is not the notebook default because its `count: 1048576`, 20000 train
steps, and 384-step beam horizon are intended for a later full TPU acceptance
run, not a stable 30-60 minute Colab execution.

## Local Smoke

Use this for local runner validation:

```bash
source .venv/bin/activate
python scripts/rref_v6e_profile.py \
  --config configs/v6e1/rref_matrixformer_smoke.yaml \
  --work-dir /tmp/nf-v6e1/rref_matrixformer_smoke/work \
  --out-dir /tmp/nf-v6e1/rref_matrixformer_smoke/report
```

The output should contain `schema_version: rref-v6e-profile-v1`, Zarr schema
versions, finite MatrixFormer train loss, verifier-beam status, CPU exact replay
status, backend/profile metadata, and the no-fallback statement. It must not
contain matrices, operation traces, checkpoints, or raw logs.
