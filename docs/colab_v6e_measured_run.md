# Colab v6e Reduced Profiles

This runbook records reduced RREF v6e-1 MatrixFormer/Zarr profiles without
tracking Zarr shards, checkpoints, raw logs, or Colab PDFs. The tracked source
artifact is the notebook; generated outputs stay under `/tmp` unless a compact
summary is intentionally imported later.

For Colab, open `notebooks/rref_v6e_measured_run.ipynb`. It clones the repo,
installs the package, probes TPU in a subprocess, calls `run_profile(...)`
through the Python API, displays compact stage progress, and downloads the two
report files from the selected profile report directory.

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

## Completed 500-Step Smoke

The completed reduced smoke is imported as compact evidence:

- `results/measured/rref_32x32_mod1009_colab_v6e1_reduced_500step.json`
- `results/measured/rref_32x32_mod1009_colab_v6e1_reduced_500step.md`

It used `configs/v6e1/rref_colab_reduced_profile.yaml`:

- config: `configs/v6e1/rref_colab_reduced_profile.yaml`
- task: 32x32 dense RREF over `F_1009`
- data: Zarr, `count: 2048`, `max_backward_ops: 64`
- model: row/col embeddings 64, hidden 256, 4 layers, 4 heads
- train: `steps: 500`, `batch_size: auto`, `checkpoint_every: 100`
- rollout: `beam_width: 8`, `max_steps: 64`, `batch_size: auto`
- work dir: `/tmp/nf-v6e1/rref_reduced/work`
- report dir: `/tmp/nf-v6e1/rref_reduced/report`

Observed result: backend `tpu`, 2048 traces, 500 train steps, wall time
`531.65s`, top-level `status: ok`, and `beam.status: max_steps_exceeded`.
Here top-level `status: ok` means the profile pipeline completed and exact
replay was checked. Beam solve status must be read from `beam.status` and
`beam.success`.

## Next Colab Default

The next notebook default is the longer reduced profile:

- config: `configs/v6e1/rref_colab_reduced_long_profile.yaml`
- task: 32x32 dense RREF over `F_1009`
- data: Zarr, `count: 8192`, `max_backward_ops: 96`
- model: row/col embeddings 64, hidden 256, 4 layers, 4 heads
- train: `steps: 2000`, `batch_size: auto`, `checkpoint_every: 250`
- rollout: `beam_width: 8`, `max_steps: 96`, `batch_size: auto`
- expected wall time: about 30-40 minutes on a similar Colab v6e-1 runtime
- work dir: `/tmp/nf-v6e1/rref_reduced_long/work`
- report dir: `/tmp/nf-v6e1/rref_reduced_long/report`

Download only:

- `/tmp/nf-v6e1/rref_reduced_long/report/summary.json`
- `/tmp/nf-v6e1/rref_reduced_long/report/report.md`

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
