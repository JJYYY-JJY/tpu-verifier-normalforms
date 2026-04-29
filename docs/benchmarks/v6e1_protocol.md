# v6e-1 Benchmark Protocol

This protocol defines how CertiNF-v6e runs are measured on Colab v6e-1 and
similar single-host TPU environments. It is a protocol for future v1.x runs;
the current measured baseline is documented in `docs/v0.9_closure.md`.

## Required Profile Metadata

Every v6e profile report must include:

- profile name;
- git commit;
- Python, JAX, JAXLIB, and platform versions;
- JAX backend;
- local/global device count;
- device kind and platform;
- host CPU count;
- host RAM total and available;
- TPU HBM total/available when a reliable API is available;
- environment variables affecting JAX/XLA;
- dataset schema versions;
- checkpoint path policy;
- memory or XProf artifact path when collected;
- no-fallback statement.

The planned status command is:

```bash
nf-agent profile v6e-status --memory-profile /tmp/nf-v6e1/profile.json
```

## v6e-1 Targets

RREF saturation target:

- 32x32 over `F_1009`;
- auto batch sizing;
- BF16 model scoring;
- device-batched beam/search;
- compact CPU verifier reports.

Resource targets:

- TPU HBM: aim for 24-30 GB used in the large profile, without OOM.
- CPU verifier/generator: support 30 or more workers when host resources allow.
- Host RAM: record available RAM before and after shard generation, training,
  rollout, and report stages.
- Neural failures: count and report failure statuses explicitly.

These targets are measurement goals, not verifier assumptions.

## Required Stages

Each measured run should separate wall time by stage:

- profile/status probe;
- exact shard generation;
- state/action shard expansion;
- train compile;
- train steady-state steps;
- checkpoint write;
- rollout compile;
- batched beam/search;
- CPU exact verification;
- report generation.

Stage output should be compact JSON. Full matrices and full operation traces do
not belong in report summaries.

## RREF Success Lines

Local smoke:

- 16x16 over `F_101`;
- train 10 steps;
- loss is finite;
- checkpoint can be loaded by rollout;
- rollout emits compact JSON with explicit success/failure status.

Colab v6e smoke:

- 32x32 over `F_1009`;
- batch size `auto`;
- compile, train, beam/search, CPU verifier, and report stages complete.

Measured v6e profile:

- backend is `tpu`;
- device count and kind are recorded;
- HBM/CPU/RAM metrics are recorded when available;
- all neural failures remain neural failure statuses;
- no hidden teacher fallback;
- at least one hard RREF family should eventually beat leftmost or min-fill on
  trace length, fill-in, or certificate size while exact verification passes.

## HNF Success Lines

HNF growth-search profiles should report:

- validity;
- step count;
- max bitlength;
- max absolute value seen;
- fill-in delta;
- certificate size;
- wall time.

The v1.1 target is to improve at least one of max bitlength, step count, or
fill-in delta on sparse 8/16/32 integer families versus `row_hnf` or a
supervised baseline, with exact verifier success and no teacher fallback.

## SNF Success Lines

SNF certificate-search profiles should report:

- replay validity;
- transform validity;
- `U * A * V = D` validity;
- diagonal divisibility validity;
- operation count;
- certificate size;
- bitlength;
- wall time;
- Lean sample pass count for exported small samples.

The v1.2 target is small SNF exported samples passing both Python verifier and
Lean checker, plus larger compact Python benchmark summaries without full
matrix or trace payloads.

## Artifact Rules

Commit allowed:

- compact `metrics.json`;
- compact `report.md`;
- small plots;
- small checker fixtures;
- sanitized config files.

Do not commit:

- NPZ/Zarr production shards;
- Orbax checkpoints;
- raw XProf directories;
- raw memory dumps;
- raw logs;
- Colab PDFs;
- full matrices or operation traces in benchmark summaries.

Recommended local layout:

```text
/tmp/nf-v6e1/
  data/
  checkpoints/
  xprof/
  memory/
  reports/
```

Only the compact report outputs should be copied into tracked paths.
