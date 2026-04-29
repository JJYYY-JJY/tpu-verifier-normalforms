# Pre-Colab Local Preflight

Before opening the v6e Colab path, run the local preflight once from repo root:

```bash
source .venv/bin/activate
python scripts/pre_colab_preflight.py \
  --work-dir /tmp/nf-pre-colab \
  --fixture-dir tests/fixtures/pre_colab \
  --write-fixtures
```

The script runs the local pieces that should not be debugged first on Colab:

1. RREF fixed-shape shard generation.
2. Short RREF pivot training with `--hidden-size 32`.
3. RREF neural rollout with the same hidden-size contract.
4. RREF shard benchmark with exact leftmost and neural policies.
5. HNF v0.8 mini experiment.
6. SNF generated-certificate benchmark.
7. v0.9 RREF/HNF/SNF report smoke.
8. Lean checker build.

Use dry-run mode to inspect the exact command list:

```bash
python scripts/pre_colab_preflight.py --dry-run
```

## Outputs

Heavy or noisy intermediates stay under `--work-dir`:

- `rref_8x8_train_smoke.npz`
- `rref_pivot_ckpt/`
- `hnf_v08/`
- `report_smoke/`
- raw stdout/stderr logs under `logs/`

Tracked fixtures stay compact under `tests/fixtures/pre_colab/`:

- `manifest.json`
- `rref_shard_benchmark_smoke.json`
- `hnf_v08_metrics_smoke.json`
- `snf_benchmark_smoke.json`
- `report_metrics_smoke.json`

These JSON fixtures intentionally omit full matrices, transforms, operation
traces, NPZ shards, checkpoints, plots, and raw logs.

## Verdicts

`manifest.json` is the audit index. It records commands, exit codes, artifact
paths, git commit, Python version, Lean version, Lake version, and compact
verdicts.

Expected status:

- `overall_status == "ok"`.
- RREF shard benchmark includes `leftmost` and `neural` policies. Neural rollout
  failures are recorded directly; no teacher fallback is allowed.
- HNF v0.8 may report `status == "failed_threshold"` and still count as a
  completed plumbing preflight. That status means the mini learned policy did
  not beat the threshold; it is not a script failure and must not be rewritten as
  success.
- SNF `certificate_replay` should have `success_rate == 1.0`.
- Report smoke should produce `report.md`, `metrics.json`, and plots in
  `--work-dir`.
- Lean `lake build` must exit 0.

Failures that must be fixed before Colab:

- Any command exits nonzero.
- Missing `neural` policy in the RREF shard benchmark.
- Malformed or non-compact benchmark/report JSON.
- SNF verifier failures.
- Lean build failure.

## Full Gate

After updating fixtures or code, still run the repository gate:

```bash
source .venv/bin/activate
python -m pip check
ruff check .
mypy src
pytest
nf-agent --help
cd lean && lake build
git diff --check
```
