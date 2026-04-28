# TPU Verifier Normal Forms

Research monorepo for verifier-guided agents that learn exact matrix normal-form
procedures while remaining checkable by deterministic algebraic replay.

The first vertical slice is finite-field RREF over `F_101`:

```text
random F_101 matrix
-> explicit deterministic teacher trajectory
-> exact modular row operations
-> final RREF
-> trace replay
-> verifier predicate
```

The integer row-HNF environment is now available as a verifier-first prototype:

```text
integer matrix
-> exact unimodular row-operation trace
-> row-style HNF
-> trace replay
-> exact coefficient-growth metrics
```

## Correctness Model

- All verifier paths are exact integer/modular arithmetic.
- No floating-point computation is accepted for certificate replay or algebraic
  predicates.
- Invalid modulus, malformed matrices, or illegal row operations fail fast.
- Neural rollout must not silently fall back to deterministic teachers.
- Deterministic teachers are oracle, baseline, and dataset sources only.

## Install

```bash
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-dev.txt -e .
```

Sage is managed outside the Python venv:

```bash
brew install micromamba
micromamba create -y -n nf-sage -c conda-forge sage
```

## Smoke Commands

```bash
source .venv/bin/activate
python -m pip check
ruff check .
mypy src
pytest
nf-agent --help
nf-agent data make-rref-shard \
  --config configs/rref_8x8_mod101.yaml \
  --count 4 \
  --seed-start 0 \
  --out /tmp/rref_8x8_smoke.npz
nf-agent train rref-pivot \
  --data /tmp/rref_8x8_smoke.npz \
  --steps 2 \
  --batch-size 4 \
  --learning-rate 0.001 \
  --seed 0 \
  --out /tmp/rref_pivot_smoke_ckpt
nf-agent rollout rref-neural \
  --data /tmp/rref_8x8_smoke.npz \
  --checkpoint /tmp/rref_pivot_smoke_ckpt \
  --sample-index 0 \
  --max-steps 8
nf-agent benchmark rref \
  --source generated \
  --rows 4 \
  --cols 4 \
  --p 101 \
  --family dense \
  --count 4 \
  --seed-start 0
nf-agent benchmark rref \
  --source shard \
  --data /tmp/rref_8x8_smoke.npz \
  --count 4 \
  --checkpoint /tmp/rref_pivot_smoke_ckpt \
  --max-steps 8
cd lean && lake build
```

Inspect shard metadata:

```bash
python - <<'PY'
import json
import numpy as np

with np.load("/tmp/rref_8x8_smoke.npz", allow_pickle=False) as shard:
    print(json.dumps(json.loads(str(shard["metadata_json"])), indent=2))
PY
```

See `docs/trajectory_shards.md` for the fixed NPZ schema and supported shard
teachers (`leftmost`, `min_fill`).

Check the latest local training checkpoint:

```bash
ls /tmp/rref_pivot_smoke_ckpt
```

The neural rollout command emits JSON with `status`, `success`, `step_count`,
`invalid_action_count`, `masked_action_count`, `invalid_action_breakdown`,
`initial_matrix`, `final_matrix`, replayed `ops`, `final_is_rref`,
`checkpoint_step`, and `modulus`. It reports failed neural rollouts directly;
it does not call the leftmost teacher as fallback.

The RREF benchmark command emits compact JSON with top-level `status`, `source`,
`count`, `rows`, `cols`, `modulus`, and `policies`. The `leftmost` policy always
runs with exact replay and `is_rref_modp`; `neural` is included only when
`--checkpoint` is provided. Per-sample benchmark summaries include trace or step
counts, rank where applicable, replay/RREF checks, fill-in density
(`initial_density`, `final_density`, `max_density`, `fill_in_delta`), invalid
action counts for neural rollout, and wall-clock timings. Matrices and full row
operation traces are omitted from benchmark samples.

Integer HNF uses a row-style convention: nonzero rows precede zero rows, pivot
columns strictly increase, pivots are positive, entries below pivots are zero,
and entries above each pivot lie in `[0, pivot)`. Its replay path supports only
unimodular integer row operations: `swap`, `negate`, and `add(target, source,
scalar)` with distinct target/source rows. Coefficient-growth metrics are exact
integers: `initial_max_abs`, `max_abs_seen`, `initial_bitlength`,
`max_bitlength`, `growth_numerator`, `growth_denominator`, and `step_count`.

## Roadmap

- `v0.2`: fixed-shape NPZ shards, `PivotMLP`, JAX/Flax/Optax imitation
  training, Orbax checkpoints, Grain pipeline.
- `v0.3`: verifier-guided neural rollout, legal action masking, invalid-action
  failure accounting, no hidden fallback.
- `v0.3+`: RREF benchmark suite for generated/shard samples, exact replay
  checks, and fill-in density metrics.
- `v0.4`: HNF training/benchmark integration on top of the integer row-HNF
  environment.
- `v0.5`: SNF certificates with `(D,U,V)` and trace replay.
- `v0.6`: Lean checker for small exported RREF/SNF certificates.
- `v0.7`: paper-style benchmark report.
- `v0.8`: DAgger, policy gradient, and verifier beam search as explicit
  experimental branches.
