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

The v0.5 SNF certificate verifier accepts integer certificates with `(D,U,V)`
plus row/column operation traces. It validates the JSON-compatible schema,
replays row ops then column ops, checks recorded transforms against identity
replay, and verifies `U * input * V = D` using exact integer arithmetic.

The v0.6 Lean checker covers small RREF JSON certificates over prime finite
fields and SNF JSON certificates over exact integers. RREF parsing replays
`swap`/`scale`/`add` row ops, checks the final RREF, and requires the supplied
pivot list to match the pivots derived from the final matrix. SNF parsing checks
required fields, replays `swap`/`negate`/`add` row and column ops, checks the
recorded transforms, verifies `U * input * V = D`, and enforces rectangular SNF
diagonal form. Python remains the strict JSON schema authority.

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
nf-agent train status
nf-agent report status
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
nf-agent benchmark hnf \
  --rows 4 \
  --cols 4 \
  --count 4 \
  --density 0.2 \
  --entry-bound 9 \
  --seed-start 0
nf-agent benchmark snf \
  --rows 4 \
  --cols 4 \
  --count 4 \
  --diagonal-factor-bound 5 \
  --row-op-count 2 \
  --col-op-count 2 \
  --op-scalar-bound 3 \
  --seed-start 0
nf-agent report benchmark \
  --out-dir /tmp/nf-v0.9-report
nf-agent benchmark rref \
  --source shard \
  --data /tmp/rref_8x8_smoke.npz \
  --count 4 \
  --checkpoint /tmp/rref_pivot_smoke_ckpt \
  --max-steps 8
nf-agent report rref-certificate \
  --rows 2 \
  --cols 3 \
  --p 5 \
  --seed 0 \
  --teacher leftmost
cd lean && lake build
```

The status commands are informational and do not run training, benchmarks,
report generation, or checker builds. `nf-agent train status` lists the
implemented RREF/HNF training command surface; `nf-agent report status` lists
the report commands and the built-in RREF/HNF/SNF `paper-smoke` benchmark
report coverage.

Inspect shard metadata:

```bash
python - <<'PY'
import json
import numpy as np

with np.load("/tmp/rref_8x8_smoke.npz", allow_pickle=False) as shard:
    print(json.dumps(json.loads(str(shard["metadata_json"])), indent=2))
PY
```

See `docs/trajectory_shards.md` for the fixed NPZ schemas. RREF shards support
teachers (`leftmost`, `min_fill`). HNF shards use `row_hnf` as an explicit
oracle/dataset source and encode integer row operations over a shard-local
scalar vocabulary.

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

The HNF benchmark command emits compact JSON for generated sparse integer
matrices with top-level `status`, `source`, `family`, `count`, `rows`, `cols`,
`density`, `entry_bound`, and `policies`. The canonical baseline policy is
`policies.row_hnf`; top-level `aggregate` and `samples` remain aliases for that
baseline for backward compatibility. Optional learned policies are
`supervised_greedy`, `dagger_greedy`, `actor_critic_greedy`, and `beam`. Each
policy is checked by exact row-operation replay and `is_row_hnf`; failed neural
rollouts are reported directly and are not replaced by `row_hnf`.

Integer HNF uses a row-style convention: nonzero rows precede zero rows, pivot
columns strictly increase, pivots are positive, entries below pivots are zero,
and entries above each pivot lie in `[0, pivot)`. Its replay path supports only
unimodular integer row operations: `swap`, `negate`, and `add(target, source,
scalar)` with distinct target/source rows. Coefficient-growth metrics are exact
integers: `initial_max_abs`, `max_abs_seen`, `initial_bitlength`,
`max_bitlength`, `growth_numerator`, `growth_denominator`, and `step_count`.

The SNF benchmark command emits compact JSON for generated integer certificates
from known rectangular diagonal forms. It does not run an SNF solver. The
`certificate_replay` policy constructs exact row/column operation certificates,
checks replay through `replay_snf_certificate`, verifies transforms and
`U * input * V = D` through `verify_snf_certificate`, and reports compact
density, max-absolute-value, bitlength, operation-count, and timing metrics.
Input matrices, diagonal matrices, transforms, and operation traces are omitted
from samples.

The paper-style benchmark report command writes a reproducible report directory:

```bash
nf-agent report benchmark --out-dir /tmp/nf-v0.9-report
```

Without `--input-json`, it runs the built-in `paper-smoke` suite: generated RREF
dense, sparse, and low-rank samples, generated sparse integer HNF samples, and
generated SNF certificate samples.
With one or more `--input-json PATH` options, it only summarizes existing compact
RREF/HNF/SNF benchmark JSON. The output directory contains `report.md`,
`metrics.json`, and `plots/*.png`. Neural RREF/HNF policy rows appear only when
input benchmark data includes rollout metrics or when run mode receives matching
model metadata and checkpoints.

The v0.8 HNF experiment command builds a reproducible supervised -> DAgger ->
actor-critic -> verifier-beam bundle:

```bash
nf-agent experiment hnf-v08 \
  --out-dir /tmp/nf-v0.8-hnf \
  --samples-per-size 256 \
  --run-seed-count 5 \
  --sizes 4 --sizes 6 --sizes 8 \
  --density 0.2 \
  --entry-bound 5
```

It writes `report.md`, `metrics.json`, plots, per-run benchmark JSON, and a
threshold verdict comparing `dagger_actor_critic_beam` with
`supervised_greedy` per size.

## Roadmap

- `v0.2`: fixed-shape NPZ shards, `PivotMLP`, JAX/Flax/Optax imitation
  training, Orbax checkpoints, Grain pipeline.
- `v0.3`: verifier-guided neural rollout, legal action masking, invalid-action
  failure accounting, no hidden fallback.
- `v0.3+`: RREF benchmark suite for generated/shard samples, exact replay
  checks, and fill-in density metrics.
- `v0.4`: HNF benchmark suite for generated sparse integer matrices, exact
  replay checks, HNF predicates, density metrics, and exact coefficient-growth
  reporting.
- `v0.5`: SNF certificates with `(D,U,V)` and trace replay.
- `v0.6`: Lean checker for small exported RREF JSON certificates and SNF JSON
  certificates with exact replay, transform, equation, and diagonal-form checks.
- `v0.7`: paper-style RREF/HNF benchmark report with Markdown, machine JSON,
  and PNG plots.
- `v0.8`: HNF NPZ shards, supervised imitation, online DAgger,
  actor-critic fine-tuning, verifier beam search, HNF learned-policy benchmark,
  and experiment report bundles.
- `v0.9`: generated SNF certificate benchmark coverage and RREF/HNF/SNF
  benchmark report integration.
