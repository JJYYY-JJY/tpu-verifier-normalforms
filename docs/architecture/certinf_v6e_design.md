# CertiNF-v6e Design

CertiNF-v6e is the mainline architecture for turning the current
verifier-first normal-form stack into a TPU-native certificate search engine.
The TPU proposes operation traces; exact CPU/Python and Lean checkers decide
whether certificates are valid.

## Objective

Build learned search policies for RREF, HNF, and SNF certificates with this
boundary:

```text
exact generators -> fixed schemas -> TPU scoring/search -> exact replay
-> compact reports -> optional Lean checker samples
```

The system is not allowed to replace failed neural search with teacher output.
Teachers remain oracle, baseline, or dataset source only.

## Compute Boundary

CPU:

- Generate exact RREF/HNF/SNF traces and backward shards.
- Validate schemas, prime moduli, matrix shapes, row/column operation legality,
  and transform dimensions.
- Replay row and column operations exactly.
- Run certificate predicates and compact report aggregation.
- Export small checker-only JSON samples for Lean.

TPU:

- Train step policies using BF16/FP tensor representations derived from
  already-validated exact shards.
- Score legal operation candidates in batches.
- Run device-batched beam/search over fixed-shape state tensors.
- Return candidate operation IDs and scalar vocabulary IDs for CPU replay.

RAM / local storage:

- Hold sharded state/action datasets, beam frontier buffers, and compact
  profile outputs.
- Store large datasets and checkpoints outside git, preferably under `/tmp`,
  mounted storage, or an explicit experiment artifact directory.

Lean:

- Remains checker-only.
- Accepts small exported RREF/SNF JSON samples.
- Replays operations and validates final predicates.
- Does not import neural code, checkpoint code, or large dataset loaders.

## Exactness Contract

- Verifier, replay, certificate, and Lean paths use exact integer or modular
  arithmetic only.
- Floating-point values may appear only after exact shards have been validated
  and only inside model training/scoring/search.
- RREF finite-field moduli must be prime.
- Malformed shapes, illegal row/column operations, bad padding, invalid
  transforms, and non-canonical final forms fail explicitly.
- Broad exception catches around algebraic kernels are forbidden.
- Neural failure must appear as failure JSON, not teacher-substituted success.

## Data Schemas

Existing read-only compatibility:

- `rref-teacher-trajectory-npz-v0.2`

Implemented v1.0-beta1 RREF smoke schemas:

- `rref-backward-trace-npz-v1`
  - `inputs`
  - `finals`
  - `pivots`
  - `ops`
  - `op_mask`
  - `metadata_json`
  - exact replay from `inputs` to `finals` required
- `rref-state-action-npz-v1`
  - `states`
  - `action_kind`
  - `action_target`
  - `action_source`
  - `action_scalar`
  - `stop_label`
  - `legal_kind_mask`
  - `legal_target_mask`
  - `legal_source_mask`
  - `legal_target_source_mask`
  - `legal_scalar_mask`
  - trace-shaped tensors for replay checks
  - `metadata_json`

Implemented storage formats:

- `.npz` eager arrays for small local smoke and fixtures;
- `.zarr` chunked arrays with the same array names and schema metadata for
  larger RREF backward/state-action shards.

HNF and SNF follow the same split:

- Backward trace/certificate shards are exact.
- State/action shards are fixed-shape policy examples.
- TPU-facing features may include modular projections, sign bits, masks,
  density, and bitlength buckets, but exact integer arithmetic stays on CPU.

## RREF v1.0 Mainline

Replace whole-trace `PivotMLP` imitation and per-sample host rollout with:

- backward-generated RREF trace shards from canonical RREF states;
- state/action shard expansion from exact traces;
- MatrixFormer-style single-step policy;
- batched legal masks and beam/search on TPU;
- CPU exact replay and verifier acceptance;
- compact `v6e-profile` reports.

Implemented v1.0-beta1 CLI/script surface:

- `nf-agent profile v6e-status`
- `nf-agent data make-rref-backward-shard` for `.npz` or `.zarr`
- `nf-agent data make-rref-state-shard` for `.npz` or `.zarr`
- `nf-agent train rref-matrixformer`
- `nf-agent rollout rref-matrixformer` as greedy local smoke
- `nf-agent rollout rref-verifier-beam` with exact CPU replay/search acceptance
- `nf-agent report v6e-profile`
- `python scripts/rref_v6e_profile.py`

## HNF v1.1 Mainline

Upgrade row-style HNF v0.8 into low coefficient-growth certificate search:

- backward row-HNF shards from exact unimodular row operations;
- modular projections, sign, zero mask, bitlength bucket, and density features;
- policy heads for `swap`, `negate`, and `add`;
- CPU replay with `replay_integer_row_ops` and `is_row_hnf`;
- reports for validity, step count, bitlength, max absolute value, fill-in,
  certificate size, and wall time.

## SNF v1.2 Mainline

Target certificate search before a general SNF solver:

- backward certificates from diagonal SNF forms;
- row/column operation policy over side, operation kind, target/source, and
  scalar bucket;
- CPU verification of row/column replay, identity transform replay,
  `U * A * V = D`, and diagonal divisibility;
- small RREF/SNF exports for Lean checker samples.

## Pallas v1.3 Boundary

Pallas kernels are optional hot-path experiments after v1.0/v1.1 stabilize.
Candidates:

- batched modular row update;
- legal mask construction;
- beam candidate scoring.

Pallas output must be compared against the pure JAX path for correctness and
speed. It never replaces exact verification.

## Artifact Policy

Commit:

- source code;
- compact JSON/Markdown summaries;
- small fixtures;
- small plots needed by reports;
- checker samples sized for tests.

Do not commit:

- NPZ/Zarr production shards;
- Orbax checkpoints;
- raw XProf traces;
- raw stdout/stderr logs;
- Colab PDFs;
- full matrices or operation traces inside benchmark summaries.
