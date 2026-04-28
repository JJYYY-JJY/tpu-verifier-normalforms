# Benchmark Protocol

`nf-agent benchmark rref` is the v0.3+ finite-field RREF benchmark suite. It
runs the exact `leftmost` baseline for every sample. When a checkpoint is
provided, it also runs neural rollout on the same samples. Neural failures are
reported as neural failures; they are not replaced by teacher traces.

`nf-agent benchmark hnf` is the v0.4 integer row-HNF benchmark suite. It only
uses generated sparse integer matrices in this slice. Every sample runs
`row_hnf`, exact integer trace replay, and `is_row_hnf`.

## Sources

- `source="generated"`: generate dense, sparse, or low-rank matrices from
  `rows`, `cols`, `count`, `modulus`, `family`, and `seed_start`.
- `source="shard"`: read exact `inputs` from an RREF `.npz` shard. `count` is an
  optional limit; omitted means all shard inputs.
- For neural generated benchmarks, `model_data_path` is required and its
  rows/cols/modulus must match the generated benchmark. For neural shard
  benchmarks, `data_path` is the default model metadata shard.
- `nf-agent benchmark hnf`: generate sparse integer matrices from `rows`,
  `cols`, `count`, `density`, `entry_bound`, and `seed_start`.

## Fill-In

Fill-in density is the fraction of nonzero entries after normalizing a matrix
modulo `p`. For a row-operation trace, the density profile contains the initial
state and one state after each exact replayed row operation.

Per-sample compact summaries record:

- `initial_density`: density before replay.
- `final_density`: density after the final operation.
- `max_density`: maximum density seen over the profile.
- `fill_in_delta`: `max_density - initial_density`.

For HNF, density is the fraction of exact nonzero integer entries. This density
is a benchmark metric only; verifier replay and predicates remain exact integer
arithmetic.

## JSON Schema

Top-level fields:

- `status`: `ok`.
- `source`: `generated` or `shard`.
- `count`: actual number of benchmarked samples.
- `rows`, `cols`, and `modulus`.
- `policies`: per-policy results keyed by `leftmost` and, when requested,
  `neural`.

Each policy contains:

- `aggregate`: `sample_count`, `success_count`, `success_rate`,
  `status_counts`, and mean metrics.
- `samples`: compact per-sample summaries. Matrices and row-operation traces are
  intentionally omitted.

Leftmost sample fields:

- `sample_index` and optional `seed`.
- `status`, `success`, `trace_length`, and `rank`.
- `replay_ok` and `final_is_rref`.
- Fill-in density fields.
- `wall_time_seconds`, `teacher_wall_time_seconds`,
  `replay_wall_time_seconds`, and `predicate_wall_time_seconds`.

Neural sample fields:

- `sample_index` and optional `seed`.
- `status`: `success` or `max_steps_exceeded`.
- `success`, `step_count`, `invalid_action_count`,
  `masked_action_count`, and `invalid_action_breakdown`.
- `checkpoint_step`, `replay_ok`, and `final_is_rref`.
- Fill-in density fields.
- `wall_time_seconds`, `rollout_wall_time_seconds`,
  `replay_wall_time_seconds`, and `predicate_wall_time_seconds`.

HNF top-level fields:

- `status`: `ok`.
- `source`: `generated`.
- `family`: `sparse_integer`.
- `count`, `rows`, `cols`, `density`, `entry_bound`, and `seed_start`.
- `aggregate`: `sample_count`, `success_count`, `success_rate`,
  `status_counts`, mean trace/density/time metrics, and exact maxima for
  coefficient-growth metrics.
- `samples`: compact per-sample summaries. Matrices and row-operation traces are
  intentionally omitted.

HNF sample fields:

- `sample_index` and `seed`.
- `status`, `success`, and `trace_length`.
- `replay_ok` and `final_is_hnf`.
- Fill-in density fields.
- Exact coefficient-growth fields: `initial_max_abs`, `max_abs_seen`,
  `initial_bitlength`, `max_bitlength`, `growth_numerator`,
  `growth_denominator`, and `step_count`.
- `wall_time_seconds`, `hnf_wall_time_seconds`, `replay_wall_time_seconds`,
  and `predicate_wall_time_seconds`.

## Metrics

- Verification success rate.
- Invalid action count.
- Masked action count.
- Invalid action breakdown by `op_kind`, `op_source`, and `op_scalar`.
- Trace length.
- Pivot count and rank.
- Fill-in density by step.
- Wall-clock time for teacher, rollout, replay, and predicate checks.

For `nf-agent rollout rref-neural`, each JSON result records:

- `status`: `success` or `max_steps_exceeded`.
- `success`: boolean mirror of `status == "success"`.
- `step_count`: exact row operations replayed.
- `invalid_action_count` and `masked_action_count`.
- `invalid_action_breakdown`.
- `initial_matrix`, `final_matrix`, replayed `ops`, and `final_is_rref`.
- `checkpoint_step` and `modulus`.

Matrix families:

- Dense uniform matrices over `F_p`.
- Sparse Bernoulli-support matrices with nonzero field values.
- Low-rank products `A * B mod p`.
- Sparse integer matrices with Bernoulli support and selected entries sampled
  from `[-entry_bound, entry_bound] \ {0}`. The default `entry_bound` is `9`.
- Future SNF families and HNF training/rollout benchmark reports.

No benchmark may replace a failed neural rollout with a deterministic teacher
without reporting the rollout as failed.
