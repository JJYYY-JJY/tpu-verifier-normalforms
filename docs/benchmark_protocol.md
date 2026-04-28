# Benchmark Protocol

Metrics:

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
- Future integer HNF/SNF families with coefficient-growth tracking.

No benchmark may replace a failed neural rollout with a deterministic teacher
without reporting the rollout as failed.
