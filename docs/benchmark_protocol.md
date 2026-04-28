# Benchmark Protocol

Metrics:

- Verification success rate.
- Invalid action count.
- Trace length.
- Pivot count and rank.
- Fill-in density by step.
- Wall-clock time for teacher, rollout, replay, and predicate checks.

Matrix families:

- Dense uniform matrices over `F_p`.
- Sparse Bernoulli-support matrices with nonzero field values.
- Low-rank products `A * B mod p`.
- Future integer HNF/SNF families with coefficient-growth tracking.

No benchmark may replace a failed neural rollout with a deterministic teacher
without reporting the rollout as failed.

