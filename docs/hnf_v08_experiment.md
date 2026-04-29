# HNF v0.8 Experiment

`nf-agent experiment hnf-v08` is the reproducible HNF learning pipeline:

1. Generate sparse integer HNF shards with `row_hnf` as the explicit oracle.
2. Train `HNFPolicyMLP` by supervised imitation.
3. Run online DAgger: policy rollout, oracle continuations on visited non-HNF
   states, aggregate shard rebuild, retrain.
4. Run actor-critic fine-tuning from the DAgger checkpoint with exact rollout
   rewards.
5. Benchmark `row_hnf`, `supervised_greedy`, `dagger_greedy`,
   `actor_critic_greedy`, and verifier `beam`.
6. Write `report.md`, `metrics.json`, plots, per-run benchmark JSON, and a
   threshold verdict.

Default paper-scale command:

```bash
nf-agent experiment hnf-v08 \
  --out-dir /tmp/nf-v0.8-hnf \
  --samples-per-size 256 \
  --run-seed-count 5 \
  --sizes 4 --sizes 6 --sizes 8 \
  --density 0.2 \
  --entry-bound 5
```

Threshold:

- Candidate: `dagger_actor_critic_beam`
- Baseline: `supervised_greedy`
- Pass condition per size:
  `success_rate >= baseline + 0.05` and
  `mean_step_count <= baseline * 0.90`

If the threshold fails, the command still writes the full artifact bundle and
emits `"status": "failed_threshold"`. This is an experimental verdict, not a
fallback path.

Exactness boundaries:

- `row_hnf` is used only as oracle, baseline, or dataset source.
- Greedy and beam rollout never call `row_hnf`.
- Replay and predicates use exact integer row operations and `is_row_hnf`.
- Float normalization is confined to model inputs after shard validation.
