# Design

## Seven Layers

1. Algebra environments: exact state transitions and legality checks.
2. Teachers: deterministic oracles that emit fully replayable traces.
3. Data: fixed-schema trajectory shards derived from teacher traces.
4. Models: masked policies that score legal pivot or operation actions.
5. Training: imitation learning and later verifier-guided fine-tuning.
6. Rollout: neural proposals checked step-by-step with no hidden fallback.
7. Certificates: portable traces replayable by Python and future Lean checkers.

The first implemented environment is RREF over `F_p`, with `F_101` as the smoke
configuration. Integer HNF/SNF are later environments, not mixed into the initial
finite-field slice.

