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
configuration.

Integer row-HNF is a separate exact environment. Its convention is row-style:
zero rows are below nonzero rows, pivot columns strictly increase, pivot entries
are positive, entries below each pivot are zero, and entries above a pivot are
reduced into `[0, pivot)`. The environment exposes only unimodular integer row
operations: row swaps, row negation, and integer multiples of one distinct row
added to another.

HNF coefficient-growth metrics stay exact. The environment records
`initial_max_abs`, `max_abs_seen`, their integer bitlengths, exact
`growth_numerator = max_abs_seen`, `growth_denominator = max(1,
initial_max_abs)`, and `step_count`. It does not compute floating-point growth
ratios in the verifier path.

SNF replay, Lean checker expansion, HNF training, and HNF rollout remain future
slices.
