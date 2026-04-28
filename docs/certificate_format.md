# Certificate Format

Initial RREF certificates are JSON-compatible records:

```json
{
  "kind": "rref_modp",
  "modulus": 101,
  "shape": [4, 4],
  "input": [[1, 2], [3, 4]],
  "ops": [
    {"kind": "swap", "target": 0, "source": 1},
    {"kind": "scale", "target": 0, "scalar": 51},
    {"kind": "add", "target": 1, "source": 0, "scalar": 98}
  ],
  "final": [[1, 0], [0, 1]],
  "pivots": [{"row": 0, "col": 0}, {"row": 1, "col": 1}]
}
```

Checker obligations:

- Validate modulus and matrix shape.
- Replay every row operation exactly modulo `p`.
- Compare replay output to `final`.
- Verify the final matrix is in RREF modulo `p`.

Integer row-HNF currently has an environment-level trace and replay API, but no
portable JSON certificate schema.

## SNF Certificates

Integer SNF certificates use schema version `snf-certificate-json-v0.1` and
kind `snf_int`. The Python schema and structural validator live in
`nf_agent.certificates`.

```json
{
  "kind": "snf_int",
  "schema_version": "snf-certificate-json-v0.1",
  "shape": [2, 2],
  "input": [[2, 4], [6, 8]],
  "diagonal": [[2, 0], [0, 4]],
  "left_transform": [[1, 0], [0, 1]],
  "right_transform": [[1, 0], [0, 1]],
  "row_ops": [
    {"kind": "swap", "target": 0, "source": 1},
    {"kind": "negate", "target": 1},
    {"kind": "add", "target": 1, "source": 0, "scalar": -3}
  ],
  "col_ops": [
    {"kind": "add", "target": 0, "source": 1, "scalar": 2}
  ]
}
```

Structural validation obligations:

- Validate `shape = [rows, cols]` with nonnegative integer dimensions.
- Validate exact integer matrices only: `input` and `diagonal` are `rows x cols`,
  `left_transform` is `rows x rows`, and `right_transform` is `cols x cols`.
- Reject bools, floats, ragged rows, malformed shapes, unknown fields, and
  unknown operation kinds.
- Verify `diagonal` is rectangular SNF form: off-diagonal entries are zero,
  diagonal entries are nonnegative, nonzero diagonal entries divide the next
  nonzero diagonal entry, and all later diagonal entries stay zero after the
  first zero.
- Bound `row_ops` by the row count and `col_ops` by the column count. Supported
  exact integer operations are `swap`, `negate`, and `add`.

Deferred checker obligations:

- Replay row and column operations.
- Verify `left_transform * input * right_transform = diagonal`.
- Prove or check unimodularity of the recorded transformations.
- Extend the Lean checker after the Python certificate schema stabilizes.
