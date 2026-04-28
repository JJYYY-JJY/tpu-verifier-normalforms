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

