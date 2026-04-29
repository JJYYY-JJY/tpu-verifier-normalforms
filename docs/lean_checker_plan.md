# Lean Checker Plan

Scope for `v0.6`:

- Checker-only Lean package.
- Small exported certificates.
- No neural code.
- No floating-point arithmetic.
- RREF replay over finite fields.
- SNF replay and equation checks over exact integers.
- Python remains the strict JSON schema authority; Lean checks required fields
  plus replay/equation/form obligations.

Current v0.6 landing includes RREF JSON and SNF JSON. The RREF Lean API is:

- `parseRREFCertificateJson : String -> Except String RREFCertificate`
- `verifyRREFCertificate : RREFCertificate -> Except String Unit`
- `verifyRREFCertificateJson : String -> Except String Unit`
- `checkRREFCertificate : RREFCertificate -> Bool`
- `checkRREFCertificateJson : String -> Bool`

The checker parses `kind = "rref_modp"`, `modulus`, `shape`, `input`, `ops`,
`final`, and mandatory `pivots`. It rejects non-prime moduli, malformed shapes,
illegal row operations, replay/final mismatches, non-RREF finals, and pivot-list
mismatches. Lake defaults build the checker and the Lean smoke tests through
`cd lean && lake build`.

The SNF Lean API is:

- `parseSNFCertificateJson : String -> Except String SNFCertificate`
- `verifySNFCertificate : SNFCertificate -> Except String Unit`
- `verifySNFCertificateJson : String -> Except String Unit`
- `checkSNFCertificate : SNFCertificate -> Bool`
- `checkSNFCertificateJson : String -> Bool`

The SNF checker parses `kind = "snf_int"`,
`schema_version = "snf-certificate-json-v0.1"`, `shape`, `input`,
`diagonal`, `left_transform`, `right_transform`, `row_ops`, and `col_ops`.
Supported exact integer row and column operations are `swap`, `negate`, and
`add(target, source, scalar)`. Verification checks matrix shapes, replays row
ops then column ops to match `diagonal`, replays row and column ops on identity
matrices to match the recorded transforms, verifies
`left_transform * input * right_transform = diagonal`, and enforces rectangular
SNF diagonal form.
