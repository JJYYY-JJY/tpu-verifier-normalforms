# Lean Checker Plan

Scope for `v0.6`:

- Checker-only Lean package.
- Small exported certificates.
- No neural code.
- No floating-point arithmetic.
- RREF replay over finite fields first.
- SNF certificate checks only after the Python schema stabilizes.

Current v0.6 landing is RREF JSON only. The Lean API is:

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
