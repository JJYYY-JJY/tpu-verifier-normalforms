# Lean Checker Plan

Scope for `v0.6`:

- Checker-only Lean package.
- Small exported certificates.
- No neural code.
- No floating-point arithmetic.
- RREF replay over finite fields first.
- SNF certificate checks only after the Python schema stabilizes.

The initial Lake workspace is intentionally minimal: it proves the toolchain and
package build path without committing to the final checker API.

