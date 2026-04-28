# AGENTS.md

This repository is a verifier-first normal-forms research codebase.

Rules:

- Exact algebra only in environments, certificate replay, and verifiers.
- Never use floating point in a verifier path.
- Reject malformed matrices, non-prime finite-field moduli, and illegal row
  operations with explicit exceptions.
- Do not catch broad exceptions around algebraic kernels.
- Do not implement silent deterministic fallback in rollout code.
- Deterministic teachers are allowed only when explicitly selected as oracle,
  baseline, or dataset source.
- Keep Sage outside `.venv`; use the `nf-sage` micromamba environment.
- Keep Lean checker code small and checker-only unless the roadmap explicitly
  expands it.

Required gates before claiming completion:

```bash
source .venv/bin/activate
python -m pip check
ruff check .
mypy src
pytest
nf-agent --help
cd lean && lake build
```

