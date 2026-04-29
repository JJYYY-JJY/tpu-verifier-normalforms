# v6e1 Configs

These YAML files are planned CertiNF-v6e profile specs. They are not consumed
by the current CLI yet.

Current executable configs remain the flat files under `configs/`. The v6e1
files define the intended shape for future commands such as:

- `nf-agent data make-rref-backward-shard`
- `nf-agent data make-rref-state-shard`
- `nf-agent train rref-matrixformer`
- `nf-agent rollout rref-verifier-beam`
- `nf-agent report v6e-profile`

Large outputs referenced by these configs must stay outside git. Commit only
compact report JSON/Markdown, small fixtures, and sanitized config files.
