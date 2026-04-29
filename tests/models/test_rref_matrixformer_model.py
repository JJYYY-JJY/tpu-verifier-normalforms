from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nf_agent.models import RREFMatrixFormer


def test_rref_matrixformer_outputs_one_step_heads_with_finite_logits() -> None:
    model = RREFMatrixFormer(
        rows=4,
        cols=4,
        modulus=101,
        row_embedding_dim=8,
        col_embedding_dim=8,
        hidden_dim=32,
        layers=1,
        num_heads=1,
    )
    inputs = jnp.zeros((2, 4, 4), dtype=jnp.float32)

    variables = model.init(jax.random.PRNGKey(0), inputs)
    outputs = model.apply(variables, inputs)

    assert outputs["action_kind_logits"].shape == (2, 4)
    assert outputs["action_target_logits"].shape == (2, 4)
    assert outputs["action_source_logits"].shape == (2, 4)
    assert outputs["action_scalar_logits"].shape == (2, 101)
    for logits in outputs.values():
        assert np.all(np.isfinite(np.asarray(logits)))
