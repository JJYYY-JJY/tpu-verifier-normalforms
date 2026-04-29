import jax
import jax.numpy as jnp

from nf_agent.models import HNFPolicyMLP


def test_hnf_policy_mlp_outputs_all_policy_and_value_heads() -> None:
    model = HNFPolicyMLP(
        rows=3,
        cols=4,
        max_ops=7,
        scalar_vocab_size=5,
        hidden_sizes=(16,),
    )
    inputs = jnp.ones((2, 3, 4), dtype=jnp.float32)

    variables = model.init(jax.random.PRNGKey(0), inputs)
    outputs = model.apply(variables, inputs)

    assert outputs["op_kind_logits"].shape == (2, 7, 4)
    assert outputs["op_target_logits"].shape == (2, 7, 3)
    assert outputs["op_source_logits"].shape == (2, 7, 3)
    assert outputs["op_scalar_logits"].shape == (2, 7, 5)
    assert outputs["value"].shape == (2,)
