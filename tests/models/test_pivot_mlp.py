from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nf_agent.data.rref_shards import make_rref_grain_dataset, write_rref_shard
from nf_agent.models import PivotMLP
from nf_agent.train import evaluate_rref_pivot_batch

CONFIG = Path("configs/rref_8x8_mod101.yaml")


def _batch(tmp_path: Path) -> dict[str, np.ndarray]:
    shard_path = tmp_path / "rref_model.npz"
    write_rref_shard(config_path=CONFIG, count=4, seed_start=0, out_path=shard_path)
    dataset = make_rref_grain_dataset(shard_path, batch_size=2, seed=0)
    return next(iter(dataset))


def test_pivot_mlp_outputs_all_training_heads_with_expected_shapes(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    model = PivotMLP(rows=8, cols=8, max_pivots=8, max_ops=72, modulus=101)
    variables = model.init(jax.random.PRNGKey(0), jnp.asarray(batch["inputs"]))

    outputs = model.apply(variables, jnp.asarray(batch["inputs"]))

    assert outputs["pivot_active_logits"].shape == (2, 8)
    assert outputs["pivot_col_logits"].shape == (2, 8, 8)
    assert outputs["op_kind_logits"].shape == (2, 72, 4)
    assert outputs["op_target_logits"].shape == (2, 72, 8)
    assert outputs["op_source_logits"].shape == (2, 72, 8)
    assert outputs["op_scalar_logits"].shape == (2, 72, 101)


def test_rref_pivot_loss_is_finite_and_ignores_padded_labels(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    batch["pivot_mask"][:] = False
    batch["pivot_cols"][:] = -999
    batch["op_mask"][:] = False
    batch["op_target"][:] = -999
    batch["op_source"][:] = -999
    batch["op_scalar"][:] = -999
    batch["op_source_mask"][:] = False
    batch["op_scalar_mask"][:] = False
    model = PivotMLP(rows=8, cols=8, max_pivots=8, max_ops=72, modulus=101)
    variables = model.init(jax.random.PRNGKey(0), jnp.asarray(batch["inputs"]))

    metrics = evaluate_rref_pivot_batch(model, variables["params"], batch)

    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["pivot_active_loss"])
    assert metrics["pivot_col_loss"] == 0.0
    assert metrics["op_kind_loss"] == 0.0
    assert metrics["op_target_loss"] == 0.0
    assert metrics["op_source_loss"] == 0.0
    assert metrics["op_scalar_loss"] == 0.0
