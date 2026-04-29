from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import linen as nn


class PivotMLP(nn.Module):
    rows: int
    cols: int
    max_pivots: int
    max_ops: int
    modulus: int
    hidden_sizes: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, inputs: Any) -> dict[str, Any]:
        x = jnp.asarray(inputs, dtype=jnp.float32)
        x = x.reshape((x.shape[0], self.rows * self.cols))
        for index, hidden_size in enumerate(self.hidden_sizes):
            x = nn.Dense(hidden_size, name=f"hidden_{index}")(x)
            x = nn.relu(x)

        pivot_active_logits = nn.Dense(self.max_pivots, name="pivot_active")(x)
        pivot_col_logits = nn.Dense(self.max_pivots * self.cols, name="pivot_col")(x)
        op_kind_logits = nn.Dense(self.max_ops * 4, name="op_kind")(x)
        op_target_logits = nn.Dense(self.max_ops * self.rows, name="op_target")(x)
        op_source_logits = nn.Dense(self.max_ops * self.rows, name="op_source")(x)
        op_scalar_logits = nn.Dense(self.max_ops * self.modulus, name="op_scalar")(x)

        batch = x.shape[0]
        return {
            "pivot_active_logits": pivot_active_logits.reshape((batch, self.max_pivots)),
            "pivot_col_logits": pivot_col_logits.reshape((batch, self.max_pivots, self.cols)),
            "op_kind_logits": op_kind_logits.reshape((batch, self.max_ops, 4)),
            "op_target_logits": op_target_logits.reshape((batch, self.max_ops, self.rows)),
            "op_source_logits": op_source_logits.reshape((batch, self.max_ops, self.rows)),
            "op_scalar_logits": op_scalar_logits.reshape((batch, self.max_ops, self.modulus)),
        }
