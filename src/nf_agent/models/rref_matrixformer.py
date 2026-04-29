from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import linen as nn


class RREFMatrixFormer(nn.Module):
    rows: int
    cols: int
    modulus: int
    row_embedding_dim: int = 32
    col_embedding_dim: int = 32
    hidden_dim: int = 256
    layers: int = 2
    num_heads: int = 4

    @nn.compact
    def __call__(self, state: Any) -> dict[str, Any]:
        x = jnp.asarray(state, dtype=jnp.float32)
        value_features = nn.Dense(self.hidden_dim, name="value_projection")(x[..., None])

        row_ids = jnp.arange(self.rows, dtype=jnp.int32)
        col_ids = jnp.arange(self.cols, dtype=jnp.int32)
        row_features = nn.Embed(
            num_embeddings=self.rows,
            features=self.row_embedding_dim,
            name="row_embedding",
        )(row_ids)
        col_features = nn.Embed(
            num_embeddings=self.cols,
            features=self.col_embedding_dim,
            name="col_embedding",
        )(col_ids)
        row_grid = jnp.broadcast_to(
            row_features[:, None, :],
            (self.rows, self.cols, self.row_embedding_dim),
        )
        col_grid = jnp.broadcast_to(
            col_features[None, :, :],
            (self.rows, self.cols, self.col_embedding_dim),
        )
        position_features = jnp.concatenate([row_grid, col_grid], axis=-1)
        position_features = jnp.broadcast_to(
            position_features[None, ...],
            (
                x.shape[0],
                self.rows,
                self.cols,
                self.row_embedding_dim + self.col_embedding_dim,
            ),
        )

        tokens = jnp.concatenate([value_features, position_features], axis=-1)
        tokens = nn.Dense(self.hidden_dim, name="token_projection")(tokens)
        tokens = tokens.reshape((x.shape[0], self.rows * self.cols, self.hidden_dim))

        for index in range(self.layers):
            residual = tokens
            attention_input = nn.LayerNorm(name=f"attention_norm_{index}")(tokens)
            attention_output = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.hidden_dim,
                out_features=self.hidden_dim,
                name=f"attention_{index}",
            )(attention_input)
            tokens = residual + attention_output

            residual = tokens
            mlp_input = nn.LayerNorm(name=f"mlp_norm_{index}")(tokens)
            mlp = nn.Dense(self.hidden_dim * 4, name=f"mlp_up_{index}")(mlp_input)
            mlp = nn.gelu(mlp)
            mlp = nn.Dense(self.hidden_dim, name=f"mlp_down_{index}")(mlp)
            tokens = residual + mlp

        pooled = jnp.mean(nn.LayerNorm(name="final_norm")(tokens), axis=1)
        return {
            "action_kind_logits": nn.Dense(4, name="action_kind")(pooled),
            "action_target_logits": nn.Dense(self.rows, name="action_target")(pooled),
            "action_source_logits": nn.Dense(self.rows, name="action_source")(pooled),
            "action_scalar_logits": nn.Dense(self.modulus, name="action_scalar")(pooled),
        }
