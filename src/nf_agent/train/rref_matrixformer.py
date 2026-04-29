from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore[import-untyped]
import orbax.checkpoint as ocp  # type: ignore[import-untyped]
from flax.training.train_state import TrainState

from nf_agent.data.rref_state_shards import (
    RREFStateActionSamples,
    make_rref_state_action_grain_dataset,
)
from nf_agent.models import RREFMatrixFormer

MetricMap = dict[str, float]
ArrayBatch = dict[str, Any]


@dataclass(frozen=True)
class RREFMatrixFormerTrainConfig:
    data_path: str | Path
    steps: int
    batch_size: int
    learning_rate: float = 0.001
    seed: int = 0
    out_dir: str | Path = Path("results/checkpoints/rref_matrixformer")
    row_embedding_dim: int = 32
    col_embedding_dim: int = 32
    hidden_dim: int = 256
    layers: int = 2
    num_heads: int = 4
    max_to_keep: int = 3
    checkpoint_every: int = 1


def _validate_config(config: RREFMatrixFormerTrainConfig) -> None:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.row_embedding_dim <= 0:
        raise ValueError("row_embedding_dim must be positive")
    if config.col_embedding_dim <= 0:
        raise ValueError("col_embedding_dim must be positive")
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if config.layers <= 0:
        raise ValueError("layers must be positive")
    if config.num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if config.hidden_dim % config.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")
    if config.max_to_keep <= 0:
        raise ValueError("max_to_keep must be positive")
    if config.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")


def _masked_logits(logits: Any, legal_mask: Any) -> Any:
    return jnp.where(jnp.asarray(legal_mask, dtype=jnp.bool_), logits, jnp.asarray(-1.0e9))


def _masked_mean(values: Any, mask: Any) -> Any:
    mask_f = jnp.asarray(mask, dtype=jnp.float32)
    values_f = jnp.asarray(values, dtype=jnp.float32)
    return jnp.sum(values_f * mask_f) / jnp.maximum(jnp.sum(mask_f), 1.0)


def _integer_ce(logits: Any, labels: Any) -> Any:
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=jnp.asarray(logits),
        labels=jnp.asarray(labels, dtype=jnp.int32),
    )


def _masked_integer_ce(logits: Any, labels: Any, mask: Any) -> Any:
    mask_jax = jnp.asarray(mask, dtype=jnp.bool_)
    safe_labels = jnp.where(mask_jax, jnp.asarray(labels, dtype=jnp.int32), 0)
    return _masked_mean(_integer_ce(logits, safe_labels), mask_jax)


def _source_legal_mask_for_targets(batch: ArrayBatch) -> Any:
    target_labels = jnp.asarray(batch["action_target"], dtype=jnp.int32)
    valid_target = target_labels >= 0
    safe_targets = jnp.where(valid_target, target_labels, 0)
    target_source_mask = jnp.asarray(batch["legal_target_source_mask"], dtype=jnp.bool_)
    selected = jnp.take_along_axis(target_source_mask, safe_targets[:, None, None], axis=1)
    selected = selected[:, 0, :]
    return selected & jnp.asarray(batch["legal_source_mask"], dtype=jnp.bool_)


def _loss_and_metrics(
    model: RREFMatrixFormer,
    params: Any,
    batch: ArrayBatch,
) -> tuple[Any, dict[str, Any]]:
    outputs = cast(dict[str, Any], model.apply({"params": params}, batch["state"]))

    action_kind = jnp.asarray(batch["action_kind"], dtype=jnp.int32)
    non_stop_mask = action_kind != 0
    source_mask = (action_kind == 1) | (action_kind == 3)
    scalar_mask = (action_kind == 2) | (action_kind == 3)

    action_kind_loss = jnp.mean(
        _integer_ce(
            _masked_logits(outputs["action_kind_logits"], batch["legal_kind_mask"]),
            action_kind,
        )
    )
    action_target_loss = _masked_integer_ce(
        _masked_logits(outputs["action_target_logits"], batch["legal_target_mask"]),
        batch["action_target"],
        non_stop_mask,
    )
    action_source_loss = _masked_integer_ce(
        _masked_logits(outputs["action_source_logits"], _source_legal_mask_for_targets(batch)),
        batch["action_source"],
        source_mask,
    )
    action_scalar_loss = _masked_integer_ce(
        _masked_logits(outputs["action_scalar_logits"], batch["legal_scalar_mask"]),
        batch["action_scalar"],
        scalar_mask,
    )

    loss = action_kind_loss + action_target_loss + action_source_loss + action_scalar_loss
    return loss, {
        "action_kind_loss": action_kind_loss,
        "action_target_loss": action_target_loss,
        "action_source_loss": action_source_loss,
        "action_scalar_loss": action_scalar_loss,
    }


def _to_float_metrics(loss: Any, metrics: dict[str, Any]) -> MetricMap:
    payload = {"loss": loss, **metrics}
    return {key: float(np.asarray(jax.device_get(value))) for key, value in payload.items()}


def _batch_to_jax(batch: ArrayBatch) -> ArrayBatch:
    return {key: jnp.asarray(value) for key, value in batch.items()}


def _model_for_samples(
    samples: RREFStateActionSamples,
    config: RREFMatrixFormerTrainConfig,
) -> RREFMatrixFormer:
    return RREFMatrixFormer(
        rows=samples.rows,
        cols=samples.cols,
        modulus=samples.modulus,
        row_embedding_dim=config.row_embedding_dim,
        col_embedding_dim=config.col_embedding_dim,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        num_heads=config.num_heads,
    )


def _initial_state(
    config: RREFMatrixFormerTrainConfig,
    samples: RREFStateActionSamples,
    model: RREFMatrixFormer,
) -> TrainState:
    key = jax.random.PRNGKey(config.seed)
    example = samples[0]["state"][np.newaxis, ...]
    variables = model.init(key, jnp.asarray(example, dtype=jnp.float32))
    return cast(
        TrainState,
        TrainState.create(  # type: ignore[no-untyped-call]
            apply_fn=model.apply,
            params=variables["params"],
            tx=optax.adam(config.learning_rate),
        ),
    )


def _checkpoint_manager(out_dir: str | Path, max_to_keep: int) -> ocp.CheckpointManager:
    return ocp.CheckpointManager(
        Path(out_dir),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=True,
            enable_async_checkpointing=False,
        ),
    )


def _restore_if_available(state: TrainState, manager: ocp.CheckpointManager) -> TrainState:
    latest_step = manager.latest_step()
    if latest_step is None:
        return state
    restored = manager.restore(latest_step, args=ocp.args.StandardRestore(state))
    if not isinstance(restored, TrainState):
        raise TypeError("restored checkpoint is not a TrainState")
    return restored


def _params_changed(before: Any, after: Any) -> bool:
    comparisons = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(
            lambda left, right: np.asarray(left != right).any(),
            before,
            after,
        )
    )
    return any(bool(value) for value in comparisons)


def _training_batches(config: RREFMatrixFormerTrainConfig) -> list[ArrayBatch]:
    batches = list(
        make_rref_state_action_grain_dataset(
            config.data_path,
            batch_size=config.batch_size,
            seed=config.seed,
            drop_remainder=True,
        )
    )
    if not batches:
        batches = list(
            make_rref_state_action_grain_dataset(
                config.data_path,
                batch_size=config.batch_size,
                seed=config.seed,
                drop_remainder=False,
            )
        )
    if not batches:
        raise ValueError("no training batches available")
    return batches


def evaluate_rref_matrixformer_batch(
    model: RREFMatrixFormer,
    params: Any,
    batch: ArrayBatch,
) -> MetricMap:
    loss, metrics = _loss_and_metrics(model, params, _batch_to_jax(batch))
    return _to_float_metrics(loss, metrics)


def restore_latest_rref_matrixformer_checkpoint(
    config: RREFMatrixFormerTrainConfig,
) -> TrainState:
    _validate_config(config)
    samples = RREFStateActionSamples(config.data_path)
    model = _model_for_samples(samples, config)
    state = _initial_state(config, samples, model)
    manager = _checkpoint_manager(config.out_dir, config.max_to_keep)
    latest_step = manager.latest_step()
    if latest_step is None:
        raise ValueError(f"no checkpoint found in {Path(config.out_dir)}")
    return _restore_if_available(state, manager)


def train_rref_matrixformer(config: RREFMatrixFormerTrainConfig) -> dict[str, Any]:
    _validate_config(config)
    samples = RREFStateActionSamples(config.data_path)
    model = _model_for_samples(samples, config)
    state = _initial_state(config, samples, model)
    manager = _checkpoint_manager(config.out_dir, config.max_to_keep)
    state = _restore_if_available(state, manager)
    initial_params = state.params
    batches = _training_batches(config)

    def train_step(train_state: TrainState, batch: ArrayBatch) -> tuple[TrainState, dict[str, Any]]:
        def loss_fn(params: Any) -> tuple[Any, dict[str, Any]]:
            return _loss_and_metrics(model, params, batch)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        next_state = cast(
            TrainState,
            train_state.apply_gradients(grads=grads),  # type: ignore[no-untyped-call]
        )
        return next_state, {"loss": loss, **metrics}

    jitted_train_step = jax.jit(train_step)
    final_metrics: MetricMap | None = None
    for step_offset in range(config.steps):
        batch = _batch_to_jax(batches[step_offset % len(batches)])
        state, raw_metrics = jitted_train_step(state, batch)
        final_metrics = {
            key: float(np.asarray(jax.device_get(value))) for key, value in raw_metrics.items()
        }
        is_final_step = step_offset == config.steps - 1
        if int(state.step) % config.checkpoint_every == 0 or is_final_step:
            manager.save(int(state.step), args=ocp.args.StandardSave(state))

    manager.wait_until_finished()
    latest_step = manager.latest_step()
    if final_metrics is None:
        raise RuntimeError("training loop did not run")

    per_head_metrics = {
        key: value
        for key, value in final_metrics.items()
        if key.endswith("_loss") and key != "loss"
    }
    return {
        "status": "ok",
        "model": "rref-matrixformer",
        "final_step": int(state.step),
        "final_loss": final_metrics["loss"],
        "metrics": final_metrics,
        "per_head_metrics": per_head_metrics,
        "checkpoint_dir": str(Path(config.out_dir)),
        "latest_step": latest_step,
        "checkpoint_every": config.checkpoint_every,
        "data_schema_version": samples.metadata["schema_version"],
        "parameters_changed": _params_changed(initial_params, state.params),
    }
