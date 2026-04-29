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

from nf_agent.data.rref_shards import RREFShardSamples, make_rref_grain_dataset
from nf_agent.models import PivotMLP

MetricMap = dict[str, float]
ArrayBatch = dict[str, Any]


@dataclass(frozen=True)
class TrainConfig:
    data_path: str | Path
    steps: int
    batch_size: int
    learning_rate: float = 0.001
    seed: int = 0
    out_dir: str | Path = Path("results/checkpoints/rref_pivot")
    hidden_sizes: tuple[int, ...] = (256, 256)
    max_to_keep: int = 3


def _validate_config(config: TrainConfig) -> None:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.max_to_keep <= 0:
        raise ValueError("max_to_keep must be positive")
    if not config.hidden_sizes:
        raise ValueError("hidden_sizes must be non-empty")
    if any(hidden_size <= 0 for hidden_size in config.hidden_sizes):
        raise ValueError("hidden_sizes entries must be positive")


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


def _loss_and_metrics(
    model: PivotMLP,
    params: Any,
    batch: ArrayBatch,
) -> tuple[Any, dict[str, Any]]:
    outputs = cast(dict[str, Any], model.apply({"params": params}, batch["inputs"]))

    pivot_active_loss = jnp.mean(
        optax.sigmoid_binary_cross_entropy(
            outputs["pivot_active_logits"],
            jnp.asarray(batch["pivot_active"], dtype=jnp.float32),
        )
    )
    pivot_col_loss = _masked_integer_ce(
        outputs["pivot_col_logits"],
        batch["pivot_cols"],
        batch["pivot_mask"],
    )
    op_kind_loss = jnp.mean(_integer_ce(outputs["op_kind_logits"], batch["op_kind"]))
    op_target_loss = _masked_integer_ce(
        outputs["op_target_logits"],
        batch["op_target"],
        batch["op_mask"],
    )
    op_source_loss = _masked_integer_ce(
        outputs["op_source_logits"],
        batch["op_source"],
        batch["op_source_mask"],
    )
    op_scalar_loss = _masked_integer_ce(
        outputs["op_scalar_logits"],
        batch["op_scalar"],
        batch["op_scalar_mask"],
    )

    loss = (
        pivot_active_loss
        + pivot_col_loss
        + op_kind_loss
        + op_target_loss
        + op_source_loss
        + op_scalar_loss
    )
    return loss, {
        "pivot_active_loss": pivot_active_loss,
        "pivot_col_loss": pivot_col_loss,
        "op_kind_loss": op_kind_loss,
        "op_target_loss": op_target_loss,
        "op_source_loss": op_source_loss,
        "op_scalar_loss": op_scalar_loss,
    }


def _to_float_metrics(loss: Any, metrics: dict[str, Any]) -> MetricMap:
    payload = {"loss": loss, **metrics}
    return {key: float(np.asarray(jax.device_get(value))) for key, value in payload.items()}


def evaluate_rref_pivot_batch(model: PivotMLP, params: Any, batch: ArrayBatch) -> MetricMap:
    loss, metrics = _loss_and_metrics(model, params, _batch_to_jax(batch))
    return _to_float_metrics(loss, metrics)


def _batch_to_jax(batch: ArrayBatch) -> ArrayBatch:
    return {key: jnp.asarray(value) for key, value in batch.items()}


def _model_for_samples(samples: RREFShardSamples, hidden_sizes: tuple[int, ...]) -> PivotMLP:
    return PivotMLP(
        rows=samples.rows,
        cols=samples.cols,
        max_pivots=samples.max_pivots,
        max_ops=samples.max_ops,
        modulus=samples.modulus,
        hidden_sizes=hidden_sizes,
    )


def _initial_state(config: TrainConfig, samples: RREFShardSamples, model: PivotMLP) -> TrainState:
    key = jax.random.PRNGKey(config.seed)
    example = samples[0]["inputs"][np.newaxis, ...]
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


def restore_latest_rref_pivot_checkpoint(config: TrainConfig) -> TrainState:
    _validate_config(config)
    samples = RREFShardSamples(config.data_path)
    model = _model_for_samples(samples, config.hidden_sizes)
    state = _initial_state(config, samples, model)
    manager = _checkpoint_manager(config.out_dir, config.max_to_keep)
    latest_step = manager.latest_step()
    if latest_step is None:
        raise ValueError(f"no checkpoint found in {Path(config.out_dir)}")
    return _restore_if_available(state, manager)


def train_rref_pivot(config: TrainConfig) -> dict[str, Any]:
    _validate_config(config)
    samples = RREFShardSamples(config.data_path)
    model = _model_for_samples(samples, config.hidden_sizes)
    state = _initial_state(config, samples, model)
    manager = _checkpoint_manager(config.out_dir, config.max_to_keep)
    state = _restore_if_available(state, manager)
    initial_params = state.params

    batches = list(
        make_rref_grain_dataset(
            config.data_path,
            batch_size=config.batch_size,
            seed=config.seed,
            drop_remainder=True,
        )
    )
    if not batches:
        batches = list(
            make_rref_grain_dataset(
                config.data_path,
                batch_size=config.batch_size,
                seed=config.seed,
                drop_remainder=False,
            )
        )
    if not batches:
        raise ValueError("no training batches available")

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
        "final_step": int(state.step),
        "final_loss": final_metrics["loss"],
        "metrics": final_metrics,
        "per_head_metrics": per_head_metrics,
        "checkpoint_dir": str(Path(config.out_dir)),
        "latest_step": latest_step,
        "data_schema_version": samples.metadata["schema_version"],
        "parameters_changed": _params_changed(initial_params, state.params),
    }
