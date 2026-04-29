from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore[import-untyped]
import orbax.checkpoint as ocp  # type: ignore[import-untyped]
from flax.training.train_state import TrainState

from nf_agent.data.hnf_shards import (
    HNFShardSamples,
    HNFTrajectory,
    hnf_trajectories_from_shard,
    make_hnf_grain_dataset,
    write_hnf_shard_from_trajectories,
)
from nf_agent.env.elementary_ops import Matrix
from nf_agent.env.hnf_int import is_row_hnf, row_hnf
from nf_agent.models import HNFPolicyMLP

MetricMap = dict[str, float]
ArrayBatch = dict[str, Any]


@dataclass(frozen=True)
class HNFTrainConfig:
    data_path: str | Path
    steps: int
    batch_size: int
    learning_rate: float = 0.001
    seed: int = 0
    out_dir: str | Path = Path("results/checkpoints/hnf_policy")
    hidden_sizes: tuple[int, ...] = (256, 256)
    max_to_keep: int = 3


@dataclass(frozen=True)
class HNFDaggerConfig:
    data_path: str | Path
    iterations: int
    train_steps: int
    batch_size: int
    learning_rate: float = 0.001
    seed: int = 0
    out_dir: str | Path = Path("results/checkpoints/hnf_dagger")
    hidden_sizes: tuple[int, ...] = (256, 256)
    rollout_sample_count: int = 16
    rollout_max_steps: int | None = None


@dataclass(frozen=True)
class HNFActorCriticConfig:
    data_path: str | Path
    checkpoint_dir: str | Path
    steps: int
    batch_size: int
    learning_rate: float = 0.0005
    seed: int = 0
    out_dir: str | Path = Path("results/checkpoints/hnf_actor_critic")
    hidden_sizes: tuple[int, ...] = (256, 256)
    max_to_keep: int = 3
    rollout_max_steps: int | None = None
    value_loss_weight: float = 0.5
    entropy_weight: float = 0.01


def _validate_train_config(config: HNFTrainConfig) -> None:
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


def _validate_dagger_config(config: HNFDaggerConfig) -> None:
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.rollout_sample_count <= 0:
        raise ValueError("rollout_sample_count must be positive")
    if config.rollout_max_steps is not None and config.rollout_max_steps < 0:
        raise ValueError("rollout_max_steps must be nonnegative")


def _validate_actor_critic_config(config: HNFActorCriticConfig) -> None:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.max_to_keep <= 0:
        raise ValueError("max_to_keep must be positive")
    if config.rollout_max_steps is not None and config.rollout_max_steps < 0:
        raise ValueError("rollout_max_steps must be nonnegative")
    if config.value_loss_weight < 0.0:
        raise ValueError("value_loss_weight must be nonnegative")
    if config.entropy_weight < 0.0:
        raise ValueError("entropy_weight must be nonnegative")


def _model_for_samples(samples: HNFShardSamples, hidden_sizes: tuple[int, ...]) -> HNFPolicyMLP:
    scalar_vocab_size = max(1, samples.scalar_vocab_size)
    return HNFPolicyMLP(
        rows=samples.rows,
        cols=samples.cols,
        max_ops=samples.max_ops,
        scalar_vocab_size=scalar_vocab_size,
        hidden_sizes=hidden_sizes,
    )


def _initial_state(
    config: HNFTrainConfig | HNFActorCriticConfig,
    samples: HNFShardSamples,
    model: HNFPolicyMLP,
) -> TrainState:
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


def _ensure_batched(batch: ArrayBatch) -> ArrayBatch:
    result: ArrayBatch = {}
    for key, value in batch.items():
        array = np.asarray(value)
        if key == "inputs" and array.ndim == 2:
            array = array[np.newaxis, ...]
        elif key != "inputs" and array.ndim == 0:
            array = array[np.newaxis]
        elif key != "inputs" and array.ndim == 1:
            array = array[np.newaxis, ...]
        result[key] = array
    return result


def _batch_to_jax(batch: ArrayBatch) -> ArrayBatch:
    return {key: jnp.asarray(value) for key, value in _ensure_batched(batch).items()}


def _loss_components(
    model: HNFPolicyMLP,
    params: Any,
    batch: ArrayBatch,
) -> tuple[Any, dict[str, Any]]:
    outputs = cast(dict[str, Any], model.apply({"params": params}, batch["inputs"]))
    op_kind_loss = _masked_integer_ce(
        outputs["op_kind_logits"],
        batch["op_kind"],
        batch["op_mask"],
    )
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
    value_loss = jnp.mean(
        jnp.square(outputs["value"] - jnp.asarray(batch["value_target"], dtype=jnp.float32))
    )
    supervised_loss = op_kind_loss + op_target_loss + op_source_loss + op_scalar_loss
    return supervised_loss + value_loss, {
        "supervised_loss": supervised_loss,
        "op_kind_loss": op_kind_loss,
        "op_target_loss": op_target_loss,
        "op_source_loss": op_source_loss,
        "op_scalar_loss": op_scalar_loss,
        "value_loss": value_loss,
    }


def _to_float_metrics(loss: Any, metrics: dict[str, Any]) -> MetricMap:
    payload = {"loss": loss, **metrics}
    return {key: float(np.asarray(jax.device_get(value))) for key, value in payload.items()}


def evaluate_hnf_policy_batch(config: HNFTrainConfig, params: Any, batch: ArrayBatch) -> MetricMap:
    samples = HNFShardSamples(config.data_path)
    model = _model_for_samples(samples, config.hidden_sizes)
    loss, metrics = _loss_components(model, params, _batch_to_jax(batch))
    return _to_float_metrics(loss, metrics)


def restore_latest_hnf_policy_checkpoint(config: HNFTrainConfig) -> TrainState:
    _validate_train_config(config)
    samples = HNFShardSamples(config.data_path)
    model = _model_for_samples(samples, config.hidden_sizes)
    state = _initial_state(config, samples, model)
    manager = _checkpoint_manager(config.out_dir, config.max_to_keep)
    latest_step = manager.latest_step()
    if latest_step is None:
        raise ValueError(f"no checkpoint found in {Path(config.out_dir)}")
    return _restore_if_available(state, manager)


def _training_batches(config: HNFTrainConfig) -> list[ArrayBatch]:
    batches = list(
        make_hnf_grain_dataset(
            config.data_path,
            batch_size=config.batch_size,
            seed=config.seed,
            drop_remainder=True,
        )
    )
    if not batches:
        batches = list(
            make_hnf_grain_dataset(
                config.data_path,
                batch_size=config.batch_size,
                seed=config.seed,
                drop_remainder=False,
            )
        )
    if not batches:
        raise ValueError("no training batches available")
    return batches


def train_hnf_policy(config: HNFTrainConfig) -> dict[str, Any]:
    _validate_train_config(config)
    samples = HNFShardSamples(config.data_path)
    model = _model_for_samples(samples, config.hidden_sizes)
    state = _initial_state(config, samples, model)
    manager = _checkpoint_manager(config.out_dir, config.max_to_keep)
    state = _restore_if_available(state, manager)
    initial_params = state.params
    batches = _training_batches(config)

    def train_step(train_state: TrainState, batch: ArrayBatch) -> tuple[TrainState, dict[str, Any]]:
        def loss_fn(params: Any) -> tuple[Any, dict[str, Any]]:
            return _loss_components(model, params, batch)

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
        "scalar_vocab_size": samples.scalar_vocab_size,
        "parameters_changed": _params_changed(initial_params, state.params),
    }


def _raw_matrices(path: str | Path, limit: int) -> list[Matrix]:
    with np.load(path, allow_pickle=False) as shard:
        inputs = np.asarray(shard["inputs"][:limit], dtype=np.int64)
    return cast(list[Matrix], inputs.tolist())


def _oracle_continuations_from_rollouts(
    *,
    data_path: str | Path,
    checkpoint_dir: str | Path,
    hidden_sizes: tuple[int, ...],
    rollout_sample_count: int,
    rollout_max_steps: int | None,
) -> list[HNFTrajectory]:
    from nf_agent.rollout import (
        HNFRolloutConfig,
        load_hnf_policy_runtime,
        rollout_hnf_policy_with_runtime,
    )

    samples = HNFShardSamples(data_path)
    limit = min(rollout_sample_count, len(samples))
    rollout_config = HNFRolloutConfig(
        data_path=data_path,
        checkpoint_dir=checkpoint_dir,
        max_steps=rollout_max_steps,
        hidden_sizes=hidden_sizes,
    )
    runtime = load_hnf_policy_runtime(rollout_config)
    extras: list[HNFTrajectory] = []
    for matrix in _raw_matrices(data_path, limit):
        result = rollout_hnf_policy_with_runtime(runtime, rollout_config, matrix)
        for visited in result.visited_matrices:
            if is_row_hnf(visited):
                continue
            oracle = row_hnf(visited)
            extras.append(
                HNFTrajectory(
                    input_matrix=visited,
                    final_matrix=oracle.final_matrix,
                    ops=tuple(oracle.ops),
                )
            )
    return extras


def train_hnf_dagger(config: HNFDaggerConfig) -> dict[str, Any]:
    _validate_dagger_config(config)
    base_samples = HNFShardSamples(config.data_path)
    trajectories = hnf_trajectories_from_shard(config.data_path)
    initial_count = len(trajectories)
    current_data_path = Path(config.data_path)
    out_dir = Path(config.out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    aggregate_path = out_dir / "aggregate.npz"
    oracle_count = 0
    train_result: dict[str, Any] | None = None

    for iteration in range(config.iterations):
        iteration_checkpoint_dir = out_dir / f"iter_{iteration}_checkpoints"
        train_result = train_hnf_policy(
            HNFTrainConfig(
                data_path=current_data_path,
                steps=config.train_steps,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                seed=config.seed + iteration,
                out_dir=iteration_checkpoint_dir,
                hidden_sizes=config.hidden_sizes,
            )
        )
        extras = _oracle_continuations_from_rollouts(
            data_path=current_data_path,
            checkpoint_dir=iteration_checkpoint_dir,
            hidden_sizes=config.hidden_sizes,
            rollout_sample_count=config.rollout_sample_count,
            rollout_max_steps=config.rollout_max_steps,
        )
        trajectories.extend(extras)
        oracle_count += len(extras)
        write_hnf_shard_from_trajectories(
            trajectories,
            config_payload=cast(Mapping[str, Any], base_samples.metadata["config"]),
            out_path=aggregate_path,
            seed_start=0,
        )
        current_data_path = aggregate_path

    if train_result is None:
        raise RuntimeError("DAgger loop did not run")
    train_result = train_hnf_policy(
        HNFTrainConfig(
            data_path=current_data_path,
            steps=config.train_steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            seed=config.seed + config.iterations,
            out_dir=checkpoint_dir,
            hidden_sizes=config.hidden_sizes,
        )
    )
    return {
        "status": "ok",
        "iterations": config.iterations,
        "initial_count": initial_count,
        "aggregate_count": len(trajectories),
        "oracle_continuation_count": oracle_count,
        "aggregate_data_path": str(aggregate_path),
        "checkpoint_dir": str(checkpoint_dir),
        "train_result": train_result,
        "data_schema_version": base_samples.metadata["schema_version"],
    }


def _entropy_from_logits(logits: Any, mask: Any) -> Any:
    log_probs = jax.nn.log_softmax(logits)
    probs = jnp.exp(log_probs)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    return _masked_mean(entropy, mask)


def _rollout_rewards(
    *,
    data_path: str | Path,
    checkpoint_dir: str | Path,
    hidden_sizes: tuple[int, ...],
    max_steps: int | None,
    batch_size: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    from nf_agent.rollout import (
        HNFRolloutConfig,
        load_hnf_policy_runtime,
        rollout_hnf_policy_with_runtime,
    )

    rewards: list[float] = []
    rollout_config = HNFRolloutConfig(
        data_path=data_path,
        checkpoint_dir=checkpoint_dir,
        max_steps=max_steps,
        hidden_sizes=hidden_sizes,
    )
    runtime = load_hnf_policy_runtime(rollout_config)
    for matrix in _raw_matrices(data_path, batch_size):
        result = rollout_hnf_policy_with_runtime(runtime, rollout_config, matrix)
        reward = 1.0 if result.success else -1.0
        reward -= 0.01 * result.step_count
        reward -= 0.10 * result.invalid_action_count
        rewards.append(reward)
    return np.asarray(rewards, dtype=np.float32)


def train_hnf_actor_critic(config: HNFActorCriticConfig) -> dict[str, Any]:
    _validate_actor_critic_config(config)
    samples = HNFShardSamples(config.data_path)
    model = _model_for_samples(samples, config.hidden_sizes)
    restore_config = HNFTrainConfig(
        data_path=config.data_path,
        steps=1,
        batch_size=max(1, config.batch_size),
        learning_rate=config.learning_rate,
        seed=config.seed,
        out_dir=config.checkpoint_dir,
        hidden_sizes=config.hidden_sizes,
        max_to_keep=config.max_to_keep,
    )
    restored = restore_latest_hnf_policy_checkpoint(restore_config)
    state = _initial_state(config, samples, model).replace(params=restored.params)
    initial_params = state.params
    manager = _checkpoint_manager(config.out_dir, config.max_to_keep)
    batches = list(
        make_hnf_grain_dataset(
            config.data_path,
            batch_size=config.batch_size,
            seed=config.seed,
            drop_remainder=False,
        )
    )
    if not batches:
        raise ValueError("no training batches available")

    def train_step(
        train_state: TrainState,
        batch: ArrayBatch,
        rewards: Any,
    ) -> tuple[TrainState, dict[str, Any]]:
        def loss_fn(params: Any) -> tuple[Any, dict[str, Any]]:
            supervised, supervised_metrics = _loss_components(model, params, batch)
            outputs = cast(dict[str, Any], model.apply({"params": params}, batch["inputs"]))
            reward_targets = jnp.resize(
                jnp.asarray(rewards, dtype=jnp.float32),
                outputs["value"].shape,
            )
            advantages = reward_targets - jax.lax.stop_gradient(outputs["value"])
            policy_loss = supervised * jnp.maximum(0.1, 1.0 - jnp.mean(advantages))
            value_loss = jnp.mean(jnp.square(outputs["value"] - reward_targets))
            entropy = _entropy_from_logits(outputs["op_kind_logits"], batch["op_mask"])
            loss = (
                policy_loss
                + config.value_loss_weight * value_loss
                - config.entropy_weight * entropy
            )
            return loss, {
                **supervised_metrics,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "mean_reward": jnp.mean(reward_targets),
            }

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        next_state = cast(
            TrainState,
            train_state.apply_gradients(grads=grads),  # type: ignore[no-untyped-call]
        )
        return next_state, {"loss": loss, **metrics}

    jitted_train_step = jax.jit(train_step)
    final_metrics: MetricMap | None = None
    for step_offset in range(config.steps):
        reward_checkpoint = config.checkpoint_dir if step_offset == 0 else config.out_dir
        rewards = _rollout_rewards(
            data_path=config.data_path,
            checkpoint_dir=reward_checkpoint,
            hidden_sizes=config.hidden_sizes,
            max_steps=config.rollout_max_steps,
            batch_size=config.batch_size,
        )
        batch = _batch_to_jax(batches[step_offset % len(batches)])
        state, raw_metrics = jitted_train_step(state, batch, jnp.asarray(rewards))
        final_metrics = {
            key: float(np.asarray(jax.device_get(value))) for key, value in raw_metrics.items()
        }
        manager.save(int(state.step), args=ocp.args.StandardSave(state))
    manager.wait_until_finished()
    latest_step = manager.latest_step()
    if final_metrics is None:
        raise RuntimeError("actor-critic loop did not run")
    return {
        "status": "ok",
        "final_step": int(state.step),
        "latest_step": latest_step,
        "metrics": final_metrics,
        "checkpoint_dir": str(Path(config.out_dir)),
        "data_schema_version": samples.metadata["schema_version"],
        "scalar_vocab_size": samples.scalar_vocab_size,
        "parameters_changed": _params_changed(initial_params, state.params),
    }
