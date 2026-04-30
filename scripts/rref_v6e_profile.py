from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]

JsonDict = dict[str, Any]
ProgressCallback = Callable[[str, JsonDict], None]
SCHEMA_VERSION = "rref-v6e-profile-v1"


def assert_backend(actual_backend: str, required_backend: str | None) -> None:
    if required_backend is None:
        return
    if actual_backend != required_backend:
        raise RuntimeError(
            f"required JAX backend {required_backend!r}, got {actual_backend!r} before training"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an RREF MatrixFormer v6e profile.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = run_profile(args.config, args.work_dir, args.out_dir)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_stdout_summary(payload), sort_keys=True))
    return 0 if payload["status"] == "ok" else 1


def run_profile(
    config_path: Path,
    work_dir: Path,
    out_dir: Path,
    *,
    progress: ProgressCallback | None = None,
) -> JsonDict:
    from nf_agent.data.rref_backward_shards import write_rref_backward_shard
    from nf_agent.data.rref_state_shards import (
        RREFStateActionSamples,
        load_rref_state_shard,
        write_rref_state_shard,
    )
    from nf_agent.env.rref_modp import is_rref_modp, replay_row_ops
    from nf_agent.profiles import NO_FALLBACK_STATEMENT, collect_v6e_status
    from nf_agent.rollout import RREFVerifierBeamConfig, rollout_rref_verifier_beam_sample
    from nf_agent.train import RREFMatrixFormerTrainConfig, train_rref_matrixformer

    started = time.perf_counter()
    config = _load_config(config_path)
    required_backend = _optional_str(
        _mapping(config.get("profile"), "profile").get("required_backend")
    )

    status_probe = collect_v6e_status(required_backend=required_backend)
    assert_backend(str(status_probe["jax"]["backend"]), required_backend)
    _emit_progress(
        progress,
        "backend",
        {
            "backend": status_probe["jax"]["backend"],
            "local_device_count": status_probe["jax"].get("local_device_count"),
            "required_backend": required_backend,
        },
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    backward_config = _write_backward_config(config, work_dir / "rref_backward_config.yaml")
    data_format = _optional_str(_mapping(config.get("data"), "data").get("format")) or "zarr"
    if data_format not in {"npz", "zarr"}:
        raise RuntimeError(f"unsupported data.format: {data_format!r}")
    suffix = f".{data_format}"
    trace_path = work_dir / f"rref_backward{suffix}"
    state_path = work_dir / f"rref_state{suffix}"
    ckpt_dir = work_dir / "rref_matrixformer_ckpt"

    data_config = _mapping(config.get("data"), "data")
    count = _positive_int(data_config.get("count"), "data.count")
    seed_start = _int(data_config.get("seed_start", 0), "data.seed_start")
    max_backward_ops = _positive_int(
        data_config.get("max_backward_ops", 4),
        "data.max_backward_ops",
    )
    write_rref_backward_shard(
        backward_config,
        count=count,
        seed_start=seed_start,
        out_path=trace_path,
    )
    _emit_progress(
        progress,
        "backward_shard",
        {
            "path": str(trace_path),
            "format": data_format,
            "count": count,
            "max_backward_ops": max_backward_ops,
        },
    )
    write_rref_state_shard(trace_path, state_path)
    _arrays, state_metadata = load_rref_state_shard(state_path)
    _emit_progress(
        progress,
        "state_shard",
        {
            "path": str(state_path),
            "trace_count": state_metadata["trace_count"],
            "flat_count": state_metadata["flat_count"],
            "max_steps": state_metadata["max_steps"],
        },
    )
    samples = RREFStateActionSamples(state_path)

    model_config = _mapping(config.get("model"), "model")
    train_config = _mapping(config.get("train"), "train")
    batch_size = _resolve_batch_size(train_config.get("batch_size"), len(samples))
    learning_rate = _positive_float(
        train_config.get("learning_rate", 0.001),
        "train.learning_rate",
    )
    checkpoint_every = _positive_int(
        train_config.get("checkpoint_every", 1),
        "train.checkpoint_every",
    )
    train_result = train_rref_matrixformer(
        RREFMatrixFormerTrainConfig(
            data_path=state_path,
            steps=_positive_int(train_config.get("steps"), "train.steps"),
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=_int(train_config.get("seed", 0), "train.seed"),
            out_dir=ckpt_dir,
            row_embedding_dim=_positive_int(
                model_config.get("row_embedding_dim", 32),
                "model.row_embedding_dim",
            ),
            col_embedding_dim=_positive_int(
                model_config.get("col_embedding_dim", 32),
                "model.col_embedding_dim",
            ),
            hidden_dim=_positive_int(model_config.get("hidden_dim", 256), "model.hidden_dim"),
            layers=_positive_int(model_config.get("layers", 2), "model.layers"),
            num_heads=_positive_int(model_config.get("num_heads", 4), "model.num_heads"),
            checkpoint_every=checkpoint_every,
        )
    )
    _emit_progress(
        progress,
        "training",
        {
            "status": train_result["status"],
            "final_step": train_result["final_step"],
            "latest_step": train_result["latest_step"],
            "final_loss": train_result["final_loss"],
            "batch_size": batch_size,
            "checkpoint_every": checkpoint_every,
        },
    )

    rollout_config = _mapping(config.get("rollout"), "rollout")
    beam_result = rollout_rref_verifier_beam_sample(
        RREFVerifierBeamConfig(
            data_path=state_path,
            checkpoint_dir=ckpt_dir,
            sample_index=0,
            max_steps=_positive_int(rollout_config.get("max_steps"), "rollout.max_steps"),
            beam_width=_positive_int(rollout_config.get("beam_width"), "rollout.beam_width"),
            batch_size=_batch_size_value(rollout_config.get("batch_size", "auto")),
            row_embedding_dim=_positive_int(
                model_config.get("row_embedding_dim", 32),
                "model.row_embedding_dim",
            ),
            col_embedding_dim=_positive_int(
                model_config.get("col_embedding_dim", 32),
                "model.col_embedding_dim",
            ),
            hidden_dim=_positive_int(model_config.get("hidden_dim", 256), "model.hidden_dim"),
            layers=_positive_int(model_config.get("layers", 2), "model.layers"),
            num_heads=_positive_int(model_config.get("num_heads", 4), "model.num_heads"),
        )
    )
    replayed = replay_row_ops(beam_result.initial_matrix, beam_result.ops, beam_result.modulus)
    exact_replay_ok = replayed == beam_result.final_matrix
    exact_final_is_rref = is_rref_modp(beam_result.final_matrix, beam_result.modulus)
    _emit_progress(
        progress,
        "beam",
        {
            "status": beam_result.status,
            "success": beam_result.success,
            "step_count": beam_result.step_count,
            "replay_ok": beam_result.replay_ok,
            "final_is_rref": beam_result.final_is_rref,
        },
    )

    matrix = _mapping(config.get("matrix"), "matrix")
    field = _mapping(config.get("field"), "field")
    payload: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if exact_replay_ok else "failed",
        "profile": _mapping(config.get("profile"), "profile").get("name", "local-rref"),
        "task": {
            "rows": _positive_int(matrix.get("rows"), "matrix.rows"),
            "cols": _positive_int(matrix.get("cols"), "matrix.cols"),
            "modulus": _positive_int(field.get("modulus"), "field.modulus"),
        },
        "status_probe": {
            "schema_version": status_probe["schema_version"],
            "jax": status_probe["jax"],
            "host": status_probe["host"],
            "hbm": status_probe["hbm"],
            "no_fallback_statement": status_probe["no_fallback_statement"],
        },
        "data": {
            "format": data_format,
            "trace_schema_version": state_metadata["source_schema_version"],
            "state_schema_version": state_metadata["schema_version"],
            "trace_count": state_metadata["trace_count"],
            "flat_count": state_metadata["flat_count"],
            "max_steps": state_metadata["max_steps"],
        },
        "train": {
            "status": train_result["status"],
            "final_step": train_result["final_step"],
            "latest_step": train_result["latest_step"],
            "final_loss": train_result["final_loss"],
            "parameters_changed": train_result["parameters_changed"],
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "checkpoint_every": train_result["checkpoint_every"],
            "per_head_metrics": train_result["per_head_metrics"],
        },
        "beam": {
            "status": beam_result.status,
            "success": beam_result.success,
            "step_count": beam_result.step_count,
            "final_is_rref": beam_result.final_is_rref,
            "beam_width": beam_result.beam_width,
            "score": beam_result.score,
            "replay_ok": beam_result.replay_ok,
            "expanded_count": beam_result.expanded_count,
            "pruned_count": beam_result.pruned_count,
            "device_batch_size": beam_result.device_batch_size,
        },
        "exact_cpu_verifier": {
            "replay_ok": exact_replay_ok,
            "final_is_rref": exact_final_is_rref,
        },
        "wall_time_seconds": time.perf_counter() - started,
        "no_fallback_statement": NO_FALLBACK_STATEMENT,
    }
    _write_outputs(payload, out_dir)
    _emit_progress(
        progress,
        "output",
        {
            "status": payload["status"],
            "summary_json": str(out_dir / "summary.json"),
            "report_md": str(out_dir / "report.md"),
        },
    )
    return payload


def _load_config(path: Path) -> JsonDict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError("config must be a YAML mapping")
    return loaded


def _mapping(value: object, name: str) -> JsonDict:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError("expected string value")
    return value


def _int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"{name} must be an integer")
    return value


def _positive_int(value: object, name: str) -> int:
    integer = _int(value, name)
    if integer <= 0:
        raise RuntimeError(f"{name} must be positive")
    return integer


def _positive_float(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise RuntimeError(f"{name} must be a number")
    number = float(value)
    if number <= 0.0:
        raise RuntimeError(f"{name} must be positive")
    return number


def _emit_progress(
    progress: ProgressCallback | None,
    stage: str,
    payload: JsonDict,
) -> None:
    if progress is not None:
        progress(stage, payload)


def _resolve_batch_size(value: object, sample_count: int) -> int:
    if value == "auto":
        return max(1, min(64, sample_count))
    return _positive_int(value, "train.batch_size")


def _batch_size_value(value: object) -> int | Literal["auto"]:
    if value == "auto":
        return "auto"
    return _positive_int(value, "rollout.batch_size")


def _write_backward_config(config: JsonDict, path: Path) -> Path:
    data_config = _mapping(config.get("data"), "data")
    field = _mapping(config.get("field"), "field")
    matrix = _mapping(config.get("matrix"), "matrix")
    payload = {
        "task": "rref_backward_state_shards",
        "field": {"modulus": field["modulus"]},
        "matrix": {
            "family": matrix.get("family", "dense"),
            "rows": matrix["rows"],
            "cols": matrix["cols"],
        },
        "backward_trace": {
            "schema": "rref-backward-trace-npz-v1",
            "format": data_config.get("format", "zarr"),
            "max_backward_ops": data_config.get("max_backward_ops", 4),
            "require_exact_replay": True,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_outputs(payload: JsonDict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(_render_report(payload), encoding="utf-8")


def _render_report(payload: JsonDict) -> str:
    return "\n".join(
        [
            f"# RREF v6e Profile: {payload['profile']}",
            "",
            f"- Status: `{payload['status']}`",
            f"- Backend: `{payload['status_probe']['jax']['backend']}`",
            f"- Task: `{json.dumps(payload['task'], sort_keys=True)}`",
            f"- Data format: `{payload['data']['format']}`",
            f"- Train checkpoint every: `{payload['train']['checkpoint_every']}`",
            f"- Train learning rate: `{payload['train']['learning_rate']}`",
            f"- Train final loss: `{payload['train']['final_loss']:.6g}`",
            f"- Beam status: `{payload['beam']['status']}`",
            f"- Exact replay ok: `{payload['exact_cpu_verifier']['replay_ok']}`",
            "",
            str(payload["no_fallback_statement"]),
            "",
        ]
    )


def _stdout_summary(payload: JsonDict) -> JsonDict:
    return {
        "schema_version": cast(str, payload["schema_version"]),
        "status": cast(str, payload["status"]),
        "backend": cast(JsonDict, cast(JsonDict, payload["status_probe"])["jax"])["backend"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
