from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

SCHEMA_VERSION = "rref-measured-run-v1"
NO_FALLBACK_STATEMENT = (
    "No hidden teacher fallback: leftmost is reported only as an explicit baseline; "
    "neural failures remain neural status counts."
)
ENV_KEYS = (
    "JAX_PLATFORMS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "XLA_FLAGS",
    "TPU_NAME",
    "COLAB_TPU_ADDR",
)


@dataclass(frozen=True)
class Profile:
    name: str
    shard_count: int
    train_steps: int
    benchmark_count: int
    hidden_sizes: tuple[int, ...]
    batch_candidates: tuple[int, ...]
    max_steps: int
    calibration_steps: int
    checkpoint_every: int
    required_backend: str | None
    env: dict[str, str]


PROFILES: dict[str, Profile] = {
    "apple-m4-large": Profile(
        name="apple-m4-large",
        shard_count=4096,
        train_steps=2000,
        benchmark_count=512,
        hidden_sizes=(512, 512),
        batch_candidates=(128, 256, 512),
        max_steps=72,
        calibration_steps=5,
        checkpoint_every=50,
        required_backend="cpu",
        env={
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "10",
            "VECLIB_MAXIMUM_THREADS": "10",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=10",
        },
    ),
    "colab-v6e1-large": Profile(
        name="colab-v6e1-large",
        shard_count=4096,
        train_steps=2000,
        benchmark_count=512,
        hidden_sizes=(512, 512),
        batch_candidates=(128, 256, 512),
        max_steps=72,
        calibration_steps=5,
        checkpoint_every=50,
        required_backend="tpu",
        env={"JAX_PLATFORMS": "tpu,cpu"},
    ),
    "local-smoke": Profile(
        name="local-smoke",
        shard_count=8,
        train_steps=2,
        benchmark_count=2,
        hidden_sizes=(32,),
        batch_candidates=(2, 4),
        max_steps=8,
        calibration_steps=1,
        checkpoint_every=1,
        required_backend=None,
        env={},
    ),
}


def assert_backend(actual_backend: str, required_backend: str | None) -> None:
    if required_backend is None:
        return
    if actual_backend != required_backend:
        raise RuntimeError(f"required JAX backend {required_backend!r}, got {actual_backend!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a measured RREF 8x8/F_101 profile.")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args(argv)

    profile_spec = PROFILES[args.profile]
    _apply_profile_env(profile_spec)
    try:
        payload = run_profile(profile_spec, args.work_dir)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _write_json(args.out, payload)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(render_summary(payload), encoding="utf-8")
    print(json.dumps(_stdout_summary(payload), sort_keys=True))
    return 0 if payload["status"] == "ok" else 1


def run_profile(profile_spec: Profile, work_dir: Path) -> JsonDict:
    from nf_agent.benchmarks.rref_benchmark import RREFBenchmarkConfig, run_rref_benchmark
    from nf_agent.data.rref_shards import write_rref_shard
    from nf_agent.train import TrainConfig, train_rref_pivot

    started = time.perf_counter()
    work_dir.mkdir(parents=True, exist_ok=True)
    _reset_run_dirs(work_dir)

    jax_info = _jax_info(profile_spec.required_backend)
    shard_path = work_dir / "rref_8x8_mod101.npz"
    train_checkpoint_dir = work_dir / "train_ckpt"

    stage_wall_times: dict[str, float] = {}
    shard_start = time.perf_counter()
    write_rref_shard(
        config_path="configs/rref_8x8_mod101.yaml",
        count=profile_spec.shard_count,
        seed_start=0,
        out_path=shard_path,
    )
    stage_wall_times["shard_seconds"] = time.perf_counter() - shard_start

    calibration = _calibrate_batches(
        profile_spec=profile_spec,
        shard_path=shard_path,
        work_dir=work_dir / "calibration",
    )
    stage_wall_times["calibration_seconds"] = calibration["wall_time_seconds"]
    selected_batch_size = int(calibration["selected_batch_size"])

    train_start = time.perf_counter()
    train_result = train_rref_pivot(
        TrainConfig(
            data_path=shard_path,
            steps=profile_spec.train_steps,
            batch_size=selected_batch_size,
            learning_rate=0.001,
            seed=0,
            out_dir=train_checkpoint_dir,
            hidden_sizes=profile_spec.hidden_sizes,
            checkpoint_every=profile_spec.checkpoint_every,
        )
    )
    stage_wall_times["train_seconds"] = time.perf_counter() - train_start

    benchmark_start = time.perf_counter()
    benchmark_result = run_rref_benchmark(
        RREFBenchmarkConfig(
            source="shard",
            data_path=shard_path,
            count=profile_spec.benchmark_count,
            checkpoint_dir=train_checkpoint_dir,
            max_steps=profile_spec.max_steps,
            hidden_sizes=profile_spec.hidden_sizes,
        )
    )
    stage_wall_times["benchmark_seconds"] = time.perf_counter() - benchmark_start
    stage_wall_times["total_seconds"] = time.perf_counter() - started

    status = _overall_status(benchmark_result)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "profile": profile_spec.name,
        "task": {"rows": 8, "cols": 8, "modulus": 101},
        "config": {
            "shard_count": profile_spec.shard_count,
            "train_steps": profile_spec.train_steps,
            "benchmark_count": profile_spec.benchmark_count,
            "hidden_sizes": list(profile_spec.hidden_sizes),
            "batch_candidates": list(profile_spec.batch_candidates),
            "max_steps": profile_spec.max_steps,
            "checkpoint_every": profile_spec.checkpoint_every,
        },
        "versions": _version_info(),
        "host": _host_info(),
        "jax": jax_info,
        "env": {key: os.environ.get(key) for key in ENV_KEYS if os.environ.get(key)},
        "selected_batch_size": selected_batch_size,
        "calibration": calibration,
        "train": _compact_train(
            train_result,
            stage_wall_times["train_seconds"],
            selected_batch_size,
        ),
        "benchmark": _compact_benchmark(
            benchmark_result,
            stage_wall_times["benchmark_seconds"],
        ),
        "stage_wall_times": stage_wall_times,
        "no_fallback_statement": NO_FALLBACK_STATEMENT,
    }


def _apply_profile_env(profile_spec: Profile) -> None:
    for key, value in profile_spec.env.items():
        os.environ[key] = value


def _jax_info(required_backend: str | None) -> JsonDict:
    import jax

    try:
        backend = str(jax.default_backend())
    except RuntimeError as exc:
        if required_backend is not None:
            raise RuntimeError(
                f"required JAX backend {required_backend!r} is unavailable before training: {exc}"
            ) from exc
        raise
    assert_backend(backend, required_backend)
    devices = [
        {
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
        }
        for device in jax.devices()
    ]
    return {
        "backend": backend,
        "required_backend": required_backend,
        "local_device_count": int(jax.local_device_count()),
        "devices": devices,
    }


def _version_info() -> JsonDict:
    import jax
    import jaxlib

    return {
        "python": platform.python_version(),
        "jax": str(jax.__version__),
        "jaxlib": str(getattr(jaxlib, "__version__", "unknown")),
    }


def _host_info() -> JsonDict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_executable": Path(sys.executable).name,
    }


def _calibrate_batches(
    *,
    profile_spec: Profile,
    shard_path: Path,
    work_dir: Path,
) -> JsonDict:
    from nf_agent.train import TrainConfig, train_rref_pivot

    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    candidates: list[JsonDict] = []
    for batch_size in profile_spec.batch_candidates:
        candidate_ckpt = work_dir / f"batch_{batch_size}"
        if candidate_ckpt.exists():
            shutil.rmtree(candidate_ckpt)
        candidate_start = time.perf_counter()
        try:
            result = train_rref_pivot(
                TrainConfig(
                    data_path=shard_path,
                    steps=profile_spec.calibration_steps,
                    batch_size=batch_size,
                    learning_rate=0.001,
                    seed=0,
                    out_dir=candidate_ckpt,
                    hidden_sizes=profile_spec.hidden_sizes,
                    checkpoint_every=max(1, profile_spec.calibration_steps),
                )
            )
        except (MemoryError, OSError, RuntimeError, TypeError, ValueError) as exc:
            wall_seconds = time.perf_counter() - candidate_start
            candidates.append(
                {
                    "batch_size": batch_size,
                    "status": "failed",
                    "wall_time_seconds": wall_seconds,
                    "error": str(exc),
                }
            )
            continue
        wall_seconds = time.perf_counter() - candidate_start
        samples_per_second = (
            batch_size * profile_spec.calibration_steps / max(wall_seconds, 1e-9)
        )
        candidates.append(
            {
                "batch_size": batch_size,
                "status": "ok",
                "wall_time_seconds": wall_seconds,
                "samples_per_second": samples_per_second,
                "final_loss": float(result["final_loss"]),
                "checkpoint_step": int(result["latest_step"]),
            }
        )

    successful = [candidate for candidate in candidates if candidate["status"] == "ok"]
    if not successful:
        errors = "; ".join(str(candidate.get("error", "")) for candidate in candidates)
        raise RuntimeError(f"batch calibration failed for all candidates: {errors}")
    selected = max(successful, key=lambda candidate: float(candidate["samples_per_second"]))
    return {
        "status": "ok",
        "wall_time_seconds": time.perf_counter() - started,
        "selected_batch_size": selected["batch_size"],
        "candidates": candidates,
    }


def _compact_train(
    train_result: JsonDict,
    wall_seconds: float,
    selected_batch_size: int,
) -> JsonDict:
    samples_seen = int(train_result["final_step"]) * selected_batch_size
    samples_per_second = samples_seen / max(wall_seconds, 1e-9)
    return {
        "status": train_result["status"],
        "final_step": train_result["final_step"],
        "checkpoint_step": train_result["latest_step"],
        "checkpoint_every": train_result["checkpoint_every"],
        "final_loss": train_result["final_loss"],
        "parameters_changed": train_result["parameters_changed"],
        "wall_time_seconds": wall_seconds,
        "steps_completed": train_result["final_step"],
        "samples_seen_proxy": samples_seen,
        "samples_per_second_proxy": samples_per_second,
        "per_head_metrics": train_result["per_head_metrics"],
    }


def _compact_benchmark(benchmark_result: JsonDict, wall_seconds: float) -> JsonDict:
    policies: JsonDict = {}
    raw_policies = benchmark_result.get("policies", {})
    if not isinstance(raw_policies, dict):
        raise RuntimeError("benchmark result missing policies")
    for name, payload in raw_policies.items():
        if not isinstance(payload, dict) or not isinstance(payload.get("aggregate"), dict):
            continue
        aggregate = dict(payload["aggregate"])
        mean_wall = float(aggregate.get("mean_wall_time_seconds", 0.0))
        policies[str(name)] = {
            **aggregate,
            "samples_per_second": 1.0 / max(mean_wall, 1e-9),
        }
    return {
        "status": benchmark_result["status"],
        "source": benchmark_result["source"],
        "count": benchmark_result["count"],
        "rows": benchmark_result["rows"],
        "cols": benchmark_result["cols"],
        "modulus": benchmark_result["modulus"],
        "wall_time_seconds": wall_seconds,
        "policies": policies,
    }


def _overall_status(benchmark_result: JsonDict) -> str:
    policies = benchmark_result.get("policies", {})
    if not isinstance(policies, dict):
        return "failed"
    if "leftmost" not in policies or "neural" not in policies:
        return "failed"
    leftmost = policies["leftmost"].get("aggregate", {})
    return "ok" if leftmost.get("success_rate") == 1.0 else "failed"


def render_summary(payload: JsonDict) -> str:
    lines = [
        f"# RREF 8x8/F_101 Measured Run: {payload['profile']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Backend: `{payload['jax']['backend']}`",
        f"- Required backend: `{payload['jax']['required_backend']}`",
        f"- Local devices: `{payload['jax']['local_device_count']}`",
        f"- Selected batch size: `{payload['selected_batch_size']}`",
        f"- Train final loss: `{payload['train']['final_loss']:.6g}`",
        f"- Checkpoint step: `{payload['train']['checkpoint_step']}`",
        f"- Total wall time seconds: `{payload['stage_wall_times']['total_seconds']:.3f}`",
        "",
        "## Calibration",
        "",
        "| batch | status | samples/sec | seconds |",
        "| ---: | :--- | ---: | ---: |",
    ]
    for candidate in payload["calibration"]["candidates"]:
        lines.append(
            "| {batch_size} | {status} | {samples_per_second} | {wall_time_seconds:.3f} |".format(
                batch_size=candidate["batch_size"],
                status=candidate["status"],
                samples_per_second=(
                    f"{candidate['samples_per_second']:.3f}"
                    if "samples_per_second" in candidate
                    else "n/a"
                ),
                wall_time_seconds=float(candidate["wall_time_seconds"]),
            )
        )
    lines.extend(
        ["", "## Benchmark", "", "| policy | success rate | status counts | samples/sec |"]
    )
    lines.append("| :--- | ---: | :--- | ---: |")
    for name, policy in payload["benchmark"]["policies"].items():
        lines.append(
            "| {name} | {success_rate:.3f} | `{status_counts}` | {samples_per_second:.3f} |".format(
                name=name,
                success_rate=float(policy["success_rate"]),
                status_counts=json.dumps(policy["status_counts"], sort_keys=True),
                samples_per_second=float(policy["samples_per_second"]),
            )
        )
    lines.extend(["", payload["no_fallback_statement"], ""])
    return "\n".join(lines)


def _reset_run_dirs(work_dir: Path) -> None:
    for child_name in ("calibration", "train_ckpt"):
        child = work_dir / child_name
        if child.exists():
            shutil.rmtree(child)
    shard = work_dir / "rref_8x8_mod101.npz"
    if shard.exists():
        shard.unlink()


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stdout_summary(payload: JsonDict) -> JsonDict:
    return {
        "status": payload["status"],
        "profile": payload["profile"],
        "backend": payload["jax"]["backend"],
        "selected_batch_size": payload["selected_batch_size"],
        "out_schema": payload["schema_version"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
