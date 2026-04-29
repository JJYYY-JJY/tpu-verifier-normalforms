from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import psutil  # type: ignore[import-untyped]

JsonDict = dict[str, Any]

NO_FALLBACK_STATEMENT = (
    "No hidden teacher fallback: exact replay/verifier stages are authoritative; "
    "neural failures remain explicit statuses."
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
class V6EStatusConfig:
    memory_profile: str | Path
    required_backend: Literal["cpu", "gpu", "tpu"] | None = None


def assert_backend(actual_backend: str, required_backend: str | None) -> None:
    if required_backend is None:
        return
    if actual_backend != required_backend:
        raise RuntimeError(
            f"required JAX backend {required_backend!r}, got {actual_backend!r} before training"
        )


def _jax_info(required_backend: str | None) -> JsonDict:
    import jax
    import jaxlib

    try:
        backend = str(jax.default_backend())
    except RuntimeError as exc:
        if required_backend is not None:
            raise RuntimeError(
                f"required JAX backend {required_backend!r} is unavailable before training: {exc}"
            ) from exc
        raise
    assert_backend(backend, required_backend)
    devices = []
    hbm_total = 0
    hbm_used = 0
    hbm_seen = False
    for device in jax.devices():
        item: JsonDict = {
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
        }
        try:
            stats = device.memory_stats()
        except (AttributeError, RuntimeError, TypeError):
            stats = None
        if isinstance(stats, dict):
            bytes_limit = stats.get("bytes_limit")
            bytes_in_use = stats.get("bytes_in_use")
            if isinstance(bytes_limit, int):
                item["memory_bytes_limit"] = bytes_limit
                hbm_total += bytes_limit
                hbm_seen = True
            if isinstance(bytes_in_use, int):
                item["memory_bytes_in_use"] = bytes_in_use
                hbm_used += bytes_in_use
                hbm_seen = True
        devices.append(item)
    return {
        "backend": backend,
        "required_backend": required_backend,
        "local_device_count": int(jax.local_device_count()),
        "devices": devices,
        "versions": {
            "jax": str(jax.__version__),
            "jaxlib": str(getattr(jaxlib, "__version__", "unknown")),
        },
        "hbm": {
            "status": "available" if hbm_seen else "unavailable",
            "total_bytes": hbm_total if hbm_seen else None,
            "used_bytes": hbm_used if hbm_seen else None,
        },
    }


def collect_v6e_status(required_backend: str | None = None) -> JsonDict:
    jax_info = _jax_info(required_backend)
    ram = psutil.virtual_memory()
    return {
        "schema_version": "v6e-status-v1",
        "status": "ok",
        "versions": {
            "python": platform.python_version(),
            **jax_info["versions"],
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_executable": Path(sys.executable).name,
        },
        "jax": {
            "backend": jax_info["backend"],
            "required_backend": jax_info["required_backend"],
            "local_device_count": jax_info["local_device_count"],
            "devices": jax_info["devices"],
        },
        "host": {
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram": {
                "total_bytes": int(ram.total),
                "available_bytes": int(ram.available),
            },
        },
        "hbm": jax_info["hbm"],
        "env": {key: os.environ.get(key) for key in ENV_KEYS if os.environ.get(key)},
        "no_fallback_statement": NO_FALLBACK_STATEMENT,
    }


def write_v6e_status(config: V6EStatusConfig) -> JsonDict:
    payload = collect_v6e_status(required_backend=config.required_backend)
    path = Path(config.memory_profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload

