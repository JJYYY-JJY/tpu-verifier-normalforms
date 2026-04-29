from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

FORBIDDEN_COMPACT_KEYS = {
    "checkpoint_dir",
    "checkpoint_path",
    "final_matrix",
    "initial_matrix",
    "matrix",
    "ops",
    "raw_logs",
    "stderr",
    "stdout",
}


@dataclass(frozen=True)
class V6EProfileReportConfig:
    input_path: str | Path
    out_dir: str | Path


def _load_json(path: str | Path) -> JsonDict:
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("v6e profile input must be a JSON object")
    return loaded


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_COMPACT_KEYS:
                raise ValueError(f"forbidden compact report key: {key}")
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child)


def _summary_from_profile(profile: JsonDict) -> JsonDict:
    _reject_forbidden_keys(profile)
    jax_info = profile.get("jax", {})
    if not isinstance(jax_info, dict):
        raise ValueError("v6e profile input must include jax object")
    host = profile.get("host", {})
    if not isinstance(host, dict):
        raise ValueError("v6e profile input must include host object")
    hbm = profile.get("hbm", {})
    if not isinstance(hbm, dict):
        raise ValueError("v6e profile input must include hbm object")
    return {
        "schema_version": "v6e-profile-report-v1",
        "source_schema_version": profile.get("schema_version"),
        "status": profile.get("status"),
        "backend": jax_info.get("backend"),
        "required_backend": jax_info.get("required_backend"),
        "local_device_count": jax_info.get("local_device_count"),
        "device_kinds": [
            str(device.get("device_kind"))
            for device in jax_info.get("devices", [])
            if isinstance(device, dict)
        ],
        "versions": profile.get("versions", {}),
        "host_ram": host.get("ram", {}),
        "hbm": hbm,
        "no_fallback_statement": profile.get("no_fallback_statement"),
    }


def render_v6e_profile_report(summary: JsonDict) -> str:
    return "\n".join(
        [
            "# v6e Profile Report",
            "",
            f"- Status: `{summary['status']}`",
            f"- Backend: `{summary['backend']}`",
            f"- Required backend: `{summary['required_backend']}`",
            f"- Local devices: `{summary['local_device_count']}`",
            f"- Device kinds: `{json.dumps(summary['device_kinds'])}`",
            f"- HBM status: `{summary['hbm'].get('status')}`",
            "",
            str(summary.get("no_fallback_statement", "")),
            "",
        ]
    )


def build_v6e_profile_report(config: V6EProfileReportConfig) -> JsonDict:
    profile = _load_json(config.input_path)
    summary = _summary_from_profile(profile)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_v6e_profile_report(summary), encoding="utf-8")
    return {
        "status": "ok",
        "summary_json": str(summary_path),
        "report_md": str(report_path),
        "schema_version": summary["schema_version"],
    }

