import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "rref_measured_run.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rref_measured_run", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _walk_json(value: object) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if isinstance(value, dict):
        records.append(value)
        for child in value.values():
            records.extend(_walk_json(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(_walk_json(child))
    return records


def test_backend_assertion_fails_explicitly() -> None:
    measured = _load_script()

    try:
        measured.assert_backend("cpu", "tpu")
    except RuntimeError as exc:
        assert "required JAX backend 'tpu', got 'cpu'" in str(exc)
    else:
        raise AssertionError("expected backend mismatch to fail")


def test_colab_profile_fails_before_training_when_tpu_unavailable(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "colab.json"
    summary_path = tmp_path / "colab.md"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path.cwd() / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--profile",
            "colab-v6e1-large",
            "--work-dir",
            str(tmp_path / "work"),
            "--out",
            str(out_path),
            "--summary-md",
            str(summary_path),
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "required JAX backend 'tpu'" in completed.stderr
    assert "before training" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not out_path.exists()
    assert not summary_path.exists()


def test_local_smoke_measured_run_writes_compact_json_and_summary(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "rref_smoke.json"
    summary_path = tmp_path / "rref_smoke.md"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path.cwd() / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--profile",
            "local-smoke",
            "--work-dir",
            str(tmp_path / "work"),
            "--out",
            str(out_path),
            "--summary-md",
            str(summary_path),
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out_path.read_text())
    summary = summary_path.read_text()

    assert payload["schema_version"] == "rref-measured-run-v1"
    assert payload["profile"] == "local-smoke"
    assert payload["status"] == "ok"
    assert payload["task"] == {"rows": 8, "cols": 8, "modulus": 101}
    assert payload["selected_batch_size"] in {2, 4}
    assert payload["benchmark"]["policies"]["leftmost"]["success_rate"] == 1.0
    assert "No hidden teacher fallback" in payload["no_fallback_statement"]
    assert "No hidden teacher fallback" in summary

    forbidden_keys = {
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
    for record in _walk_json(payload):
        leaked = forbidden_keys.intersection(record)
        assert not leaked, f"measured output leaked forbidden keys: {sorted(leaked)}"
