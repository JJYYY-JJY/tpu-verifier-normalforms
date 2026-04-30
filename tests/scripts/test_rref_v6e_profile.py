import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import yaml

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "rref_v6e_profile.py"
REDUCED_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "v6e1" / "rref_colab_reduced_profile.yaml"
)
LONG_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "v6e1"
    / "rref_colab_reduced_long_profile.yaml"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rref_v6e_profile", SCRIPT_PATH)
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


def _write_smoke_config(tmp_path: Path, *, required_backend: str | None = None) -> Path:
    config_path = tmp_path / "smoke.yaml"
    required = f"  required_backend: {required_backend}\n" if required_backend else ""
    config_path.write_text(
        "task: rref_matrixformer_smoke\n"
        "profile:\n"
        "  name: local-rref-smoke\n"
        f"{required}"
        "field:\n"
        "  modulus: 101\n"
        "matrix:\n"
        "  family: dense\n"
        "  rows: 4\n"
        "  cols: 4\n"
        "data:\n"
        "  format: zarr\n"
        "  count: 4\n"
        "  seed_start: 0\n"
        "  max_backward_ops: 4\n"
        "model:\n"
        "  row_embedding_dim: 8\n"
        "  col_embedding_dim: 8\n"
        "  hidden_dim: 32\n"
        "  layers: 1\n"
        "  num_heads: 1\n"
        "train:\n"
        "  steps: 2\n"
        "  batch_size: auto\n"
        "  learning_rate: 0.002\n"
        "  seed: 0\n"
        "  checkpoint_every: 2\n"
        "rollout:\n"
        "  max_steps: 4\n"
        "  beam_width: 4\n"
        "  batch_size: auto\n",
        encoding="utf-8",
    )
    return config_path


def test_assert_backend_fails_before_training() -> None:
    runner = _load_script()
    try:
        runner.assert_backend("cpu", "tpu")
    except RuntimeError as exc:
        assert "required JAX backend 'tpu', got 'cpu'" in str(exc)
        assert "before training" in str(exc)
    else:
        raise AssertionError("expected backend mismatch")


def test_rref_colab_reduced_profile_has_bounded_colab_defaults() -> None:
    config = yaml.safe_load(REDUCED_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["profile"] == {
        "name": "colab-v6e1-rref-reduced-32x32-mod1009",
        "required_backend": "tpu",
    }
    assert config["field"]["modulus"] == 1009
    assert config["matrix"] == {"family": "dense", "rows": 32, "cols": 32}
    assert config["data"]["format"] == "zarr"
    assert config["data"]["count"] == 2048
    assert config["data"]["max_backward_ops"] == 64
    assert config["model"] == {
        "name": "rref-matrixformer",
        "row_embedding_dim": 64,
        "col_embedding_dim": 64,
        "hidden_dim": 256,
        "layers": 4,
        "num_heads": 4,
    }
    assert config["train"]["steps"] == 500
    assert config["train"]["batch_size"] == "auto"
    assert config["train"]["checkpoint_every"] == 100
    assert config["rollout"]["beam_width"] == 8
    assert config["rollout"]["max_steps"] == 64
    assert config["artifacts"]["report_dir"] == "/tmp/nf-v6e1/rref_reduced/report"


def test_rref_colab_reduced_long_profile_is_next_colab_default() -> None:
    config = yaml.safe_load(LONG_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["profile"] == {
        "name": "colab-v6e1-rref-reduced-long-32x32-mod1009",
        "required_backend": "tpu",
    }
    assert config["field"]["modulus"] == 1009
    assert config["matrix"] == {"family": "dense", "rows": 32, "cols": 32}
    assert config["data"]["format"] == "zarr"
    assert config["data"]["count"] == 8192
    assert config["data"]["max_backward_ops"] == 96
    assert config["model"] == {
        "name": "rref-matrixformer",
        "row_embedding_dim": 64,
        "col_embedding_dim": 64,
        "hidden_dim": 256,
        "layers": 4,
        "num_heads": 4,
    }
    assert config["train"]["steps"] == 2000
    assert config["train"]["batch_size"] == "auto"
    assert config["train"]["learning_rate"] == 0.001
    assert config["train"]["checkpoint_every"] == 250
    assert config["rollout"]["beam_width"] == 8
    assert config["rollout"]["max_steps"] == 96
    assert config["artifacts"]["work_dir"] == "/tmp/nf-v6e1/rref_reduced_long/work"
    assert config["artifacts"]["report_dir"] == "/tmp/nf-v6e1/rref_reduced_long/report"


def test_rref_v6e_profile_required_tpu_fails_before_training(tmp_path: Path) -> None:
    config = _write_smoke_config(tmp_path, required_backend="tpu")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path.cwd() / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(config),
            "--work-dir",
            str(tmp_path / "work"),
            "--out-dir",
            str(tmp_path / "out"),
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


def test_rref_v6e_profile_local_smoke_writes_compact_report(tmp_path: Path) -> None:
    config = _write_smoke_config(tmp_path)
    out_dir = tmp_path / "out"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path.cwd() / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(config),
            "--work-dir",
            str(tmp_path / "work"),
            "--out-dir",
            str(out_dir),
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((out_dir / "summary.json").read_text())
    report = (out_dir / "report.md").read_text()
    stdout_payload = json.loads(completed.stdout)

    assert stdout_payload["schema_version"] == "rref-v6e-profile-v1"
    assert payload["schema_version"] == "rref-v6e-profile-v1"
    assert payload["status"] == "ok"
    assert payload["task"] == {"rows": 4, "cols": 4, "modulus": 101}
    assert payload["data"]["format"] == "zarr"
    assert payload["train"]["final_loss"] >= 0.0
    assert payload["train"]["learning_rate"] == 0.002
    assert payload["train"]["checkpoint_every"] == 2
    assert payload["beam"]["status"] in {"success", "max_steps_exceeded"}
    assert payload["beam"]["replay_ok"] is True
    assert payload["exact_cpu_verifier"]["replay_ok"] is True
    assert "No hidden teacher fallback" in payload["no_fallback_statement"]
    assert "No hidden teacher fallback" in report

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
        assert not leaked, f"v6e profile leaked forbidden keys: {sorted(leaked)}"
