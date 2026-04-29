import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "pre_colab_preflight.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("pre_colab_preflight", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
pre_colab_preflight = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = pre_colab_preflight
SCRIPT_SPEC.loader.exec_module(pre_colab_preflight)

FORBIDDEN_COMPACT_KEYS = pre_colab_preflight.FORBIDDEN_COMPACT_KEYS
build_steps = pre_colab_preflight.build_steps


def _step_command_text(work_dir: Path) -> str:
    return "\n".join(" ".join(step.command) for step in build_steps(work_dir))


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


def test_dry_run_steps_cover_pre_colab_gates() -> None:
    work_dir = Path("/tmp/nf-pre-colab")
    steps = build_steps(work_dir)
    names = [step.name for step in steps]

    assert names == [
        "rref_shard",
        "rref_train",
        "rref_rollout",
        "rref_shard_benchmark",
        "hnf_v08_experiment",
        "snf_benchmark",
        "report_smoke",
        "lean_build",
    ]

    commands = _step_command_text(work_dir)
    assert "nf-agent data make-rref-shard" in commands
    assert "nf-agent train rref-pivot" in commands
    assert "nf-agent rollout rref-neural" in commands
    assert "nf-agent benchmark rref --source shard" in commands
    assert "nf-agent experiment hnf-v08" in commands
    assert "--allow-threshold-failure" in commands
    assert "nf-agent benchmark snf" in commands
    assert "nf-agent report benchmark" in commands
    assert "lake build" in commands


def test_rref_smoke_uses_colab_hidden_size_parity() -> None:
    commands = _step_command_text(Path("/tmp/nf-pre-colab"))
    assert commands.count("--hidden-size 32") == 3

    notebook = json.loads(Path("notebooks/rref_v6e_smoke_training.ipynb").read_text())
    source = "".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else ""
        for cell in notebook["cells"]
    )
    assert 'HIDDEN_SIZE = "32"' in source
    for snippet in (
        "nf-agent",
        "data",
        "make-rref-shard",
        "train",
        "rref-pivot",
        "rollout",
        "rref-neural",
        "benchmark",
        "rref",
    ):
        assert snippet in source


def test_tracked_pre_colab_fixtures_are_compact_json() -> None:
    fixture_dir = Path("tests/fixtures/pre_colab")
    expected = {
        "manifest.json",
        "rref_shard_benchmark_smoke.json",
        "hnf_v08_metrics_smoke.json",
        "snf_benchmark_smoke.json",
        "report_metrics_smoke.json",
    }
    assert {path.name for path in fixture_dir.glob("*.json")} == expected

    for path in sorted(fixture_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        assert isinstance(payload, dict)
        assert path.stat().st_size < 200_000
        for record in _walk_json(payload):
            leaked = FORBIDDEN_COMPACT_KEYS.intersection(record)
            assert not leaked, f"{path} leaked compact-forbidden keys: {sorted(leaked)}"

    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == "pre-colab-preflight-manifest-v1"
    assert manifest["overall_status"] == "ok"
    assert manifest["artifacts"]["rref_shard_benchmark"].endswith(
        "rref_shard_benchmark_smoke.json"
    )
    assert manifest["verdicts"]["hnf_v08"]["status"] in {"ok", "failed_threshold"}
    assert "certificate_replay" in manifest["verdicts"]["snf_benchmark"]["policies"]
