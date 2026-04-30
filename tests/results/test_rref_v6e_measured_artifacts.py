import json
from pathlib import Path

ARTIFACT_JSON = Path("results/measured/rref_32x32_mod1009_colab_v6e1_reduced_500step.json")
ARTIFACT_MD = Path("results/measured/rref_32x32_mod1009_colab_v6e1_reduced_500step.md")


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


def test_reduced_500step_colab_artifact_is_compact_tpu_smoke_evidence() -> None:
    payload = json.loads(ARTIFACT_JSON.read_text(encoding="utf-8"))
    report = ARTIFACT_MD.read_text(encoding="utf-8")

    assert payload["schema_version"] == "rref-v6e-profile-v1"
    assert payload["profile"] == "colab-v6e1-rref-reduced-32x32-mod1009"
    assert payload["status"] == "ok"
    assert payload["status_probe"]["jax"]["backend"] == "tpu"
    assert payload["task"] == {"rows": 32, "cols": 32, "modulus": 1009}
    assert payload["data"]["format"] == "zarr"
    assert payload["data"]["trace_count"] == 2048
    assert payload["train"]["final_step"] == 500
    assert payload["train"]["latest_step"] == 500
    assert payload["train"]["checkpoint_every"] == 100
    assert payload["beam"]["status"] == "max_steps_exceeded"
    assert payload["beam"]["success"] is False
    assert payload["beam"]["replay_ok"] is True
    assert payload["exact_cpu_verifier"]["replay_ok"] is True
    assert "Beam status: `max_steps_exceeded`" in report
    assert "Exact replay ok: `True`" in report
    assert "No hidden teacher fallback" in report

    forbidden_keys = {
        "checkpoint_dir",
        "checkpoint_path",
        "checkpoints",
        "final_matrix",
        "initial_matrix",
        "matrices",
        "matrix",
        "ops",
        "raw_logs",
        "stderr",
        "stdout",
    }
    for record in _walk_json(payload):
        leaked = forbidden_keys.intersection(record)
        assert not leaked, f"reduced Colab artifact leaked forbidden keys: {sorted(leaked)}"
