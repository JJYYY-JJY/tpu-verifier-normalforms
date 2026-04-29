import json
from pathlib import Path

from click.testing import CliRunner

from nf_agent.cli import main


def test_profile_v6e_status_writes_local_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"

    result = CliRunner().invoke(
        main,
        ["profile", "v6e-status", "--memory-profile", str(profile_path)],
    )

    assert result.exit_code == 0, result.output
    stdout_payload = json.loads(result.output)
    file_payload = json.loads(profile_path.read_text())
    assert stdout_payload == file_payload
    assert file_payload["schema_version"] == "v6e-status-v1"
    assert file_payload["jax"]["backend"] in {"cpu", "gpu", "tpu"}
    assert "python" in file_payload["versions"]
    assert "ram" in file_payload["host"]
    assert "hbm" in file_payload
    assert "No hidden teacher fallback" in file_payload["no_fallback_statement"]


def test_profile_v6e_status_required_tpu_fails_before_training(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"

    result = CliRunner().invoke(
        main,
        [
            "profile",
            "v6e-status",
            "--memory-profile",
            str(profile_path),
            "--required-backend",
            "tpu",
        ],
    )

    if result.exit_code == 0:
        payload = json.loads(result.output)
        assert payload["jax"]["backend"] == "tpu"
    else:
        assert "required JAX backend 'tpu'" in result.output
        assert "before training" in result.output
        assert not profile_path.exists()

