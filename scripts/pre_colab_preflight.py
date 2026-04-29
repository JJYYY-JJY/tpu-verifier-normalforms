from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

MANIFEST_SCHEMA_VERSION = "pre-colab-preflight-manifest-v1"
FORBIDDEN_COMPACT_KEYS = {
    "col_ops",
    "diagonal",
    "final",
    "final_matrix",
    "initial_matrix",
    "input",
    "left_transform",
    "matrix",
    "ops",
    "right_transform",
    "row_ops",
}


@dataclass(frozen=True)
class PreflightStep:
    name: str
    command: list[str]
    cwd: Path | None = None


@dataclass(frozen=True)
class StepResult:
    name: str
    command: list[str]
    cwd: Path
    exit_code: int
    stdout_path: Path
    stderr_path: Path


def build_steps(work_dir: Path) -> list[PreflightStep]:
    rref_shard = work_dir / "rref_8x8_train_smoke.npz"
    rref_checkpoint = work_dir / "rref_pivot_ckpt"
    hnf_out_dir = work_dir / "hnf_v08"
    report_out_dir = work_dir / "report_smoke"
    return [
        PreflightStep(
            name="rref_shard",
            command=[
                "nf-agent",
                "data",
                "make-rref-shard",
                "--config",
                "configs/rref_8x8_mod101.yaml",
                "--count",
                "8",
                "--seed-start",
                "0",
                "--out",
                str(rref_shard),
            ],
        ),
        PreflightStep(
            name="rref_train",
            command=[
                "nf-agent",
                "train",
                "rref-pivot",
                "--data",
                str(rref_shard),
                "--steps",
                "2",
                "--batch-size",
                "4",
                "--learning-rate",
                "0.001",
                "--seed",
                "0",
                "--hidden-size",
                "32",
                "--out",
                str(rref_checkpoint),
            ],
        ),
        PreflightStep(
            name="rref_rollout",
            command=[
                "nf-agent",
                "rollout",
                "rref-neural",
                "--data",
                str(rref_shard),
                "--checkpoint",
                str(rref_checkpoint),
                "--sample-index",
                "0",
                "--max-steps",
                "8",
                "--hidden-size",
                "32",
            ],
        ),
        PreflightStep(
            name="rref_shard_benchmark",
            command=[
                "nf-agent",
                "benchmark",
                "rref",
                "--source",
                "shard",
                "--data",
                str(rref_shard),
                "--count",
                "4",
                "--checkpoint",
                str(rref_checkpoint),
                "--max-steps",
                "8",
                "--hidden-size",
                "32",
            ],
        ),
        PreflightStep(
            name="hnf_v08_experiment",
            command=[
                "nf-agent",
                "experiment",
                "hnf-v08",
                "--out-dir",
                str(hnf_out_dir),
                "--samples-per-size",
                "2",
                "--run-seed-count",
                "1",
                "--sizes",
                "3",
                "--density",
                "0.5",
                "--entry-bound",
                "4",
                "--train-steps",
                "1",
                "--dagger-iterations",
                "1",
                "--actor-critic-steps",
                "1",
                "--batch-size",
                "2",
                "--hidden-size",
                "16",
                "--benchmark-max-steps",
                "2",
                "--allow-threshold-failure",
            ],
        ),
        PreflightStep(
            name="snf_benchmark",
            command=[
                "nf-agent",
                "benchmark",
                "snf",
                "--rows",
                "3",
                "--cols",
                "3",
                "--count",
                "2",
                "--diagonal-factor-bound",
                "5",
                "--row-op-count",
                "2",
                "--col-op-count",
                "1",
                "--op-scalar-bound",
                "3",
                "--seed-start",
                "7",
            ],
        ),
        PreflightStep(
            name="report_smoke",
            command=[
                "nf-agent",
                "report",
                "benchmark",
                "--out-dir",
                str(report_out_dir),
                "--sample-count",
                "2",
                "--rows",
                "3",
                "--cols",
                "3",
                "--p",
                "5",
                "--seed-start",
                "0",
                "--sparse-density",
                "0.3",
                "--low-rank",
                "2",
                "--hnf-entry-bound",
                "5",
            ],
        ),
        PreflightStep(name="lean_build", command=["lake", "build"], cwd=Path("lean")),
    ]


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the local pre-Colab verifier-first artifact freeze."
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/nf-pre-colab"),
        help="Directory for heavy intermediates and raw logs.",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=repo_root / "tests" / "fixtures" / "pre_colab",
        help="Directory for compact tracked JSON fixtures.",
    )
    parser.add_argument(
        "--write-fixtures",
        action="store_true",
        help="Write compact fixture JSON under --fixture-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands as JSON and exit without running them.",
    )
    args = parser.parse_args(argv)

    work_dir = args.work_dir
    steps = build_steps(work_dir)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "work_dir": str(work_dir),
                    "steps": [
                        {
                            "name": step.name,
                            "cwd": str(step.cwd) if step.cwd is not None else ".",
                            "command": step.command,
                        }
                        for step in steps
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    work_dir.mkdir(parents=True, exist_ok=True)
    _reset_known_outputs(work_dir)
    logs_dir = work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    results: list[StepResult] = []
    parsed_outputs: dict[str, JsonDict] = {}
    for step in steps:
        result = _run_step(step=step, repo_root=repo_root, logs_dir=logs_dir)
        results.append(result)
        if result.exit_code != 0:
            manifest = _build_manifest(
                work_dir=work_dir,
                repo_root=repo_root,
                results=results,
                parsed_outputs=parsed_outputs,
                overall_status="failed",
            )
            _write_json(work_dir / "manifest.json", manifest)
            if args.write_fixtures:
                _write_fixtures(args.fixture_dir, manifest, work_dir, parsed_outputs)
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return result.exit_code
        if step.name != "lean_build":
            parsed_outputs[step.name] = _read_stdout_json(result.stdout_path, step.name)
        _materialize_step_artifact(step.name, work_dir, parsed_outputs)

    manifest = _build_manifest(
        work_dir=work_dir,
        repo_root=repo_root,
        results=results,
        parsed_outputs=parsed_outputs,
        overall_status=_overall_status(parsed_outputs),
    )
    _write_json(work_dir / "manifest.json", manifest)
    if args.write_fixtures:
        _write_fixtures(args.fixture_dir, manifest, work_dir, parsed_outputs)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["overall_status"] == "ok" else 1


def _reset_known_outputs(work_dir: Path) -> None:
    for path in (
        work_dir / "rref_8x8_train_smoke.npz",
        work_dir / "rref_shard_benchmark_smoke.json",
        work_dir / "snf_benchmark_smoke.json",
        work_dir / "manifest.json",
    ):
        if path.is_file():
            path.unlink()
    for path in (
        work_dir / "rref_pivot_ckpt",
        work_dir / "hnf_v08",
        work_dir / "report_smoke",
        work_dir / "logs",
    ):
        if path.is_dir():
            for child in sorted(path.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            path.rmdir()


def _run_step(*, step: PreflightStep, repo_root: Path, logs_dir: Path) -> StepResult:
    cwd = repo_root / step.cwd if step.cwd is not None else repo_root
    completed = subprocess.run(
        step.command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path = logs_dir / f"{step.name}.stdout"
    stderr_path = logs_dir / f"{step.name}.stderr"
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    return StepResult(
        name=step.name,
        command=step.command,
        cwd=cwd,
        exit_code=completed.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def _read_stdout_json(path: Path, step_name: str) -> JsonDict:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{step_name} did not emit JSON on stdout: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{step_name} stdout JSON must be an object")
    return payload


def _materialize_step_artifact(
    step_name: str,
    work_dir: Path,
    parsed_outputs: dict[str, JsonDict],
) -> None:
    if step_name == "rref_shard_benchmark":
        _write_json(work_dir / "rref_shard_benchmark_smoke.json", parsed_outputs[step_name])
    elif step_name == "snf_benchmark":
        _write_json(work_dir / "snf_benchmark_smoke.json", parsed_outputs[step_name])


def _build_manifest(
    *,
    work_dir: Path,
    repo_root: Path,
    results: list[StepResult],
    parsed_outputs: dict[str, JsonDict],
    overall_status: str,
) -> JsonDict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "overall_status": overall_status,
        "git_commit": _command_stdout(["git", "rev-parse", "HEAD"], repo_root),
        "versions": {
            "python": sys.version.split()[0],
            "lean": _command_stdout(["lean", "--version"], repo_root / "lean"),
            "lake": _command_stdout(["lake", "--version"], repo_root / "lean"),
        },
        "work_dir": str(work_dir),
        "commands": [
            {
                "name": result.name,
                "cwd": _relative_cwd(result.cwd, repo_root),
                "command": result.command,
                "exit_code": result.exit_code,
                "stdout_path": str(result.stdout_path),
                "stderr_path": str(result.stderr_path),
            }
            for result in results
        ],
        "artifacts": {
            "manifest": str(work_dir / "manifest.json"),
            "rref_shard": str(work_dir / "rref_8x8_train_smoke.npz"),
            "rref_checkpoint": str(work_dir / "rref_pivot_ckpt"),
            "rref_shard_benchmark": str(work_dir / "rref_shard_benchmark_smoke.json"),
            "hnf_v08_metrics": str(work_dir / "hnf_v08" / "metrics.json"),
            "snf_benchmark": str(work_dir / "snf_benchmark_smoke.json"),
            "report_metrics": str(work_dir / "report_smoke" / "metrics.json"),
        },
        "verdicts": _verdicts(parsed_outputs, work_dir),
    }


def _overall_status(parsed_outputs: dict[str, JsonDict]) -> str:
    hnf_status = _hnf_metrics_status(parsed_outputs)
    if hnf_status not in {"ok", "failed_threshold"}:
        return "failed"
    if parsed_outputs.get("rref_shard_benchmark", {}).get("status") != "ok":
        return "failed"
    if "neural" not in parsed_outputs.get("rref_shard_benchmark", {}).get("policies", {}):
        return "failed"
    if parsed_outputs.get("snf_benchmark", {}).get("status") != "ok":
        return "failed"
    if _policy_success_rate(parsed_outputs["snf_benchmark"], "certificate_replay") != 1.0:
        return "failed"
    if parsed_outputs.get("report_smoke", {}).get("status") != "ok":
        return "failed"
    return "ok"


def _verdicts(parsed_outputs: dict[str, JsonDict], work_dir: Path) -> JsonDict:
    rref_benchmark = parsed_outputs.get("rref_shard_benchmark", {})
    snf_benchmark = parsed_outputs.get("snf_benchmark", {})
    report_smoke = parsed_outputs.get("report_smoke", {})
    hnf_metrics = _read_json_if_exists(work_dir / "hnf_v08" / "metrics.json")
    return {
        "rref_rollout": _compact_status(parsed_outputs.get("rref_rollout", {})),
        "rref_shard_benchmark": _policy_summary(rref_benchmark),
        "hnf_v08": {
            "status": hnf_metrics.get("status", "missing"),
            "threshold_passed": hnf_metrics.get("threshold_verdict", {}).get("passed"),
            "benchmark_count": hnf_metrics.get("benchmark_count"),
        },
        "snf_benchmark": _policy_summary(snf_benchmark),
        "report_smoke": {
            "status": report_smoke.get("status", "missing"),
            "mode": report_smoke.get("mode"),
            "suite": report_smoke.get("suite"),
            "plot_count": report_smoke.get("plot_count"),
        },
    }


def _compact_status(payload: JsonDict) -> JsonDict:
    return {
        "status": payload.get("status", "missing"),
        "success": payload.get("success"),
        "step_count": payload.get("step_count"),
        "invalid_action_count": payload.get("invalid_action_count"),
        "masked_action_count": payload.get("masked_action_count"),
        "final_is_rref": payload.get("final_is_rref"),
    }


def _policy_summary(payload: JsonDict) -> JsonDict:
    policies = payload.get("policies")
    if not isinstance(policies, dict):
        return {"status": payload.get("status", "missing"), "policies": {}}
    summaries: JsonDict = {}
    for name, policy in policies.items():
        if isinstance(policy, dict) and isinstance(policy.get("aggregate"), dict):
            aggregate = policy["aggregate"]
            summaries[str(name)] = {
                "sample_count": aggregate.get("sample_count"),
                "success_count": aggregate.get("success_count"),
                "success_rate": aggregate.get("success_rate"),
                "status_counts": aggregate.get("status_counts"),
            }
    return {
        "status": payload.get("status", "missing"),
        "source": payload.get("source"),
        "count": payload.get("count"),
        "policies": summaries,
    }


def _policy_success_rate(payload: JsonDict, policy_name: str) -> float | None:
    policies = payload.get("policies")
    if not isinstance(policies, dict):
        return None
    policy = policies.get(policy_name)
    if not isinstance(policy, dict):
        return None
    aggregate = policy.get("aggregate")
    if not isinstance(aggregate, dict):
        return None
    success_rate = aggregate.get("success_rate")
    return float(success_rate) if isinstance(success_rate, int | float) else None


def _hnf_metrics_status(parsed_outputs: dict[str, JsonDict]) -> str:
    hnf = parsed_outputs.get("hnf_v08_experiment", {})
    status = hnf.get("status", "missing")
    return str(status)


def _write_fixtures(
    fixture_dir: Path,
    manifest: JsonDict,
    work_dir: Path,
    parsed_outputs: dict[str, JsonDict],
) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixtures = {
        "manifest.json": manifest,
        "hnf_v08_metrics_smoke.json": _read_json_if_exists(work_dir / "hnf_v08" / "metrics.json"),
        "report_metrics_smoke.json": _read_json_if_exists(
            work_dir / "report_smoke" / "metrics.json"
        ),
    }
    if "rref_shard_benchmark" in parsed_outputs:
        fixtures["rref_shard_benchmark_smoke.json"] = parsed_outputs["rref_shard_benchmark"]
    if "snf_benchmark" in parsed_outputs:
        fixtures["snf_benchmark_smoke.json"] = parsed_outputs["snf_benchmark"]
    for filename, payload in fixtures.items():
        _write_json(fixture_dir / filename, _sanitize_compact(payload))


def _sanitize_compact(value: object, key: str | None = None) -> object:
    if isinstance(value, dict):
        sanitized: JsonDict = {}
        for child_key, child_value in value.items():
            if child_key in FORBIDDEN_COMPACT_KEYS:
                continue
            sanitized[child_key] = _sanitize_compact(child_value, child_key)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_compact(child, key) for child in value]
    if key is not None and key.endswith("_seconds"):
        return 0.0
    if key == "generated_at_utc":
        return "normalized"
    return value


def _read_json_if_exists(path: Path) -> JsonDict:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected object JSON at {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _command_stdout(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return "unavailable"
    return completed.stdout.strip()


def _relative_cwd(cwd: Path, repo_root: Path) -> str:
    try:
        return str(cwd.relative_to(repo_root)) or "."
    except ValueError:
        return str(cwd)


if __name__ == "__main__":
    raise SystemExit(main())
