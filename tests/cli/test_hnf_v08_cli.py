import json
from pathlib import Path

from click.testing import CliRunner

from nf_agent.cli import main
from nf_agent.data.hnf_shards import write_hnf_shard


def _config(tmp_path: Path) -> Path:
    config_path = tmp_path / "hnf.yaml"
    config_path.write_text(
        "task: hnf\n"
        "integer_matrix:\n"
        "  family: sparse\n"
        "  rows: 3\n"
        "  cols: 3\n"
        "  density: 0.5\n"
        "  entry_bound: 4\n"
    )
    return config_path


def test_hnf_v08_data_train_rollout_cli_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    shard_path = tmp_path / "hnf_cli.npz"
    ckpt_dir = tmp_path / "ckpt"

    made = runner.invoke(
        main,
        [
            "data",
            "make-hnf-shard",
            "--config",
            str(_config(tmp_path)),
            "--count",
            "4",
            "--seed-start",
            "0",
            "--out",
            str(shard_path),
        ],
    )
    assert made.exit_code == 0, made.output
    assert json.loads(made.output)["status"] == "ok"

    trained = runner.invoke(
        main,
        [
            "train",
            "hnf-policy",
            "--data",
            str(shard_path),
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--hidden-size",
            "16",
            "--out",
            str(ckpt_dir),
        ],
    )
    assert trained.exit_code == 0, trained.output
    assert json.loads(trained.output)["status"] == "ok"

    greedy = runner.invoke(
        main,
        [
            "rollout",
            "hnf-neural",
            "--data",
            str(shard_path),
            "--checkpoint",
            str(ckpt_dir),
            "--sample-index",
            "0",
            "--max-steps",
            "2",
            "--hidden-size",
            "16",
        ],
    )
    assert greedy.exit_code == 0, greedy.output
    assert json.loads(greedy.output)["status"] in {
        "success",
        "max_steps_exceeded",
        "invalid_action",
    }

    beam = runner.invoke(
        main,
        [
            "rollout",
            "hnf-beam",
            "--data",
            str(shard_path),
            "--checkpoint",
            str(ckpt_dir),
            "--sample-index",
            "0",
            "--max-steps",
            "2",
            "--beam-width",
            "2",
            "--hidden-size",
            "16",
        ],
    )
    assert beam.exit_code == 0, beam.output
    assert json.loads(beam.output)["status"] in {"success", "max_steps_exceeded", "invalid_action"}


def test_hnf_dagger_actor_critic_benchmark_and_experiment_cli_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    shard_path = tmp_path / "hnf_cli.npz"
    write_hnf_shard(_config(tmp_path), count=4, seed_start=0, out_path=shard_path)

    dagger_dir = tmp_path / "dagger"
    dagger = runner.invoke(
        main,
        [
            "train",
            "hnf-dagger",
            "--data",
            str(shard_path),
            "--iterations",
            "1",
            "--train-steps",
            "1",
            "--batch-size",
            "2",
            "--rollout-sample-count",
            "2",
            "--rollout-max-steps",
            "1",
            "--hidden-size",
            "16",
            "--out",
            str(dagger_dir),
        ],
    )
    assert dagger.exit_code == 0, dagger.output
    dagger_payload = json.loads(dagger.output)

    actor = runner.invoke(
        main,
        [
            "train",
            "hnf-actor-critic",
            "--data",
            dagger_payload["aggregate_data_path"],
            "--checkpoint",
            dagger_payload["checkpoint_dir"],
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--rollout-max-steps",
            "2",
            "--hidden-size",
            "16",
            "--out",
            str(tmp_path / "actor_critic"),
        ],
    )
    assert actor.exit_code == 0, actor.output
    assert json.loads(actor.output)["status"] == "ok"

    bench = runner.invoke(
        main,
        [
            "benchmark",
            "hnf",
            "--rows",
            "3",
            "--cols",
            "3",
            "--count",
            "2",
            "--density",
            "0.5",
            "--entry-bound",
            "4",
            "--seed-start",
            "0",
            "--supervised-checkpoint",
            dagger_payload["checkpoint_dir"],
            "--model-data",
            dagger_payload["aggregate_data_path"],
            "--hidden-size",
            "16",
            "--max-steps",
            "2",
        ],
    )
    assert bench.exit_code == 0, bench.output
    bench_payload = json.loads(bench.output)
    assert "policies" in bench_payload
    assert "row_hnf" in bench_payload["policies"]
    assert "supervised_greedy" in bench_payload["policies"]
    assert bench_payload["aggregate"] == bench_payload["policies"]["row_hnf"]["aggregate"]

    experiment = runner.invoke(
        main,
        [
            "experiment",
            "hnf-v08",
            "--out-dir",
            str(tmp_path / "experiment"),
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
    )
    assert experiment.exit_code == 0, experiment.output
    experiment_payload = json.loads(experiment.output)
    assert experiment_payload["status"] in {"ok", "failed_threshold"}
    assert Path(experiment_payload["report_md"]).exists()
    assert Path(experiment_payload["metrics_json"]).exists()
