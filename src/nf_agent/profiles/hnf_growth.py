from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from nf_agent.data.hnf_shards import (
    BACKWARD_SCHEMA_VERSION,
    integer_row_ops_from_hnf_shard_arrays,
    load_hnf_backward_shard,
    load_hnf_backward_shard_config,
    write_hnf_backward_shard,
)
from nf_agent.env.elementary_ops import Matrix
from nf_agent.env.hnf_int import IntegerRowOp, is_row_hnf, replay_integer_row_ops

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class HNFGrowthProfileConfig:
    config_path: str | Path
    work_dir: str | Path
    out_dir: str | Path
    family: str
    count: int
    seed_start: int = 0


def _max_abs(matrix: Sequence[Sequence[int]]) -> int:
    return max((abs(entry) for row in matrix for entry in row), default=0)


def _density(matrix: Sequence[Sequence[int]]) -> float:
    total = sum(len(row) for row in matrix)
    if total == 0:
        return 0.0
    nonzero = sum(1 for row in matrix for entry in row if entry != 0)
    return nonzero / total


def _growth_metrics(input_matrix: Matrix, ops: Sequence[IntegerRowOp]) -> JsonDict:
    current = [[entry for entry in row] for row in input_matrix]
    max_abs_seen = _max_abs(current)
    initial_density = _density(current)
    for op in ops:
        current = replay_integer_row_ops(current, [op])
        max_abs_seen = max(max_abs_seen, _max_abs(current))
    final_density = _density(current)
    return {
        "step_count": len(ops),
        "max_abs_seen": max_abs_seen,
        "max_bitlength": max_abs_seen.bit_length(),
        "fill_in_delta": final_density - initial_density,
        "certificate_op_count": len(ops),
        "certificate_size_entries": 4 * len(ops),
    }


def _mean(samples: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(sample[key]) for sample in samples if key in sample]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _max_int(samples: Sequence[Mapping[str, Any]], key: str) -> int:
    return max((int(sample[key]) for sample in samples if key in sample), default=0)


def _summarize_shard(shard_path: Path, *, family: str) -> JsonDict:
    arrays, metadata = load_hnf_backward_shard(shard_path)
    samples: list[JsonDict] = []
    for sample_index in range(int(arrays["inputs"].shape[0])):
        input_matrix = arrays["inputs"][sample_index].tolist()
        final_matrix = arrays["finals"][sample_index].tolist()
        ops = integer_row_ops_from_hnf_shard_arrays(arrays, sample_index)
        replay_ok = replay_integer_row_ops(input_matrix, ops) == final_matrix
        predicate_ok = is_row_hnf(final_matrix)
        success = replay_ok and predicate_ok
        sample: JsonDict = {
            "sample_index": sample_index,
            "seed": int(metadata["seed_start"]) + sample_index,
            "status": "success" if success else "verification_failed",
            "success": success,
            "replay_ok": replay_ok,
            "predicate_ok": predicate_ok,
            **_growth_metrics(input_matrix, ops),
        }
        samples.append(sample)

    success_count = sum(1 for sample in samples if bool(sample["success"]))
    aggregate: JsonDict = {
        "sample_count": len(samples),
        "success_count": success_count,
        "success_rate": success_count / len(samples) if samples else 0.0,
        "mean_step_count": _mean(samples, "step_count"),
        "mean_fill_in_delta": _mean(samples, "fill_in_delta"),
        "mean_certificate_op_count": _mean(samples, "certificate_op_count"),
        "mean_certificate_size_entries": _mean(samples, "certificate_size_entries"),
        "max_step_count": _max_int(samples, "step_count"),
        "max_bitlength": _max_int(samples, "max_bitlength"),
        "max_abs_seen": _max_int(samples, "max_abs_seen"),
        "max_certificate_op_count": _max_int(samples, "certificate_op_count"),
        "max_certificate_size_entries": _max_int(samples, "certificate_size_entries"),
    }
    shape = metadata["shape"]
    return {
        "schema_version": "hnf-growth-profile-v1",
        "source_schema_version": metadata["schema_version"],
        "status": "ok" if success_count == len(samples) else "verification_failed",
        "family": family,
        "count": len(samples),
        "shape": shape,
        "seed_start": metadata["seed_start"],
        "seed_stop_exclusive": metadata["seed_stop_exclusive"],
        "shard_path": str(shard_path),
        "aggregate": aggregate,
        "samples": samples,
        "validation": {
            "verifier": "replay_integer_row_ops",
            "predicate": "is_row_hnf",
            "require_no_fallback": True,
        },
    }


def render_hnf_growth_report(summary: JsonDict) -> str:
    aggregate = summary["aggregate"]
    shape = summary["shape"]
    return "\n".join(
        [
            "# HNF Growth Profile",
            "",
            f"- Status: `{summary['status']}`",
            f"- Family: `{summary['family']}`",
            f"- Shape: `{shape['rows']}x{shape['cols']}`",
            f"- Count: `{summary['count']}`",
            f"- Source schema: `{summary['source_schema_version']}`",
            f"- Success rate: `{aggregate['success_rate']}`",
            f"- Mean step count: `{aggregate['mean_step_count']}`",
            f"- Max bitlength: `{aggregate['max_bitlength']}`",
            f"- Max abs seen: `{aggregate['max_abs_seen']}`",
            f"- Mean fill-in delta: `{aggregate['mean_fill_in_delta']}`",
            f"- Max certificate entries: `{aggregate['max_certificate_size_entries']}`",
            f"- Wall time seconds: `{aggregate['wall_time_seconds']}`",
            "",
            "Exact replay and row-HNF predicate checks are authoritative. No hidden fallback.",
            "",
        ]
    )


def write_hnf_growth_profile(config: HNFGrowthProfileConfig) -> JsonDict:
    if config.count <= 0:
        raise ValueError("count must be positive")
    if not isinstance(config.seed_start, int) or isinstance(config.seed_start, bool):
        raise ValueError("seed_start must be an integer")
    shard_config = load_hnf_backward_shard_config(config.config_path)
    shard_format = shard_config.storage_format
    work_dir = Path(config.work_dir)
    out_dir = Path(config.out_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = (
        work_dir
        / f"hnf_backward_{config.family}_seed{config.seed_start}_count{config.count}.{shard_format}"
    )

    start = perf_counter()
    write_hnf_backward_shard(
        config_path=config.config_path,
        family=config.family,
        count=config.count,
        seed_start=config.seed_start,
        out_path=shard_path,
    )
    summary = _summarize_shard(shard_path, family=config.family)
    wall_time_seconds = perf_counter() - start
    summary["aggregate"]["wall_time_seconds"] = wall_time_seconds
    summary["wall_time_seconds"] = wall_time_seconds
    if summary["source_schema_version"] != BACKWARD_SCHEMA_VERSION:
        raise ValueError("HNF growth profile loaded an unexpected shard schema")

    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_hnf_growth_report(summary), encoding="utf-8")
    return {
        "status": summary["status"],
        "schema_version": summary["schema_version"],
        "source_schema_version": summary["source_schema_version"],
        "family": summary["family"],
        "summary_json": str(summary_path),
        "report_md": str(report_path),
        "shard_path": str(shard_path),
        "count": summary["count"],
    }
