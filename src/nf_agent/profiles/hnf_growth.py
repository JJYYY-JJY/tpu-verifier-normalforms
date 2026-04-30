from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
from time import perf_counter
from typing import Any

from nf_agent.data.hnf_shards import (
    BACKWARD_SCHEMA_VERSION,
    load_hnf_backward_shard,
    load_hnf_backward_shard_config,
    write_hnf_backward_shard,
)
from nf_agent.env.elementary_ops import Matrix
from nf_agent.env.hnf_int import (
    IntegerRowOp,
    is_row_hnf,
    normalize_integer_matrix,
    replay_integer_row_ops,
    row_hnf,
)

JsonDict = dict[str, Any]
_SEARCH_POLICY = "row_preconditioned_row_hnf"
_SEARCH_MODE = "exact_row_preconditioned"
_OBJECTIVE_KEYS = (
    "max_bitlength",
    "max_abs_seen",
    "fill_in_delta",
    "step_count",
    "certificate_size_entries",
)


@dataclass(frozen=True)
class HNFGrowthProfileConfig:
    config_path: str | Path
    work_dir: str | Path
    out_dir: str | Path
    family: str
    count: int
    seed_start: int = 0
    candidate_limit: int | None = None


@dataclass(frozen=True)
class HNFGrowthCandidateSummary:
    candidate_index: int
    policy: str
    status: str
    success: bool
    replay_ok: bool
    predicate_ok: bool
    metrics: JsonDict


@dataclass(frozen=True)
class _HNFGrowthCandidateEvaluation:
    summary: HNFGrowthCandidateSummary
    ops: tuple[IntegerRowOp, ...]
    final_matrix: Matrix


@dataclass(frozen=True)
class HNFGrowthSearchResult:
    baseline: HNFGrowthCandidateSummary
    best: HNFGrowthCandidateSummary
    best_candidate: int
    best_policy: str
    candidate_count: int
    rejected_candidate_count: int
    improved_metrics: list[str]
    best_ops: tuple[IntegerRowOp, ...]
    best_final_matrix: Matrix


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
    initial_max_abs = _max_abs(current)
    max_abs_seen = initial_max_abs
    initial_density = _density(current)
    for op in ops:
        current = replay_integer_row_ops(current, [op])
        max_abs_seen = max(max_abs_seen, _max_abs(current))
    final_density = _density(current)
    return {
        "step_count": len(ops),
        "initial_max_abs": initial_max_abs,
        "max_abs_seen": max_abs_seen,
        "initial_bitlength": initial_max_abs.bit_length(),
        "max_bitlength": max_abs_seen.bit_length(),
        "growth_numerator": max_abs_seen,
        "growth_denominator": max(1, initial_max_abs),
        "fill_in_delta": final_density - initial_density,
        "certificate_op_count": len(ops),
        "certificate_size_entries": 4 * len(ops),
    }


def _candidate_to_json(summary: HNFGrowthCandidateSummary) -> JsonDict:
    return {
        "candidate_index": summary.candidate_index,
        "policy": summary.policy,
        "status": summary.status,
        "success": summary.success,
        "replay_ok": summary.replay_ok,
        "predicate_ok": summary.predicate_ok,
        **summary.metrics,
    }


def _permutation_to_swap_ops(permutation: Sequence[int]) -> tuple[IntegerRowOp, ...]:
    current = list(range(len(permutation)))
    ops: list[IntegerRowOp] = []
    for target, wanted_source in enumerate(permutation):
        source = current.index(wanted_source)
        if target == source:
            continue
        ops.append(IntegerRowOp.swap(target, source))
        current[target], current[source] = current[source], current[target]
    return tuple(ops)


def _row_preconditioning_candidates(
    row_count: int,
    candidate_limit: int,
) -> list[tuple[IntegerRowOp, ...]]:
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive")

    base = tuple(range(row_count))
    seen: set[tuple[int, ...]] = set()
    candidates: list[tuple[IntegerRowOp, ...]] = []

    def add_permutation(permutation: Iterable[int]) -> None:
        if len(candidates) >= candidate_limit:
            return
        key = tuple(permutation)
        if key in seen:
            return
        seen.add(key)
        candidates.append(_permutation_to_swap_ops(key))

    add_permutation(base)
    for left, right in combinations(range(row_count), 2):
        swapped = list(base)
        swapped[left], swapped[right] = swapped[right], swapped[left]
        add_permutation(swapped)
    for shift in range(1, row_count):
        add_permutation((*base[shift:], *base[:shift]))
    if row_count > 0:
        add_permutation(reversed(base))
    for ordered in permutations(base):
        if len(candidates) >= candidate_limit:
            break
        add_permutation(ordered)
    return candidates


def _evaluate_candidate(
    input_matrix: Matrix,
    candidate_index: int,
    precondition_ops: Sequence[IntegerRowOp],
) -> _HNFGrowthCandidateEvaluation | None:
    ops_prefix = tuple(precondition_ops)
    policy = "row_hnf" if candidate_index == 0 and not ops_prefix else _SEARCH_POLICY
    try:
        preconditioned = replay_integer_row_ops(input_matrix, ops_prefix)
        result = row_hnf(preconditioned)
        ops = (*ops_prefix, *result.ops)
        replay_ok = replay_integer_row_ops(input_matrix, ops) == result.final_matrix
        predicate_ok = is_row_hnf(result.final_matrix)
    except (IndexError, TypeError, ValueError):
        return None

    success = replay_ok and predicate_ok
    if not success:
        return None

    summary = HNFGrowthCandidateSummary(
        candidate_index=candidate_index,
        policy=policy,
        status="success",
        success=True,
        replay_ok=True,
        predicate_ok=True,
        metrics=_growth_metrics(input_matrix, ops),
    )
    return _HNFGrowthCandidateEvaluation(
        summary=summary,
        ops=ops,
        final_matrix=result.final_matrix,
    )


def _objective_key(
    candidate: _HNFGrowthCandidateEvaluation,
) -> tuple[int, int, float, int, int, int]:
    metrics = candidate.summary.metrics
    return (
        int(metrics["max_bitlength"]),
        int(metrics["max_abs_seen"]),
        float(metrics["fill_in_delta"]),
        int(metrics["step_count"]),
        int(metrics["certificate_size_entries"]),
        candidate.summary.candidate_index,
    )


def _improved_metrics(
    baseline: HNFGrowthCandidateSummary,
    best: HNFGrowthCandidateSummary,
) -> list[str]:
    improved: list[str] = []
    for key in _OBJECTIVE_KEYS:
        if float(best.metrics[key]) < float(baseline.metrics[key]):
            improved.append(key)
    return improved


def _search_row_preconditioned_row_hnf(
    matrix: Sequence[Sequence[int]],
    *,
    candidate_limit: int,
    candidate_preconditioners: Sequence[Sequence[IntegerRowOp]] | None = None,
) -> HNFGrowthSearchResult:
    input_matrix = normalize_integer_matrix(matrix)
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive")
    if candidate_preconditioners is None:
        preconditioners = _row_preconditioning_candidates(len(input_matrix), candidate_limit)
    else:
        preconditioners = [
            tuple(candidate)
            for candidate in candidate_preconditioners[:candidate_limit]
        ]

    evaluations: list[_HNFGrowthCandidateEvaluation] = []
    rejected_candidate_count = 0
    for candidate_index, precondition_ops in enumerate(preconditioners):
        evaluation = _evaluate_candidate(input_matrix, candidate_index, precondition_ops)
        if evaluation is None:
            rejected_candidate_count += 1
            continue
        evaluations.append(evaluation)

    if not evaluations:
        raise ValueError("no valid HNF growth-search candidates")
    baseline = next(
        (
            evaluation
            for evaluation in evaluations
            if evaluation.summary.candidate_index == 0 and evaluation.summary.policy == "row_hnf"
        ),
        None,
    )
    if baseline is None:
        raise ValueError("candidate 0 must be the row_hnf baseline")

    best = min(evaluations, key=_objective_key)
    return HNFGrowthSearchResult(
        baseline=baseline.summary,
        best=best.summary,
        best_candidate=best.summary.candidate_index,
        best_policy=best.summary.policy,
        candidate_count=len(preconditioners),
        rejected_candidate_count=rejected_candidate_count,
        improved_metrics=_improved_metrics(baseline.summary, best.summary),
        best_ops=best.ops,
        best_final_matrix=best.final_matrix,
    )


def _mean(samples: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(sample[key]) for sample in samples if key in sample]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _max_int(samples: Sequence[Mapping[str, Any]], key: str) -> int:
    return max((int(sample[key]) for sample in samples if key in sample), default=0)


def _resolve_candidate_limit(
    config: HNFGrowthProfileConfig,
    shard_config_candidate_limit: int,
) -> int:
    if config.candidate_limit is not None:
        if config.candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")
        return config.candidate_limit
    return shard_config_candidate_limit


def _summarize_shard(shard_path: Path, *, family: str, candidate_limit: int) -> JsonDict:
    arrays, metadata = load_hnf_backward_shard(shard_path)
    samples: list[JsonDict] = []
    for sample_index in range(int(arrays["inputs"].shape[0])):
        input_matrix = arrays["inputs"][sample_index].tolist()
        search_result = _search_row_preconditioned_row_hnf(
            input_matrix,
            candidate_limit=candidate_limit,
        )
        success = search_result.best.success
        sample: JsonDict = {
            "sample_index": sample_index,
            "seed": int(metadata["seed_start"]) + sample_index,
            "status": "success" if success else "verification_failed",
            "success": success,
            "replay_ok": search_result.best.replay_ok,
            "predicate_ok": search_result.best.predicate_ok,
            "best_policy": search_result.best_policy,
            "best_candidate": search_result.best_candidate,
            "candidate_count": search_result.candidate_count,
            "rejected_candidate_count": search_result.rejected_candidate_count,
            "improved_metrics": search_result.improved_metrics,
            "baseline": _candidate_to_json(search_result.baseline),
            "best": _candidate_to_json(search_result.best),
            **search_result.best.metrics,
        }
        samples.append(sample)

    success_count = sum(1 for sample in samples if bool(sample["success"]))
    improved_sample_count = sum(1 for sample in samples if sample["improved_metrics"])
    improved_metric_counts = {
        key: sum(1 for sample in samples if key in sample["improved_metrics"])
        for key in _OBJECTIVE_KEYS
    }
    aggregate: JsonDict = {
        "sample_count": len(samples),
        "success_count": success_count,
        "success_rate": success_count / len(samples) if samples else 0.0,
        "improved_sample_count": improved_sample_count,
        "improvement_rate": improved_sample_count / len(samples) if samples else 0.0,
        "improved_metric_counts": improved_metric_counts,
        "v1_1_target_met": improved_sample_count > 0,
        "candidate_count": sum(int(sample["candidate_count"]) for sample in samples),
        "rejected_candidate_count": sum(
            int(sample["rejected_candidate_count"]) for sample in samples
        ),
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
        "profile_version": "v1.1-beta",
        "source_schema_version": metadata["schema_version"],
        "status": "ok" if success_count == len(samples) else "verification_failed",
        "family": family,
        "count": len(samples),
        "shape": shape,
        "seed_start": metadata["seed_start"],
        "seed_stop_exclusive": metadata["seed_stop_exclusive"],
        "shard_path": str(shard_path),
        "search": {
            "mode": _SEARCH_MODE,
            "policy": _SEARCH_POLICY,
            "baseline_policy": "row_hnf",
            "candidate_limit": candidate_limit,
            "objective": list(_OBJECTIVE_KEYS),
        },
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
            f"- Search policy: `{summary['search']['policy']}`",
            f"- Candidate limit: `{summary['search']['candidate_limit']}`",
            f"- Improved samples: `{aggregate['improved_sample_count']}`",
            f"- Improvement rate: `{aggregate['improvement_rate']}`",
            "- v1.1 beta target: "
            f"`{'met' if aggregate['v1_1_target_met'] else 'not met'}`",
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
    resolved_candidate_limit = _resolve_candidate_limit(
        config,
        shard_config.search_candidate_limit
        or shard_config.rollout_beam_size
        or 32,
    )
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
    summary = _summarize_shard(
        shard_path,
        family=config.family,
        candidate_limit=resolved_candidate_limit,
    )
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
