from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

from nf_agent.benchmarks import (
    HNFBenchmarkConfig,
    RREFBenchmarkConfig,
    run_hnf_benchmark,
    run_rref_benchmark,
)

ReportKind = Literal["rref", "hnf"]
ReportMode = Literal["run", "summary"]
JsonDict = dict[str, Any]

_FORBIDDEN_SAMPLE_FIELDS = {
    "final",
    "final_matrix",
    "input",
    "initial_matrix",
    "matrix",
    "ops",
}


@dataclass(frozen=True)
class BenchmarkReportConfig:
    out_dir: str | Path
    input_json_paths: tuple[str | Path, ...] = ()
    suite: str = "paper-smoke"
    sample_count: int = 16
    rows: int = 8
    cols: int = 8
    modulus: int = 101
    seed_start: int = 0
    sparse_density: float = 0.2
    low_rank: int = 3
    hnf_entry_bound: int = 9
    rref_checkpoint: str | Path | None = None
    rref_model_data: str | Path | None = None


@dataclass(frozen=True)
class BenchmarkEntry:
    kind: ReportKind
    label: str
    payload: JsonDict
    metadata: JsonDict
    source_path: str | None = None


def build_benchmark_report(config: BenchmarkReportConfig) -> JsonDict:
    out_dir = Path(config.out_dir)
    mode: ReportMode = "summary" if config.input_json_paths else "run"
    entries = _load_entries(config) if mode == "summary" else _run_paper_smoke_suite(config)
    return _write_report(config=config, mode=mode, out_dir=out_dir, entries=entries)


def _run_paper_smoke_suite(config: BenchmarkReportConfig) -> list[BenchmarkEntry]:
    _validate_run_config(config)

    rref_dense = run_rref_benchmark(_rref_generated_config(config, family="dense"))
    rref_sparse = run_rref_benchmark(
        _rref_generated_config(config, family="sparse", density=config.sparse_density)
    )
    rref_low_rank = run_rref_benchmark(
        _rref_generated_config(config, family="low_rank", rank=config.low_rank)
    )
    hnf_sparse = run_hnf_benchmark(
        HNFBenchmarkConfig(
            count=config.sample_count,
            rows=config.rows,
            cols=config.cols,
            density=config.sparse_density,
            entry_bound=config.hnf_entry_bound,
            seed_start=config.seed_start,
        )
    )

    return [
        BenchmarkEntry(
            kind="rref",
            label="rref_generated_dense",
            payload=rref_dense,
            metadata={"family": "dense"},
        ),
        BenchmarkEntry(
            kind="rref",
            label="rref_generated_sparse",
            payload=rref_sparse,
            metadata={"family": "sparse", "density": config.sparse_density},
        ),
        BenchmarkEntry(
            kind="rref",
            label="rref_generated_low_rank",
            payload=rref_low_rank,
            metadata={"family": "low_rank", "rank": config.low_rank},
        ),
        BenchmarkEntry(
            kind="hnf",
            label="hnf_generated_sparse_integer",
            payload=hnf_sparse,
            metadata={"family": "sparse_integer"},
        ),
    ]


def _rref_generated_config(
    config: BenchmarkReportConfig,
    *,
    family: Literal["dense", "sparse", "low_rank"],
    density: float | None = None,
    rank: int | None = None,
) -> RREFBenchmarkConfig:
    return RREFBenchmarkConfig(
        source="generated",
        count=config.sample_count,
        rows=config.rows,
        cols=config.cols,
        modulus=config.modulus,
        family=family,
        seed_start=config.seed_start,
        density=density,
        rank=rank,
        model_data_path=config.rref_model_data,
        checkpoint_dir=config.rref_checkpoint,
    )


def _validate_run_config(config: BenchmarkReportConfig) -> None:
    if config.suite != "paper-smoke":
        raise ValueError(f"unsupported benchmark report suite: {config.suite}")
    _validate_positive_int(config.sample_count, "sample_count")
    _validate_positive_int(config.rows, "rows")
    _validate_positive_int(config.cols, "cols")
    _validate_positive_int(config.modulus, "p")
    _validate_positive_int(config.low_rank, "low_rank")
    _validate_positive_int(config.hnf_entry_bound, "hnf_entry_bound")
    if not 0.0 <= float(config.sparse_density) <= 1.0:
        raise ValueError("sparse_density must lie in [0, 1]")
    if config.low_rank > min(config.rows, config.cols):
        raise ValueError("low_rank must be at most min(rows, cols)")
    if (config.rref_checkpoint is None) != (config.rref_model_data is None):
        raise ValueError("rref_checkpoint and rref_model_data must be supplied together")


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _load_entries(config: BenchmarkReportConfig) -> list[BenchmarkEntry]:
    return [
        _load_entry(Path(path), index)
        for index, path in enumerate(config.input_json_paths, start=1)
    ]


def _load_entry(path: Path, index: int) -> BenchmarkEntry:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid benchmark JSON {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported benchmark JSON {path}: expected object")
    entry = _entry_from_payload(
        payload=cast(JsonDict, payload),
        label=f"{path.stem or f'input_{index}'}",
        source_path=str(path),
    )
    _validate_compact_entry(entry)
    return entry


def _entry_from_payload(payload: JsonDict, label: str, source_path: str | None) -> BenchmarkEntry:
    if _looks_like_rref_benchmark(payload):
        return BenchmarkEntry(
            kind="rref",
            label=label,
            payload=payload,
            metadata={"family": payload.get("family", "unknown")},
            source_path=source_path,
        )
    if _looks_like_hnf_benchmark(payload):
        return BenchmarkEntry(
            kind="hnf",
            label=label,
            payload=payload,
            metadata={"family": payload.get("family", "unknown")},
            source_path=source_path,
        )
    origin = f" {source_path}" if source_path is not None else ""
    raise ValueError(f"unsupported benchmark JSON{origin}")


def _looks_like_rref_benchmark(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("status") == "ok"
        and isinstance(payload.get("policies"), dict)
        and "modulus" in payload
        and "rows" in payload
        and "cols" in payload
    )


def _looks_like_hnf_benchmark(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("status") == "ok"
        and payload.get("family") == "sparse_integer"
        and (
            isinstance(payload.get("policies"), dict)
            or (
                isinstance(payload.get("aggregate"), dict)
                and isinstance(payload.get("samples"), list)
            )
        )
    )


def _validate_compact_entry(entry: BenchmarkEntry) -> None:
    for sample in _sample_records(entry):
        for key in _FORBIDDEN_SAMPLE_FIELDS:
            if key in sample:
                raise ValueError(f"compact benchmark JSON must not include sample field: {key}")


def _sample_records(entry: BenchmarkEntry) -> list[JsonDict]:
    if entry.kind == "hnf":
        policies = entry.payload.get("policies")
        if isinstance(policies, dict):
            hnf_records: list[JsonDict] = []
            for policy_name, policy_value in policies.items():
                policy = _as_dict(policy_value, f"{entry.label}.policies.{policy_name}")
                samples = _required_list(policy, "samples", f"{entry.label}.{policy_name}")
                hnf_records.extend(
                    _as_dict(sample, f"{entry.label}.{policy_name}.samples[]")
                    for sample in samples
                )
            return hnf_records
        samples = _required_list(entry.payload, "samples", entry.label)
        return [_as_dict(sample, f"{entry.label}.samples[]") for sample in samples]

    policies = _required_dict(entry.payload, "policies", entry.label)
    records: list[JsonDict] = []
    for policy_name, policy_value in policies.items():
        policy = _as_dict(policy_value, f"{entry.label}.policies.{policy_name}")
        samples = _required_list(policy, "samples", f"{entry.label}.{policy_name}")
        records.extend(
            _as_dict(sample, f"{entry.label}.{policy_name}.samples[]") for sample in samples
        )
    return records


def _write_report(
    *,
    config: BenchmarkReportConfig,
    mode: ReportMode,
    out_dir: Path,
    entries: Sequence[BenchmarkEntry],
) -> JsonDict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for existing_plot in plots_dir.glob("*.png"):
        existing_plot.unlink()

    suite_rows = _suite_rows(entries)
    normalized_rows = _normalized_rows(entries)
    hnf_coefficient_rows = _hnf_coefficient_rows(entries)
    plot_paths = _write_plots(
        out_dir=out_dir,
        normalized_rows=normalized_rows,
        hnf_coefficient_rows=hnf_coefficient_rows,
    )

    provenance = _provenance(config=config, mode=mode)
    artifacts: JsonDict = {
        "report": "report.md",
        "metrics": "metrics.json",
        "plots": plot_paths,
    }
    metrics: JsonDict = {
        "schema_version": 1,
        "provenance": provenance,
        "artifacts": artifacts,
        "benchmarks": [_entry_json(entry) for entry in entries],
        "suite_rows": suite_rows,
        "normalized_rows": normalized_rows,
        "hnf_coefficient_rows": hnf_coefficient_rows,
    }

    report_text = _render_markdown(
        provenance=provenance,
        suite_rows=suite_rows,
        normalized_rows=normalized_rows,
        hnf_coefficient_rows=hnf_coefficient_rows,
        plot_paths=plot_paths,
    )

    report_path = out_dir / "report.md"
    metrics_path = out_dir / "metrics.json"
    report_path.write_text(report_text)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    return {
        "status": "ok",
        "mode": mode,
        "suite": config.suite,
        "report_md": str(report_path),
        "metrics_json": str(metrics_path),
        "plot_count": len(plot_paths),
    }


def _provenance(config: BenchmarkReportConfig, mode: ReportMode) -> JsonDict:
    return {
        "mode": mode,
        "suite": config.suite,
        "generated_at_utc": datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "input_json_paths": [str(path) for path in config.input_json_paths],
        "run_config": {
            "sample_count": config.sample_count,
            "rows": config.rows,
            "cols": config.cols,
            "p": config.modulus,
            "seed_start": config.seed_start,
            "sparse_density": config.sparse_density,
            "low_rank": config.low_rank,
            "hnf_entry_bound": config.hnf_entry_bound,
            "rref_checkpoint": None
            if config.rref_checkpoint is None
            else str(config.rref_checkpoint),
            "rref_model_data": None
            if config.rref_model_data is None
            else str(config.rref_model_data),
        },
    }


def _entry_json(entry: BenchmarkEntry) -> JsonDict:
    result: JsonDict = {
        "kind": entry.kind,
        "label": entry.label,
        "metadata": entry.metadata,
        "payload": entry.payload,
    }
    if entry.source_path is not None:
        result["source_path"] = entry.source_path
    return result


def _suite_rows(entries: Sequence[BenchmarkEntry]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for entry in entries:
        payload = entry.payload
        if entry.kind == "rref":
            policies = _required_dict(payload, "policies", entry.label)
            rows.append(
                {
                    "benchmark": entry.label,
                    "kind": "rref",
                    "source": _string(payload.get("source", "unknown")),
                    "family": _string(entry.metadata.get("family", "unknown")),
                    "shape": f"{payload.get('rows', '?')}x{payload.get('cols', '?')}",
                    "samples": _int(payload.get("count", 0)),
                    "modulus": _int(payload.get("modulus", 0)),
                    "policies": ", ".join(sorted(str(key) for key in policies)),
                }
            )
        else:
            hnf_policies = entry.payload.get("policies")
            policy_names = (
                sorted(str(key) for key in hnf_policies)
                if isinstance(hnf_policies, dict)
                else ["row_hnf"]
            )
            rows.append(
                {
                    "benchmark": entry.label,
                    "kind": "hnf",
                    "source": _string(payload.get("source", "unknown")),
                    "family": _string(payload.get("family", "unknown")),
                    "shape": f"{payload.get('rows', '?')}x{payload.get('cols', '?')}",
                    "samples": _int(payload.get("count", 0)),
                    "modulus": "",
                    "policies": ", ".join(policy_names),
                }
            )
    return rows


def _normalized_rows(entries: Sequence[BenchmarkEntry]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for entry in entries:
        if entry.kind == "rref":
            rows.extend(_normalized_rref_rows(entry))
        else:
            rows.extend(_normalized_hnf_rows(entry))
    return rows


def _normalized_rref_rows(entry: BenchmarkEntry) -> list[JsonDict]:
    payload = entry.payload
    policies = _required_dict(payload, "policies", entry.label)
    rows: list[JsonDict] = []
    for policy_name, policy_value in policies.items():
        policy = _as_dict(policy_value, f"{entry.label}.policies.{policy_name}")
        aggregate = _required_dict(policy, "aggregate", f"{entry.label}.{policy_name}")
        row = _base_normalized_row(
            entry=entry,
            policy=str(policy_name),
            aggregate=aggregate,
            source=_string(payload.get("source", "unknown")),
            family=_string(entry.metadata.get("family", "unknown")),
        )
        row["modulus"] = _int(payload.get("modulus", 0))
        row["mean_step_or_trace_length"] = _float(
            aggregate.get("mean_step_count", aggregate.get("mean_trace_length", 0.0))
        )
        row["mean_rank"] = _float(aggregate.get("mean_rank", 0.0))
        row["mean_invalid_action_count"] = _float(
            aggregate.get("mean_invalid_action_count", 0.0)
        )
        row["mean_masked_action_count"] = _float(
            aggregate.get("mean_masked_action_count", 0.0)
        )
        rows.append(row)
    return rows


def _normalized_hnf_rows(entry: BenchmarkEntry) -> list[JsonDict]:
    policies = entry.payload.get("policies")
    if isinstance(policies, dict):
        rows: list[JsonDict] = []
        for policy_name, policy_value in policies.items():
            policy = _as_dict(policy_value, f"{entry.label}.policies.{policy_name}")
            rows.append(_normalized_hnf_row(entry, str(policy_name), policy))
        return rows
    return [_normalized_hnf_row(entry, "row_hnf", entry.payload)]


def _normalized_hnf_row(
    entry: BenchmarkEntry,
    policy_name: str,
    policy_payload: Mapping[str, Any],
) -> JsonDict:
    payload = entry.payload
    aggregate = _required_dict(policy_payload, "aggregate", f"{entry.label}.{policy_name}")
    row = _base_normalized_row(
        entry=entry,
        policy=policy_name,
        aggregate=aggregate,
        source=_string(payload.get("source", "unknown")),
        family=_string(payload.get("family", "unknown")),
    )
    row["modulus"] = ""
    row["mean_step_or_trace_length"] = _float(
        aggregate.get("mean_step_count", aggregate.get("mean_trace_length", 0.0))
    )
    row["mean_rank"] = ""
    row["mean_invalid_action_count"] = _float(aggregate.get("mean_invalid_action_count", 0.0))
    row["mean_masked_action_count"] = _float(aggregate.get("mean_masked_action_count", 0.0))
    return row


def _base_normalized_row(
    *,
    entry: BenchmarkEntry,
    policy: str,
    aggregate: Mapping[str, Any],
    source: str,
    family: str,
) -> JsonDict:
    return {
        "benchmark": entry.label,
        "kind": entry.kind,
        "source": source,
        "family": family,
        "policy": policy,
        "sample_count": _int(_required_value(aggregate, "sample_count", entry.label)),
        "success_count": _int(_required_value(aggregate, "success_count", entry.label)),
        "success_rate": _float(_required_value(aggregate, "success_rate", entry.label)),
        "status_counts": _required_dict(aggregate, "status_counts", entry.label),
        "mean_wall_time_seconds": _float(aggregate.get("mean_wall_time_seconds", 0.0)),
        "mean_fill_in_delta": _float(aggregate.get("mean_fill_in_delta", 0.0)),
        "mean_initial_density": _float(aggregate.get("mean_initial_density", 0.0)),
        "mean_final_density": _float(aggregate.get("mean_final_density", 0.0)),
        "mean_max_density": _float(aggregate.get("mean_max_density", 0.0)),
    }


def _hnf_coefficient_rows(entries: Sequence[BenchmarkEntry]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for entry in entries:
        if entry.kind != "hnf":
            continue
        policies = entry.payload.get("policies")
        if isinstance(policies, dict):
            items = [
                (str(policy_name), _as_dict(policy, f"{entry.label}.{policy_name}"))
                for policy_name, policy in policies.items()
            ]
        else:
            items = [("row_hnf", entry.payload)]
        for policy_name, policy in items:
            aggregate = _required_dict(policy, "aggregate", f"{entry.label}.{policy_name}")
            rows.append(
                {
                    "benchmark": f"{entry.label}:{policy_name}",
                    "initial_max_abs": _int(aggregate.get("max_initial_max_abs", 0)),
                    "max_abs_seen": _int(aggregate.get("max_max_abs_seen", 0)),
                    "initial_bitlength": _int(aggregate.get("max_initial_bitlength", 0)),
                    "max_bitlength": _int(aggregate.get("max_max_bitlength", 0)),
                    "growth_numerator": _int(aggregate.get("max_growth_numerator", 0)),
                    "growth_denominator": _int(aggregate.get("max_growth_denominator", 0)),
                    "step_count": _int(aggregate.get("max_step_count", 0)),
                }
            )
    return rows


def _write_plots(
    *,
    out_dir: Path,
    normalized_rows: Sequence[Mapping[str, Any]],
    hnf_coefficient_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    plots_dir = out_dir / "plots"
    plots = {
        "success_rate": "plots/success_rate.png",
        "trace_or_step_length": "plots/trace_or_step_length.png",
        "fill_in_delta": "plots/fill_in_delta.png",
        "hnf_coefficient_growth": "plots/hnf_coefficient_growth.png",
    }
    _save_bar_plot(
        out_dir / plots["success_rate"],
        "Success rate",
        "success rate",
        normalized_rows,
        "success_rate",
    )
    _save_bar_plot(
        out_dir / plots["trace_or_step_length"],
        "Trace or step length",
        "mean length",
        normalized_rows,
        "mean_step_or_trace_length",
    )
    _save_bar_plot(
        out_dir / plots["fill_in_delta"],
        "Fill-in delta",
        "mean fill-in delta",
        normalized_rows,
        "mean_fill_in_delta",
    )
    _save_bar_plot(
        out_dir / plots["hnf_coefficient_growth"],
        "HNF coefficient growth",
        "max bitlength",
        hnf_coefficient_rows,
        "max_bitlength",
    )

    neural_rows = [
        row
        for row in normalized_rows
        if row.get("policy")
        in {"neural", "supervised_greedy", "dagger_greedy", "actor_critic_greedy", "beam"}
    ]
    if neural_rows:
        plots["neural_invalid_actions"] = "plots/neural_invalid_actions.png"
        _save_bar_plot(
            plots_dir / "neural_invalid_actions.png",
            "Neural invalid actions",
            "mean invalid actions",
            neural_rows,
            "mean_invalid_action_count",
        )
    return plots


def _save_bar_plot(
    path: Path,
    title: str,
    ylabel: str,
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
) -> None:
    matplotlib = cast(Any, import_module("matplotlib"))
    matplotlib.use("Agg")
    pyplot = cast(Any, import_module("matplotlib.pyplot"))

    labels = [_plot_label(row) for row in rows]
    values = [_float(row.get(value_key, 0.0)) for row in rows]
    width = max(6.0, 1.3 * max(len(labels), 1))
    fig, ax = pyplot.subplots(figsize=(width, 4.0))
    if values:
        positions = list(range(len(values)))
        ax.bar(positions, values)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path)
    pyplot.close(fig)


def _plot_label(row: Mapping[str, Any]) -> str:
    policy = row.get("policy")
    if policy is None:
        return _string(row.get("benchmark", "unknown"))
    return f"{row.get('benchmark', 'unknown')}\n{policy}"


def _render_markdown(
    *,
    provenance: Mapping[str, Any],
    suite_rows: Sequence[Mapping[str, Any]],
    normalized_rows: Sequence[Mapping[str, Any]],
    hnf_coefficient_rows: Sequence[Mapping[str, Any]],
    plot_paths: Mapping[str, str],
) -> str:
    correctness_rows = [
        {
            "benchmark": row["benchmark"],
            "policy": row["policy"],
            "sample_count": row["sample_count"],
            "success_count": row["success_count"],
            "success_rate": _format_float(row["success_rate"]),
            "status_counts": json.dumps(row["status_counts"], sort_keys=True),
        }
        for row in normalized_rows
    ]
    timing_rows = [
        {
            "benchmark": row["benchmark"],
            "policy": row["policy"],
            "mean_length": _format_float(row["mean_step_or_trace_length"]),
            "mean_wall_time_seconds": _format_float(row["mean_wall_time_seconds"]),
        }
        for row in normalized_rows
    ]
    fill_rows = [
        {
            "benchmark": row["benchmark"],
            "policy": row["policy"],
            "initial_density": _format_float(row["mean_initial_density"]),
            "final_density": _format_float(row["mean_final_density"]),
            "max_density": _format_float(row["mean_max_density"]),
            "fill_in_delta": _format_float(row["mean_fill_in_delta"]),
        }
        for row in normalized_rows
    ]

    plot_lines = "\n".join(
        f"- {name}: [{relative_path}]({relative_path})"
        for name, relative_path in sorted(plot_paths.items())
    )

    return "\n".join(
        [
            "# v0.8 Benchmark Report",
            "",
            "## Provenance",
            "",
            f"- Mode: `{provenance['mode']}`",
            f"- Suite: `{provenance['suite']}`",
            f"- Generated at UTC: `{provenance['generated_at_utc']}`",
            f"- Input JSON paths: `{json.dumps(provenance['input_json_paths'])}`",
            "",
            "## Exactness and Fallback Policy",
            "",
            "Verifier paths use exact integer or modular arithmetic. This report does not "
            "introduce floating point into certificate replay, row-operation replay, or "
            "normal-form predicates. Neural RREF rollout is reported only as its own "
            "policy when supplied; failed neural rollouts remain failures and are not "
            "replaced by deterministic teachers.",
            "",
            "## Suite",
            "",
            _markdown_table(
                (
                    "benchmark",
                    "kind",
                    "source",
                    "family",
                    "shape",
                    "samples",
                    "modulus",
                    "policies",
                ),
                suite_rows,
            ),
            "",
            "## Correctness",
            "",
            _markdown_table(
                (
                    "benchmark",
                    "policy",
                    "sample_count",
                    "success_count",
                    "success_rate",
                    "status_counts",
                ),
                correctness_rows,
            ),
            "",
            "## Timing",
            "",
            _markdown_table(
                ("benchmark", "policy", "mean_length", "mean_wall_time_seconds"),
                timing_rows,
            ),
            "",
            "## Fill-In",
            "",
            _markdown_table(
                (
                    "benchmark",
                    "policy",
                    "initial_density",
                    "final_density",
                    "max_density",
                    "fill_in_delta",
                ),
                fill_rows,
            ),
            "",
            "## HNF Coefficient Growth",
            "",
            _markdown_table(
                (
                    "benchmark",
                    "initial_max_abs",
                    "max_abs_seen",
                    "initial_bitlength",
                    "max_bitlength",
                    "growth_numerator",
                    "growth_denominator",
                    "step_count",
                ),
                hnf_coefficient_rows,
            ),
            "",
            "## Plots",
            "",
            plot_lines,
            "",
            "## Limitations",
            "",
            "- `paper-smoke` is a local reproducibility suite, not a publication-scale run.",
            "- Timing values are local wall-clock measurements with no warmup or repeat protocol.",
            "- SNF benchmark and report coverage is out of scope for v0.8.",
            "- Full matrices and row-operation traces are intentionally omitted from samples.",
            "",
        ]
    )


def _markdown_table(headers: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "_No data._"
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| "
        + " | ".join(_escape_markdown_cell(row.get(header_name, "")) for header_name in headers)
        + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _escape_markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _required_value(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context} is missing required field: {key}")
    return mapping[key]


def _required_dict(mapping: Mapping[str, Any], key: str, context: str) -> JsonDict:
    return _as_dict(_required_value(mapping, key, context), f"{context}.{key}")


def _required_list(mapping: Mapping[str, Any], key: str, context: str) -> list[Any]:
    value = _required_value(mapping, key, context)
    if not isinstance(value, list):
        raise ValueError(f"{context}.{key} must be a list")
    return value


def _as_dict(value: Any, context: str) -> JsonDict:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return cast(JsonDict, value)


def _int(value: Any) -> int:
    return int(value)


def _float(value: Any) -> float:
    if value == "":
        return 0.0
    return float(value)


def _string(value: Any) -> str:
    return str(value)


def _format_float(value: Any) -> str:
    return f"{_float(value):.6g}"
