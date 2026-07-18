"""Rank-gated detailed benchmark evidence and Phase 3 comparison artifacts."""

from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from amora.benchmarking.classification import (
    ClassificationManifest,
    ClassificationResult,
)
from amora.benchmarking.schema import BenchmarkCase, CaseSetManifest, DetailedCaseResult


STALL_REASONS = (
    "selected",
    "not_selected",
    "dispatch_stall",
    "warpgroup_arrive",
    "long_scoreboard",
    "short_scoreboard",
    "barrier",
    "wait",
    "mio_throttle",
    "math_pipe_throttle",
    "mma",
    "no_instructions",
    "imc_miss",
    "sleeping",
    "branch_resolving",
    "membar",
    "drain",
    "lg_throttle",
    "tex_throttle",
    "misc",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_classification_manifest(path: str | Path) -> ClassificationManifest:
    """Load a serialized Phase 2 classification overlay."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results = tuple(
        ClassificationResult(
            case_key=row["case_key"],
            status=row["status"],
            total_instructions=row.get("total_instructions"),
            kernel_name=row.get("kernel_name"),
            metrics=row.get("metrics") or {},
            resolved_metrics=row.get("resolved_metrics") or {},
            provenance=row.get("provenance") or {},
            reason=row.get("reason"),
        )
        for row in data.get("results") or ()
    )
    return ClassificationManifest(
        case_set_digest=data["case_set_digest"],
        target=data.get("target") or {},
        recipe=data["recipe"],
        instruction_logical=data["instruction_logical"],
        results=results,
        case_count_expected=int(data["case_count_expected"]),
        case_count_attempted=int(data["case_count_attempted"]),
        case_coverage_complete=bool(data["case_coverage_complete"]),
        rank_assignments=data.get("rank_assignments") or {},
        rank_boundaries=data.get("rank_boundaries") or {},
        classification_digest=data["classification_digest"],
        schema_version=int(data.get("schema_version", 1)),
    )


def select_ranked_cases(
    manifest: CaseSetManifest,
    classification: ClassificationManifest,
    *,
    size_rank: str,
) -> list[BenchmarkCase]:
    """Return cases in a persisted rank, rejecting partial/mismatched overlays."""

    if size_rank not in {"small", "medium", "large"}:
        raise ValueError(f"unsupported size rank: {size_rank}")
    if manifest.case_set_digest != classification.case_set_digest:
        raise ValueError("classification case-set digest does not match the manifest")
    if not classification.case_coverage_complete:
        raise ValueError("detailed comparison requires a complete classification overlay")
    by_key = {case.case_key: case for case in manifest.cases}
    selected = [
        by_key[case_key]
        for case_key, assignment in classification.rank_assignments.items()
        if assignment.get("size_rank") == size_rank and case_key in by_key
    ]
    selected.sort(key=lambda case: int(classification.rank_assignments[case.case_key]["size_rank_ordinal"]))
    if not selected:
        raise ValueError(f"classification overlay has no {size_rank} cases")
    return selected


def write_case_results(results: Iterable[DetailedCaseResult], path: str | Path) -> Path:
    """Persist one immutable detail evidence stream as JSONL."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(result.to_dict(), sort_keys=True)
        for result in results
    ]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return destination


def _counter_rows(
    hardware: DetailedCaseResult | None,
    simulation: DetailedCaseResult | None,
) -> list[dict[str, Any]]:
    hw = hardware.logical_metrics if hardware else {}
    sim = simulation.logical_metrics if simulation else {}
    rows = []
    for logical in sorted(set(hw) | set(sim)):
        if logical.startswith("stall_"):
            continue
        sim_value = sim.get(logical)
        if isinstance(sim_value, dict):
            sim_value = sim_value.get("value")
        rows.append(
            {
                "logical": logical,
                "hardware_ncu": hw.get(logical),
                "gcom": sim_value,
                "comparison_status": "evidence_only_od2_deferred",
            }
        )
    return rows


def _stall_rows(
    hardware: DetailedCaseResult | None,
    simulation: DetailedCaseResult | None,
) -> list[dict[str, Any]]:
    hw_reasons = ((hardware.stall_histogram or {}).get("reasons") or {}) if hardware else {}
    sim_reasons = ((simulation.stall_histogram or {}).get("reasons") or {}) if simulation else {}
    rows = []
    for reason in STALL_REASONS:
        hw_value = hw_reasons.get(reason)
        sim_entry = sim_reasons.get(reason)
        sim_pct = sim_entry.get("pct") if isinstance(sim_entry, dict) else None
        rows.append(
            {
                "reason": reason,
                "hardware_ncu": hw_value,
                "hardware_metric": hardware.resolved_metrics.get(f"stall_{reason}") if hardware else None,
                "gcom_pct": sim_pct,
                "gcom_count": sim_entry.get("count") if isinstance(sim_entry, dict) else None,
                "comparison_status": (
                    "evidence_available"
                    if hw_value is not None and sim_pct is not None
                    else "missing_counter"
                ),
            }
        )
    return rows


def build_detailed_comparison(
    *,
    manifest: CaseSetManifest,
    classification: ClassificationManifest,
    size_rank: str,
    hardware_results: Iterable[DetailedCaseResult],
    simulation_results: Iterable[DetailedCaseResult],
    rank_case_count: int | None = None,
    selected_cases: Iterable[BenchmarkCase] | None = None,
) -> dict[str, Any]:
    """Build Phase 3 evidence comparison without selecting an OD2 error metric."""

    hardware_by_key = {result.case_key: result for result in hardware_results}
    simulation_by_key = {result.case_key: result for result in simulation_results}
    ranked_cases = select_ranked_cases(manifest, classification, size_rank=size_rank)
    ranked_by_key = {case.case_key: case for case in ranked_cases}
    if selected_cases is None:
        cases = ranked_cases
    else:
        cases = list(selected_cases)
        unknown = [case.case_key for case in cases if case.case_key not in ranked_by_key]
        if unknown:
            raise ValueError(f"selected cases are not in the persisted {size_rank} rank: {unknown}")
    case_rows = []
    for case in cases:
        hardware = hardware_by_key.get(case.case_key)
        simulation = simulation_by_key.get(case.case_key)
        hw_status = hardware.status if hardware else "missing_hw"
        sim_status = simulation.status if simulation else "missing_sim"
        semantic = case.execution_contract.get("measurement_semantics")
        if semantic == "fused_component_aggregate":
            comparison_status = "semantic_mismatch_component_aggregation"
        elif hw_status == "measured" and sim_status == "simulated":
            comparison_status = "evidence_collected_od2_deferred"
        else:
            comparison_status = "incomplete_evidence"
        case_rows.append(
            {
                "case_key": case.case_key,
                "kernel_id": case.kernel_id,
                "shape": dict(case.shape),
                "shape_class": case.shape_class,
                "size_rank": size_rank,
                "measurement_semantics": semantic,
                "hardware_status": hw_status,
                "gcom_status": sim_status,
                "comparison_status": comparison_status,
                "scalar_error_status": "deferred_od2",
                "hardware_measurement": dict(hardware.measurement) if hardware else {},
                "gcom_measurement": dict(simulation.measurement) if simulation else {},
                "counter_comparison": _counter_rows(hardware, simulation),
                "stall_reason_comparison": _stall_rows(hardware, simulation),
            }
        )
    coverage = {
        "rank_case_count": rank_case_count if rank_case_count is not None else len(ranked_cases),
        "selected_case_count": len(cases),
        "selection_complete": (
            len(cases) == (rank_case_count if rank_case_count is not None else len(ranked_cases))
        ),
        "hardware_status": dict(Counter(row["hardware_status"] for row in case_rows)),
        "gcom_status": dict(Counter(row["gcom_status"] for row in case_rows)),
        "both_detailed": sum(
            row["comparison_status"] == "evidence_collected_od2_deferred"
            for row in case_rows
        ),
    }
    payload = {
        "schema_version": 1,
        "phase": "small_rank_detailed_evidence",
        "case_set_digest": manifest.case_set_digest,
        "classification_digest": classification.classification_digest,
        "size_rank": size_rank,
        "od2_scalar_accuracy": "deferred",
        "coverage": coverage,
        "cases": case_rows,
    }
    payload["comparison_digest"] = sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def render_detailed_markdown(comparison: dict[str, Any]) -> str:
    """Render a compact Phase 3 evidence summary."""

    coverage = comparison["coverage"]
    lines = [
        f"# PPP {comparison['size_rank']} Rank Detailed Evidence",
        "",
        f"- case-set digest: `{comparison['case_set_digest']}`",
        f"- classification digest: `{comparison['classification_digest']}`",
        f"- scalar accuracy decision: `{comparison['od2_scalar_accuracy']}`",
        f"- rank cases: {coverage['rank_case_count']}",
        f"- selected cases: {coverage['selected_case_count']}",
        f"- rank selection complete: {coverage['selection_complete']}",
        f"- detailed hardware + GCoM evidence: {coverage['both_detailed']}",
        "",
        "## Case Coverage",
        "",
        "| case | kernel | hardware | gcom | comparison |",
        "|---|---|---|---|---|",
    ]
    for row in comparison["cases"]:
        lines.append(
            f"| {row['case_key']} | {row['kernel_id']} | {row['hardware_status']} | "
            f"{row['gcom_status']} | {row['comparison_status']} |"
        )
    lines.extend(
        [
            "",
            "## Detailed Counters",
            "",
            "Counter and stall values are preserved as evidence only. OD2 has not yet selected "
            "a canonical scalar accuracy metric for this benchmark suite.",
            "",
        ]
    )
    return "\n".join(lines)


def write_detailed_comparison(comparison: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """Write immutable Phase 3 comparison JSON and Markdown artifacts."""

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "comparison.json"
    markdown_path = destination / "SUMMARY.md"
    json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_detailed_markdown(comparison) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
