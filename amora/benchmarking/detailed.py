"""Rank-gated detailed benchmark evidence and Phase 3 comparison artifacts."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from amora.benchmarking.classification import (
    ClassificationManifest,
    ClassificationResult,
)
from amora.benchmarking.materialize import canonical_json
from amora.benchmarking.schema import BenchmarkCase, CaseSetManifest, DetailedCaseResult


SIZE_RANKS = ("small", "medium", "large")

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
    return canonical_json(value)


def _sha256_payload(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_new_text(path: str | Path, content: str) -> Path:
    """Write a benchmark artifact exactly once."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite immutable benchmark artifact: {destination}"
        ) from exc
    return destination


def _previous_rank(size_rank: str) -> str | None:
    try:
        index = SIZE_RANKS.index(size_rank)
    except ValueError as exc:
        raise ValueError(f"unsupported size rank: {size_rank}") from exc
    return SIZE_RANKS[index - 1] if index else None


@dataclass(frozen=True)
class DetailReviewMarker:
    """An explicit acceptance of a completed detailed rank comparison."""

    case_set_digest: str
    classification_digest: str
    reviewed_rank: str
    accepted_run_ids: tuple[str, ...]
    accepted_comparison_digests: tuple[str, ...]
    rank_case_count: int
    known_failures: tuple[str, ...]
    semantic_decisions: tuple[str, ...]
    reviewer: str | None
    reviewed_at: str
    review_marker_digest: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "benchmark_detail_review",
            "case_set_digest": self.case_set_digest,
            "classification_digest": self.classification_digest,
            "reviewed_rank": self.reviewed_rank,
            "accepted_run_ids": list(self.accepted_run_ids),
            "accepted_comparison_digests": list(self.accepted_comparison_digests),
            "rank_case_count": self.rank_case_count,
            "known_failures": list(self.known_failures),
            "semantic_decisions": list(self.semantic_decisions),
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "review_marker_digest": self.review_marker_digest,
        }


def _review_marker_payload(marker: DetailReviewMarker) -> dict[str, Any]:
    payload = marker.to_dict()
    payload.pop("review_marker_digest")
    return payload


def _validate_review_marker_digest(marker: DetailReviewMarker) -> None:
    actual = _sha256_payload(_review_marker_payload(marker))
    if actual != marker.review_marker_digest:
        raise ValueError("review marker digest does not match its contents")


def load_review_marker(path: str | Path) -> DetailReviewMarker:
    """Load and validate a persisted explicit rank-review marker."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    marker = DetailReviewMarker(
        case_set_digest=data["case_set_digest"],
        classification_digest=data["classification_digest"],
        reviewed_rank=data["reviewed_rank"],
        accepted_run_ids=tuple(data.get("accepted_run_ids") or ()),
        accepted_comparison_digests=tuple(data.get("accepted_comparison_digests") or ()),
        rank_case_count=int(data["rank_case_count"]),
        known_failures=tuple(data.get("known_failures") or ()),
        semantic_decisions=tuple(data.get("semantic_decisions") or ()),
        reviewer=data.get("reviewer"),
        reviewed_at=data["reviewed_at"],
        review_marker_digest=data["review_marker_digest"],
        schema_version=int(data.get("schema_version", 1)),
    )
    _validate_review_marker_digest(marker)
    return marker


def write_review_marker(marker: DetailReviewMarker, path: str | Path) -> Path:
    """Persist one immutable review decision."""

    _validate_review_marker_digest(marker)
    return _write_new_text(
        path,
        json.dumps(marker.to_dict(), indent=2, sort_keys=True) + "\n",
    )


def review_gate_for_rank(
    manifest: CaseSetManifest,
    classification: ClassificationManifest,
    *,
    size_rank: str,
    review_marker: DetailReviewMarker | None,
) -> dict[str, Any]:
    """Validate the predecessor review required before a detailed rank can run."""

    previous_rank = _previous_rank(size_rank)
    if previous_rank is None:
        if review_marker is not None:
            raise ValueError("small rank does not accept a predecessor review marker")
        return {
            "required": False,
            "previous_rank": None,
            "review_marker": None,
        }

    if review_marker is None:
        raise ValueError(
            f"detailed {size_rank} rank requires an accepted {previous_rank} review marker"
        )
    _validate_review_marker_digest(review_marker)
    if review_marker.case_set_digest != manifest.case_set_digest:
        raise ValueError("review marker case-set digest does not match the manifest")
    if review_marker.classification_digest != classification.classification_digest:
        raise ValueError("review marker classification digest does not match the overlay")
    if review_marker.reviewed_rank != previous_rank:
        raise ValueError(
            f"detailed {size_rank} rank requires a {previous_rank} review marker"
        )
    expected_rank_count = len(
        select_ranked_cases(manifest, classification, size_rank=previous_rank)
    )
    if review_marker.rank_case_count != expected_rank_count:
        raise ValueError("review marker rank population does not match the overlay")
    if not review_marker.accepted_run_ids:
        raise ValueError("review marker does not accept any detailed runs")
    if not review_marker.accepted_comparison_digests:
        raise ValueError("review marker does not accept any detailed comparisons")
    return {
        "required": True,
        "previous_rank": previous_rank,
        "review_marker": {
            "review_marker_digest": review_marker.review_marker_digest,
            "reviewed_rank": review_marker.reviewed_rank,
            "accepted_run_ids": list(review_marker.accepted_run_ids),
            "accepted_comparison_digests": list(
                review_marker.accepted_comparison_digests
            ),
        },
    }


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

    if size_rank not in SIZE_RANKS:
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

    lines = [
        json.dumps(result.to_dict(), sort_keys=True)
        for result in results
    ]
    return _write_new_text(path, "\n".join(lines) + ("\n" if lines else ""))


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
    run_id: str | None = None,
    review_gate: dict[str, Any] | None = None,
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
        "phase": "rank_detailed_evidence",
        "case_set_digest": manifest.case_set_digest,
        "classification_digest": classification.classification_digest,
        "size_rank": size_rank,
        "run_id": run_id,
        "review_gate": review_gate or {
            "required": False,
            "previous_rank": None,
            "review_marker": None,
        },
        "od2_scalar_accuracy": "deferred",
        "coverage": coverage,
        "cases": case_rows,
    }
    payload["comparison_digest"] = _sha256_payload(payload)
    return payload


def _comparison_payload(comparison: dict[str, Any]) -> dict[str, Any]:
    payload = dict(comparison)
    payload.pop("comparison_digest", None)
    return payload


def validate_detailed_comparison(comparison: dict[str, Any]) -> None:
    """Validate the content digest of a detailed comparison artifact."""

    expected = comparison.get("comparison_digest")
    if not isinstance(expected, str) or not expected:
        raise ValueError("detailed comparison has no comparison digest")
    actual = _sha256_payload(_comparison_payload(comparison))
    if actual != expected:
        raise ValueError("detailed comparison digest does not match its contents")


def load_detailed_comparison(path: str | Path) -> dict[str, Any]:
    """Load and validate an immutable detailed comparison artifact."""

    comparison = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_detailed_comparison(comparison)
    return comparison


def build_review_marker(
    *,
    manifest: CaseSetManifest,
    classification: ClassificationManifest,
    comparisons: Iterable[dict[str, Any]],
    known_failures: Iterable[str] = (),
    semantic_decisions: Iterable[str] = (),
    reviewer: str | None = None,
    reviewed_at: str,
) -> DetailReviewMarker:
    """Accept complete matching evidence artifacts for one finished rank."""

    evidence = list(comparisons)
    if not evidence:
        raise ValueError("a review marker requires at least one comparison artifact")
    reviewed_rank = evidence[0].get("size_rank")
    if reviewed_rank not in SIZE_RANKS:
        raise ValueError(f"comparison has unsupported size rank: {reviewed_rank}")
    expected_rank_count = len(
        select_ranked_cases(manifest, classification, size_rank=reviewed_rank)
    )
    run_ids: set[str] = set()
    comparison_digests: set[str] = set()
    for comparison in evidence:
        validate_detailed_comparison(comparison)
        if comparison.get("case_set_digest") != manifest.case_set_digest:
            raise ValueError("comparison case-set digest does not match the manifest")
        if comparison.get("classification_digest") != classification.classification_digest:
            raise ValueError("comparison classification digest does not match the overlay")
        if comparison.get("size_rank") != reviewed_rank:
            raise ValueError("review accepts comparisons from exactly one size rank")
        coverage = comparison.get("coverage") or {}
        if coverage.get("rank_case_count") != expected_rank_count:
            raise ValueError("comparison rank population does not match the classification overlay")
        if coverage.get("selected_case_count") != expected_rank_count or not coverage.get(
            "selection_complete"
        ):
            raise ValueError(
                "review requires a complete detailed comparison for the persisted rank"
            )
        run_id = comparison.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("comparison does not identify its immutable detail run")
        if run_id in run_ids:
            raise ValueError(f"review contains duplicate detail run ID: {run_id}")
        digest = comparison["comparison_digest"]
        if digest in comparison_digests:
            raise ValueError(f"review contains duplicate comparison digest: {digest}")
        run_ids.add(run_id)
        comparison_digests.add(digest)
    template = DetailReviewMarker(
        case_set_digest=manifest.case_set_digest,
        classification_digest=classification.classification_digest,
        reviewed_rank=reviewed_rank,
        accepted_run_ids=tuple(sorted(run_ids)),
        accepted_comparison_digests=tuple(sorted(comparison_digests)),
        rank_case_count=expected_rank_count,
        known_failures=tuple(known_failures),
        semantic_decisions=tuple(semantic_decisions),
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        review_marker_digest="",
    )
    return DetailReviewMarker(
        **{
            **template.__dict__,
            "review_marker_digest": _sha256_payload(_review_marker_payload(template)),
        }
    )


def build_detail_run_manifest(
    *,
    manifest: CaseSetManifest,
    classification: ClassificationManifest,
    comparison: dict[str, Any],
    hardware_results: Iterable[DetailedCaseResult],
    simulation_results: Iterable[DetailedCaseResult],
    hardware_path: str | Path,
    simulation_path: str | Path,
    comparison_path: str | Path,
    review_gate: dict[str, Any],
    started_at: str,
    completed_at: str,
    run_options: dict[str, Any],
) -> dict[str, Any]:
    """Build the compact immutable manifest for one detailed rank invocation."""

    validate_detailed_comparison(comparison)
    hardware = list(hardware_results)
    simulation = list(simulation_results)
    payload = {
        "schema_version": 1,
        "kind": "benchmark_detail_run",
        "run_id": comparison["run_id"],
        "case_set_digest": manifest.case_set_digest,
        "benchmark_id": manifest.benchmark_id,
        "benchmark_revision": manifest.benchmark_revision,
        "classification_digest": classification.classification_digest,
        "size_rank": comparison["size_rank"],
        "rank_case_count": comparison["coverage"]["rank_case_count"],
        "selected_case_keys": [
            row["case_key"] for row in comparison["cases"]
        ],
        "review_gate": review_gate,
        "comparison_digest": comparison["comparison_digest"],
        "artifacts": {
            "hardware_results": str(hardware_path),
            "gcom_results": str(simulation_path),
            "comparison": str(comparison_path),
        },
        "hardware_status_counts": dict(Counter(result.status for result in hardware)),
        "gcom_status_counts": dict(Counter(result.status for result in simulation)),
        "run_options": run_options,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    payload["run_digest"] = _sha256_payload(payload)
    return payload


def write_detail_run_manifest(run_manifest: dict[str, Any], path: str | Path) -> Path:
    """Persist one immutable detailed-run manifest."""

    return _write_new_text(
        path,
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
    )


def render_detailed_markdown(comparison: dict[str, Any]) -> str:
    """Render a compact rank-scoped detailed evidence summary."""

    coverage = comparison["coverage"]
    lines = [
        f"# PPP {comparison['size_rank']} Rank Detailed Evidence",
        "",
        f"- case-set digest: `{comparison['case_set_digest']}`",
        f"- classification digest: `{comparison['classification_digest']}`",
        f"- detail run ID: `{comparison.get('run_id') or 'not recorded'}`",
        f"- scalar accuracy decision: `{comparison['od2_scalar_accuracy']}`",
        f"- rank cases: {coverage['rank_case_count']}",
        f"- selected cases: {coverage['selected_case_count']}",
        f"- rank selection complete: {coverage['selection_complete']}",
        f"- detailed hardware + GCoM evidence: {coverage['both_detailed']}",
        "",
        "## Execution Gate",
        "",
    ]
    gate = comparison.get("review_gate") or {}
    if gate.get("required"):
        marker = gate.get("review_marker") or {}
        lines.extend(
            [
                f"- predecessor rank: `{gate['previous_rank']}`",
                f"- accepted review marker: `{marker.get('review_marker_digest')}`",
                f"- accepted runs: {', '.join(marker.get('accepted_run_ids') or ())}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- no predecessor review is required for the small rank.",
                "",
            ]
        )
    lines.extend(
        [
        "## Case Coverage",
        "",
        "| case | kernel | hardware | gcom | comparison |",
        "|---|---|---|---|---|",
        ]
    )
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
    """Write immutable detailed comparison JSON and Markdown artifacts."""

    destination = Path(out_dir)
    json_path = destination / "comparison.json"
    markdown_path = destination / "SUMMARY.md"
    if json_path.exists() or markdown_path.exists():
        existing = json_path if json_path.exists() else markdown_path
        raise FileExistsError(
            f"refusing to overwrite immutable benchmark artifact: {existing}"
        )
    _write_new_text(json_path, json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    _write_new_text(markdown_path, render_detailed_markdown(comparison) + "\n")
    return {"json": json_path, "markdown": markdown_path}
