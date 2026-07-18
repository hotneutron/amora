from __future__ import annotations

import pytest

from amora.benchmarking.classification import ClassificationResult, build_classification_manifest
from amora.benchmarking.detailed import build_detailed_comparison, select_ranked_cases
from amora.benchmarking.materialize import materialize_benchmark
from amora.benchmarking.schema import DetailedCaseResult


TARGET = {
    "vendor": "nvidia",
    "family": "hopper",
    "hardware_sku": "h100-80g",
    "arch_profile": "sm_90_h100",
}


def _manifest():
    return materialize_benchmark(
        "ppp_canonical",
        target=TARGET,
        case_count=9,
        seed=7,
    )


def _classification(manifest, *, complete: bool = True):
    cases = manifest.cases if complete else manifest.cases[:2]
    results = [
        ClassificationResult(
            case_key=case.case_key,
            status="classified",
            total_instructions=float(index + 1),
        )
        for index, case in enumerate(cases)
    ]
    return build_classification_manifest(
        case_set_digest=manifest.case_set_digest,
        target=manifest.target,
        results=results,
        expected_case_keys=[case.case_key for case in manifest.cases],
    )


def test_select_ranked_cases_requires_complete_overlay():
    manifest = _manifest()
    classification = _classification(manifest, complete=False)

    with pytest.raises(ValueError, match="complete classification overlay"):
        select_ranked_cases(manifest, classification, size_rank="small")


def test_detailed_comparison_keeps_od2_scalar_error_deferred():
    manifest = _manifest()
    classification = _classification(manifest)
    selected = select_ranked_cases(manifest, classification, size_rank="small")
    hardware = [
        DetailedCaseResult(
            case_key=case.case_key,
            kernel_id=case.kernel_id,
            size_rank="small",
            backend="nvidia_cuda",
            status="measured",
            measurement={"elapsed_cycles": 100.0},
            logical_metrics={"inst_executed": 1000.0, "stall_wait": 12.0},
            stall_histogram={"reasons": {"wait": 12.0}},
        )
        for case in selected
    ]
    simulation = [
        DetailedCaseResult(
            case_key=case.case_key,
            kernel_id=case.kernel_id,
            size_rank="small",
            backend="gcom_cuda",
            status="simulated",
            measurement={"gpu_sim_cycle": 95.0},
            logical_metrics={"inst_executed": {"value": 900.0}},
            stall_histogram={"reasons": {"wait": {"count": 8.0, "pct": 8.0}}},
        )
        for case in selected
    ]

    comparison = build_detailed_comparison(
        manifest=manifest,
        classification=classification,
        size_rank="small",
        hardware_results=hardware,
        simulation_results=simulation,
        rank_case_count=len(selected),
    )

    assert comparison["od2_scalar_accuracy"] == "deferred"
    assert comparison["coverage"]["selection_complete"] is True
    assert comparison["coverage"]["both_detailed"] == len(selected)
    first = next(
        row
        for row in comparison["cases"]
        if row["measurement_semantics"] != "fused_component_aggregate"
    )
    assert first["scalar_error_status"] == "deferred_od2"
    assert first["counter_comparison"][0]["comparison_status"] == "evidence_only_od2_deferred"
    wait = next(row for row in first["stall_reason_comparison"] if row["reason"] == "wait")
    assert wait["comparison_status"] == "evidence_available"


def test_detailed_comparison_marks_bounded_rank_selection_incomplete():
    manifest = _manifest()
    classification = _classification(manifest)
    selected = select_ranked_cases(manifest, classification, size_rank="small")

    comparison = build_detailed_comparison(
        manifest=manifest,
        classification=classification,
        size_rank="small",
        hardware_results=[],
        simulation_results=[],
        rank_case_count=len(selected),
        selected_cases=selected[:1],
    )

    assert comparison["coverage"]["rank_case_count"] == len(selected)
    assert comparison["coverage"]["selected_case_count"] == 1
    assert comparison["coverage"]["selection_complete"] is False
