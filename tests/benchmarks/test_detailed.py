from __future__ import annotations

import pytest

from amora.benchmarking.classification import ClassificationResult, build_classification_manifest
from amora.benchmarking.detailed import (
    build_detailed_comparison,
    build_detail_run_manifest,
    build_review_marker,
    review_gate_for_rank,
    select_ranked_cases,
    write_case_results,
    write_detailed_comparison,
)
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


def _complete_comparison(manifest, classification, size_rank):
    selected = select_ranked_cases(manifest, classification, size_rank=size_rank)
    return build_detailed_comparison(
        manifest=manifest,
        classification=classification,
        size_rank=size_rank,
        hardware_results=[],
        simulation_results=[],
        run_id=f"{size_rank}-run",
    )


def test_review_marker_gates_medium_and_large_rank_progression():
    manifest = _manifest()
    classification = _classification(manifest)
    small = _complete_comparison(manifest, classification, "small")
    small_review = build_review_marker(
        manifest=manifest,
        classification=classification,
        comparisons=[small],
        known_failures=["missing GCoM stats are retained as evidence"],
        semantic_decisions=["OD2 scalar accuracy remains deferred"],
        reviewer="test",
        reviewed_at="2026-07-18T00:00:00+00:00",
    )

    with pytest.raises(ValueError, match="requires a medium review marker"):
        review_gate_for_rank(
            manifest,
            classification,
            size_rank="large",
            review_marker=small_review,
        )

    medium_gate = review_gate_for_rank(
        manifest,
        classification,
        size_rank="medium",
        review_marker=small_review,
    )
    assert medium_gate["required"] is True
    assert medium_gate["previous_rank"] == "small"
    assert medium_gate["review_marker"]["accepted_run_ids"] == ["small-run"]

    medium = _complete_comparison(manifest, classification, "medium")
    medium_review = build_review_marker(
        manifest=manifest,
        classification=classification,
        comparisons=[medium],
        reviewer="test",
        reviewed_at="2026-07-18T00:01:00+00:00",
    )
    large_gate = review_gate_for_rank(
        manifest,
        classification,
        size_rank="large",
        review_marker=medium_review,
    )
    assert large_gate["previous_rank"] == "medium"
    assert large_gate["review_marker"]["accepted_run_ids"] == ["medium-run"]


def test_review_rejects_bounded_detail_comparison():
    manifest = _manifest()
    classification = _classification(manifest)
    selected = select_ranked_cases(manifest, classification, size_rank="small")
    comparison = build_detailed_comparison(
        manifest=manifest,
        classification=classification,
        size_rank="small",
        hardware_results=[],
        simulation_results=[],
        selected_cases=selected[:1],
        run_id="partial-small",
    )

    with pytest.raises(ValueError, match="complete detailed comparison"):
        build_review_marker(
            manifest=manifest,
            classification=classification,
            comparisons=[comparison],
            reviewed_at="2026-07-18T00:00:00+00:00",
        )


def test_detail_result_artifacts_are_write_once(tmp_path):
    result = DetailedCaseResult(
        case_key="case",
        kernel_id="kernel",
        size_rank="small",
        backend="nvidia_cuda",
        status="measured",
    )
    path = tmp_path / "hardware.jsonl"

    assert write_case_results([result], path) == path
    with pytest.raises(FileExistsError, match="immutable benchmark artifact"):
        write_case_results([result], path)


def test_detail_comparison_writer_preflights_all_immutable_artifacts(tmp_path):
    manifest = _manifest()
    classification = _classification(manifest)
    comparison = _complete_comparison(manifest, classification, "small")
    destination = tmp_path / "comparison"
    destination.mkdir()
    (destination / "SUMMARY.md").write_text("existing\n")

    with pytest.raises(FileExistsError, match="SUMMARY.md"):
        write_detailed_comparison(comparison, destination)
    assert not (destination / "comparison.json").exists()


def test_detail_run_manifest_records_review_gate_and_artifacts(tmp_path):
    manifest = _manifest()
    classification = _classification(manifest)
    comparison = _complete_comparison(manifest, classification, "small")
    run_manifest = build_detail_run_manifest(
        manifest=manifest,
        classification=classification,
        comparison=comparison,
        hardware_results=[],
        simulation_results=[],
        hardware_path=tmp_path / "hardware.jsonl",
        simulation_path=tmp_path / "gcom.jsonl",
        comparison_path=tmp_path / "comparison.json",
        review_gate=review_gate_for_rank(
            manifest,
            classification,
            size_rank="small",
            review_marker=None,
        ),
        started_at="2026-07-18T00:00:00+00:00",
        completed_at="2026-07-18T00:01:00+00:00",
        run_options={"limit": None},
    )

    assert run_manifest["run_id"] == "small-run"
    assert run_manifest["review_gate"]["required"] is False
    assert run_manifest["selected_case_keys"] == [
        case.case_key
        for case in select_ranked_cases(manifest, classification, size_rank="small")
    ]
    assert run_manifest["run_digest"]
