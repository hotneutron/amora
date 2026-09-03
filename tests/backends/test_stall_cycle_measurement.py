"""Tests for all-corner replay warp-stall-cycle measurement."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.cuda_event_run import (
    CudaEventProcessSample,
    CudaEventTimingResult,
)
from amora.backends.nvidia.ncu_run import NcuPcSamplingResult, NcuResult, PcStallSample
from amora.backends.nvidia.stall_cycle_measurement import (
    AllCornerCampaign,
    AllCornerLaunchSpec,
    RAW_STALL_CYCLE_UNIT,
    _operation_timing_identity_reasons,
    collect_stall_cycle_repeats,
    execute_all_corner_campaign,
    load_all_corner_campaign,
    reduce_stall_cycle_repeats,
    reduce_stall_cycle_row,
    validate_all_corner_manifest,
    _operation_totals,
)
from amora.backends.nvidia.stall_metrics import resolve_stall_cycle_metrics


REASON_METRICS = {
    "selected": "smsp__warp_issue_stalled_selected_per_warp_active.ratio",
    "not_selected": "smsp__warp_issue_stalled_not_selected_per_warp_active.ratio",
    "wait": "smsp__warp_issue_stalled_wait_per_warp_active.ratio",
    "barrier": "smsp__warp_issue_stalled_barrier_per_warp_active.ratio",
}
CYCLE_METRICS = {
    "active_warp_cycles": "smsp__warps_active.sum",
    "active_smsp_cycles": "smsp__cycles_active.sum",
    "eligible_warp_cycles": "smsp__warps_eligible.sum",
}
IDENTITY = {
    "kernel_name": "kernel_a",
    "ttgir_sha256": "ttgir",
    "ptx_sha256": "ptx",
    "sass_sha256": "sass",
    "cubin_sha256": "cubin",
}
METADATA = {
    "registers_per_thread": 64,
    "shared_memory_bytes": 32768,
    "total_warps": 8,
    "spill_count": 0,
}
AXES = {
    "topology": "joint",
    "cache_protocol": "warm_reuse",
    "clock_policy": "none",
}
CONTEXT = {
    "operation_id": "operation-a",
    "contract": "contract-a",
    "corner_id": "corner-min",
    "launch_ordinal": 0,
    "kernel_name": "kernel_a",
    "grid": "1x1x1",
    "workgroup": "128x1x1",
    "compile_time_axes": {"topology": "joint"},
    "runtime_axes": {
        "cache_protocol": "warm_reuse",
        "clock_policy": "none",
    },
    "ordered_launches": [
        {
            "launch_ordinal": 0,
            "kernel_name": "kernel_a",
            "subject_identity": dict(IDENTITY),
        }
    ],
}
TOOLS = {"target": "1", "ncu": "2026.2"}


def _selection(*, pct=False, missing=()):
    suffix = "pct" if pct else "ratio"
    supported = {
        *(metric.rsplit(".", 1)[0] for metric in CYCLE_METRICS.values()),
        *(
            f"smsp__warp_issue_stalled_{reason}_per_warp_active"
            for reason in REASON_METRICS
            if reason not in missing
        ),
    }
    selection = resolve_stall_cycle_metrics(frozenset(supported))
    if pct:
        selection = replace(
            selection,
            input_unit=suffix,
            reason_to_metric={
                reason: metric.rsplit(".", 1)[0] + ".pct"
                for reason, metric in selection.reason_to_metric.items()
            },
        )
    return selection


def _evidence(*, axes=AXES, context=CONTEXT, cubin="cubin"):
    return {
        "subject_identity": {**IDENTITY, "cubin_sha256": cubin},
        "device": {"uuid": "GPU-test", "name": "H100"},
        "subject_metadata": dict(METADATA),
        "measurement_axes": dict(axes),
        "measurement_context": json.loads(json.dumps(context)),
        "tool_versions": dict(TOOLS),
    }


def _row(total_scale=1.0, *, pct=False):
    values = {
        "selected": 0.10 * total_scale,
        "not_selected": 0.20 * total_scale,
        "wait": 0.40 * total_scale,
        "barrier": 0.30 * total_scale,
    }
    row = {
        "Kernel Name": "kernel_a",
        "Grid Size": "(1, 1, 1)",
        "Block Size": "(128, 1, 1)",
        "smsp__warps_active.sum": "1000",
        "smsp__cycles_active.sum": "250",
        "smsp__warps_eligible.sum": "200",
        "profiler__replayer_passes": "3",
    }
    for reason, value in values.items():
        metric = REASON_METRICS[reason]
        if pct:
            metric = metric.rsplit(".", 1)[0] + ".pct"
            value *= 100.0
        row[metric] = str(value)
    return row


def _ncu_result(
    scales=(1.4, 1.0, 1.2),
    *,
    repeat=0,
    pct=False,
    cubin="cubin",
    cache_control=None,
    clock_control=None,
):
    rows = [_row(scale, pct=pct) for scale in scales]
    return NcuResult(
        metrics={
            key: float(value)
            for key, value in rows[-1].items()
            if key not in {
                "profiler__replayer_passes",
                "Kernel Name",
                "Grid Size",
                "Block Size",
            }
        },
        raw_rows=rows,
        target_evidence=_evidence(cubin=cubin),
        report_path=None,
        report_sha256=f"report-{repeat}",
        command=("ncu",),
        cache_control=cache_control,
        clock_control=clock_control,
    )


def _timing(context=CONTEXT):
    process = CudaEventProcessSample(
        process_index=0,
        samples_us=(10.0,),
        warmup_launches=1,
        launch_batch_size=2,
        cache_protocol="warm_reuse",
        clock_policy="none",
        device={"uuid": "GPU-test", "name": "H100"},
        subject_identity=dict(IDENTITY),
        subject_metadata=dict(METADATA),
        measurement_axes=dict(AXES),
        tool_versions=dict(TOOLS),
        command=("target",),
        cwd=None,
        environment_overrides={},
        stdout="",
        stderr="",
        returncode=0,
        started_at="start",
        completed_at="end",
        measurement_context=json.loads(json.dumps(context)),
    )
    return CudaEventTimingResult(
        target_command=("target",),
        process_samples=(process,),
        median_us=10.0,
        p05_us=10.0,
        p95_us=10.0,
        process_cv=0.0,
        launch_batch_size=2,
        cache_protocol="warm_reuse",
        clock_policy="none",
        device=dict(process.device),
        subject_identity=dict(IDENTITY),
        subject_metadata=dict(METADATA),
        measurement_axes=dict(AXES),
        tool_versions=dict(TOOLS),
        provenance={},
        measurement_context=json.loads(json.dumps(context)),
    )


def _spec(*, point_id="point-a", context=CONTEXT, source_counters=False):
    return AllCornerLaunchSpec(
        point_id=point_id,
        target=("target",),
        kernel_name=context["kernel_name"],
        measurement_axes=dict(AXES),
        measurement_context=json.loads(json.dumps(context)),
        clock_control="none",
        profile_launch_count=3,
        timing_repeats=1,
        collect_source_counters=source_counters,
        source_counter_selection_reasons=("minimum",) if source_counters else (),
    )


def test_resolver_prefers_per_warp_active_ratio_and_requires_cycle_metrics():
    selection = _selection()

    assert selection.available is True
    assert selection.input_unit == "ratio"
    assert selection.family == "warp_issue_stalled_per_warp_active"
    assert selection.cycle_metrics == CYCLE_METRICS
    assert "mma" in selection.missing_reasons


def test_same_row_ratio_reduction_emits_all_three_views():
    record = reduce_stall_cycle_row(
        _ncu_result(), _selection(), repeat_index=0
    )

    assert record.selected_row_index == 2
    assert record.active_warp_cycles == 1000.0
    assert record.reason_warp_stall_cycles == {
        "selected": 120.0,
        "not_selected": 240.0,
        "wait": 480.0,
        "barrier": 360.0,
    }
    assert record.structural_denominator_warp_cycles == 840.0
    assert record.structural_fractions["wait"] == pytest.approx(4 / 7)
    assert "selected" not in record.structural_fractions
    assert record.raw_unit == RAW_STALL_CYCLE_UNIT
    assert record.conservation_status == "fail_overcoverage"


def test_row_selection_uses_structural_total_not_conserved_full_sum():
    rows = [
        _row(1.0),
        _row(1.0),
        _row(1.0),
    ]
    values = (
        (0.70, 0.20, 0.05, 0.05),
        (0.15, 0.15, 0.40, 0.30),
        (0.40, 0.20, 0.20, 0.20),
    )
    for row, (selected, not_selected, wait, barrier) in zip(rows, values):
        row[REASON_METRICS["selected"]] = str(selected)
        row[REASON_METRICS["not_selected"]] = str(not_selected)
        row[REASON_METRICS["wait"]] = str(wait)
        row[REASON_METRICS["barrier"]] = str(barrier)
    result = replace(_ncu_result(), raw_rows=rows)

    record = reduce_stall_cycle_row(result, _selection(), repeat_index=0)

    # Structural totals are 0.10, 0.70, and 0.40; the observed median is row 2.
    assert record.selected_row_index == 2
    assert record.issue_state_ratios["wait"] == 0.20


def test_same_row_reduction_rejects_selected_grid_mismatch():
    result = _ncu_result(scales=(1.0,))
    result.raw_rows[0]["Grid Size"] = "(2, 1, 1)"

    with pytest.raises(ValueError, match="exact launch context"):
        reduce_stall_cycle_row(result, _selection(), repeat_index=0)


def test_same_row_reduction_accepts_raw_ncu_launch_geometry_columns():
    result = _ncu_result(scales=(1.0,))
    row = result.raw_rows[0]
    row.pop("Grid Size")
    row.pop("Block Size")
    row.update(
        {
            "launch__grid_dim_x": "1",
            "launch__grid_dim_y": "1",
            "launch__grid_dim_z": "1",
            "launch__block_dim_x": "128.000000",
            "launch__block_dim_y": "1.000000",
            "launch__block_dim_z": "1.000000",
        }
    )

    record = reduce_stall_cycle_row(
        result, _selection(), repeat_index=0
    )

    assert record.selected_row_index == 0
    assert record.active_warp_cycles == 1000.0


def test_pct_fallback_divides_before_warp_cycle_conversion():
    record = reduce_stall_cycle_row(
        _ncu_result(scales=(1.0,), pct=True),
        _selection(pct=True),
        repeat_index=0,
    )

    assert record.issue_state_ratios["wait"] == 0.4
    assert record.reason_warp_stall_cycles["wait"] == 400.0
    assert record.input_unit == "pct"
    assert record.normalized_ratio_unit == "ratio"
    assert record.issue_state_source_values["wait"] == 40.0


def test_missing_state_coverage_is_quantified_without_zero_fill():
    selection = _selection(missing=("barrier",))
    result = _ncu_result(scales=(1.0,))
    result.raw_rows[0].pop(REASON_METRICS["barrier"])
    record = reduce_stall_cycle_row(result, selection, repeat_index=0)

    assert "barrier" not in record.issue_state_ratios
    assert record.missing_state_ratio == pytest.approx(0.3)
    assert record.conservation_status == "incomplete_missing_state_coverage_quantified"


def test_row_missing_supported_reason_is_explicit_and_diagnostic():
    result = _ncu_result(scales=(1.0,))
    result.raw_rows[0].pop(REASON_METRICS["barrier"])

    record = reduce_stall_cycle_row(result, _selection(), repeat_index=0)

    assert record.capability_missing_reasons == _selection().missing_reasons
    assert record.row_missing_reasons == ("barrier",)
    assert "barrier" in record.missing_reasons
    assert record.qualification_status == "diagnostic"
    assert "selected_row_missing_issue_states:barrier" in record.qualification_reasons


def test_row_selection_prefers_complete_rows_over_partial_rows():
    result = _ncu_result(scales=(0.1, 1.0, 1.2))
    result.raw_rows[0].pop(REASON_METRICS["barrier"])

    record = reduce_stall_cycle_row(result, _selection(), repeat_index=0)

    assert record.selected_row_index == 2
    assert record.row_missing_reasons == ()


def test_repeat_reducer_selects_observed_median_and_detects_identity_drift():
    selection = _selection()
    records = tuple(
        reduce_stall_cycle_row(
            _ncu_result(scales=(scale,), repeat=index),
            selection,
            repeat_index=index,
        )
        for index, scale in enumerate((0.9, 1.2, 1.0))
    )
    reduced = reduce_stall_cycle_repeats(records)

    assert reduced.selected_repeat_index == 2
    assert reduced.selected is records[2]

    drift = replace(
        records[2], target_evidence=_evidence(cubin="cubin-other")
    )
    invalid = reduce_stall_cycle_repeats((records[0], records[1], drift))
    assert invalid.identity_status == "invalid"
    assert invalid.qualification_status == "diagnostic"


def test_timing_and_aggregate_signatures_include_the_same_metadata():
    from amora.backends.nvidia.stall_cycle_measurement import (
        _evidence_signature,
        _signature,
    )

    record = reduce_stall_cycle_row(
        _ncu_result(scales=(1.0,)), _selection(), repeat_index=0
    )
    timing = _timing()
    timing_evidence = {
        "subject_identity": timing.subject_identity,
        "device": timing.device,
        "subject_metadata": timing.subject_metadata,
        "measurement_axes": timing.measurement_axes,
        "measurement_context": timing.measurement_context,
        "tool_versions": timing.tool_versions,
    }

    assert _signature(record) == _evidence_signature(timing_evidence)


def test_collect_repeats_passes_three_report_paths_and_preserves_rows(tmp_path):
    calls = []

    def collector(*args, **kwargs):
        repeat = int(kwargs["environment_overrides"]["AMORA_AGGREGATE_REPEAT_INDEX"])
        calls.append(kwargs)
        report = kwargs["report_path"]
        report.write_bytes(f"aggregate-{repeat}".encode())
        return replace(
            _ncu_result(
                scales=(1.4, 1.0, 1.2),
                repeat=repeat,
                cache_control=kwargs["cache_control"],
                clock_control=kwargs["clock_control"],
            ),
            report_path=report,
            report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
            cache_control=kwargs["cache_control"],
            clock_control=kwargs["clock_control"],
        )

    import hashlib

    reduced = collect_stall_cycle_repeats(
        _spec(),
        capabilities=NvidiaCapabilities(),
        selection=_selection(),
        report_dir=tmp_path,
        timeout=10,
        cwd=None,
        environment_overrides={},
        aggregate_collector=collector,
    )

    assert len(calls) == 3
    assert all(call["launch_count"] == 3 for call in calls)
    assert all(
        call["environment_overrides"]["AMORA_PROFILE_LAUNCH_COUNT"] == "3"
        for call in calls
    )
    assert all(
        call["environment_overrides"]["AMORA_CACHE_PROTOCOL"] == "warm_reuse"
        for call in calls
    )
    assert [call["report_path"].name for call in calls] == [
        "aggregate.repeat-01.ncu-rep",
        "aggregate.repeat-02.ncu-rep",
        "aggregate.repeat-03.ncu-rep",
    ]
    assert len(reduced.repeats) == 3


def test_campaign_requires_every_ordered_launch_not_just_launch_count():
    context = json.loads(json.dumps(CONTEXT))
    context["ordered_launches"] = [
        {
            "launch_ordinal": 0,
            "kernel_name": "kernel_a",
            "subject_identity": dict(IDENTITY),
        },
        {
            "launch_ordinal": 1,
            "kernel_name": "kernel_b",
            "subject_identity": {**IDENTITY, "kernel_name": "kernel_b"},
        },
    ]
    context["launch_ordinal"] = 0
    spec = replace(
        _spec(context=context),
        ncu_kernel_filter="regex:kernel_a",
        launch_skip=0,
        pc_launch_skip=0,
    )
    campaign = AllCornerCampaign("campaign", "registry", 1, 1, (spec,))

    from amora.backends.nvidia.stall_cycle_measurement import validate_all_corner_campaign

    with pytest.raises(ValueError, match="every ordered launch"):
        validate_all_corner_campaign(campaign)


def test_campaign_rejects_duplicate_launch_ordinal():
    context = json.loads(json.dumps(CONTEXT))
    first = _spec(point_id="point-a", context=context)
    second = _spec(point_id="point-b", context=context)
    campaign = AllCornerCampaign(
        "campaign", "registry", 1, 2, (first, second)
    )

    with pytest.raises(ValueError, match="duplicate launch ordinals"):
        from amora.backends.nvidia.stall_cycle_measurement import (
            validate_all_corner_campaign,
        )

        validate_all_corner_campaign(campaign)


def test_operation_timing_matches_selected_launch_identity_and_metadata():
    context = json.loads(json.dumps(CONTEXT))
    operation_context = json.loads(json.dumps(CONTEXT))
    operation_context.update(
        {
            "scope": "application_operation",
            "launch_ordinal": -1,
            "kernel_name": "operation_a",
        }
    )
    timing = replace(
        _timing(context=operation_context),
        subject_identity={**IDENTITY, "kernel_name": "operation_a"},
        subject_metadata={
            "ordered_launches": [
                {
                    "metadata": dict(METADATA),
                }
            ]
        },
    )
    spec = replace(
        _spec(context=context),
        operation_target=("operation-target",),
        operation_measurement_context=operation_context,
    )
    record = reduce_stall_cycle_row(
        _ncu_result(scales=(1.0,)), _selection(), repeat_index=0
    )

    assert _operation_timing_identity_reasons(
        timing,
        record,
        spec,
        required_subject_metadata_fields=tuple(METADATA),
    ) == []


def test_campaign_rejects_unsafe_run_and_campaign_ids(tmp_path):
    from amora.backends.nvidia.stall_cycle_measurement import (
        validate_all_corner_campaign,
    )

    with pytest.raises(ValueError, match="filesystem-safe"):
        validate_all_corner_campaign(
            AllCornerCampaign("../escape", "registry", 1, 1, (_spec(),))
        )
    with pytest.raises(ValueError, match="filesystem-safe"):
        execute_all_corner_campaign(
            AllCornerCampaign("campaign", "registry", 1, 1, (_spec(),)),
            capabilities=NvidiaCapabilities(),
            output_root=tmp_path,
            run_id="../escape",
        )


def test_campaign_requires_explicit_registry_launch_count():
    campaign = AllCornerCampaign("campaign", "registry", 1, 2, (_spec(),))

    from amora.backends.nvidia.stall_cycle_measurement import (
        validate_all_corner_campaign,
    )

    with pytest.raises(ValueError, match="corner registry"):
        validate_all_corner_campaign(campaign)


def test_campaign_requires_explicit_registry_corner_count():
    campaign = AllCornerCampaign("campaign", "registry", 2, 1, (_spec(),))

    from amora.backends.nvidia.stall_cycle_measurement import (
        validate_all_corner_campaign,
    )

    with pytest.raises(ValueError, match="corner count"):
        validate_all_corner_campaign(campaign)


@pytest.mark.parametrize(
    ("axis_name", "match"),
    [
        ("cache_protocol", "semantic cache protocol"),
        ("clock_policy", "clock policy"),
    ],
)
def test_campaign_requires_semantic_cache_and_clock_axes(axis_name, match):
    context = json.loads(json.dumps(CONTEXT))
    context["runtime_axes"].pop(axis_name)
    spec = replace(
        _spec(context=context),
        measurement_axes={
            key: value for key, value in AXES.items() if key != axis_name
        },
    )
    campaign = AllCornerCampaign("campaign", "registry", 1, 1, (spec,))

    from amora.backends.nvidia.stall_cycle_measurement import (
        validate_all_corner_campaign,
    )

    with pytest.raises(ValueError, match=match):
        validate_all_corner_campaign(campaign)


def test_campaign_rejects_unsafe_path_identifiers():
    campaign = AllCornerCampaign("../escape", "registry", 1, 1, (_spec(),))

    from amora.backends.nvidia.stall_cycle_measurement import (
        validate_all_corner_campaign,
    )

    with pytest.raises(ValueError, match="filesystem-safe"):
        validate_all_corner_campaign(campaign)


def test_execute_campaign_emits_launch_repeats_operation_totals_and_hashes(tmp_path):
    spec = _spec(source_counters=True)
    campaign = AllCornerCampaign("campaign", "registry", 1, 1, (spec,))
    capabilities = NvidiaCapabilities(ncu_metrics=frozenset(
        metric.rsplit(".", 1)[0]
        for metric in (*CYCLE_METRICS.values(), *REASON_METRICS.values())
    ))

    def aggregate(*args, **kwargs):
        repeat = int(kwargs["environment_overrides"]["AMORA_AGGREGATE_REPEAT_INDEX"])
        report = kwargs["report_path"]
        report.write_bytes(f"aggregate-{repeat}".encode())
        return replace(
            _ncu_result(
                repeat=repeat,
                cache_control=kwargs["cache_control"],
                clock_control=kwargs["clock_control"],
            ),
            report_path=report,
            report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
            cache_control=kwargs["cache_control"],
            clock_control=kwargs["clock_control"],
        )

    import hashlib

    def pc(*args, **kwargs):
        report = kwargs["report_path"]
        report.write_bytes(b"pc-report")
        return NcuPcSamplingResult(
            samples=[
                PcStallSample(
                    0x60,
                    "kernel_a",
                    "BRA",
                    40.0,
                    {"wait": 40.0},
                    sass_joined=True,
                    sass_opcode="BRA",
                    sass_instruction="/*0060*/ BRA ;",
                )
            ],
            report_path=report,
            report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
            target_evidence=_evidence(),
            cache_control=kwargs["cache_control"],
            clock_control=kwargs["clock_control"],
        )

    run_dir = execute_all_corner_campaign(
        campaign,
        capabilities=capabilities,
        output_root=tmp_path,
        run_id="run-1",
        timing_collector=lambda *a, **k: _timing(),
        aggregate_collector=aggregate,
        pc_collector=pc,
    )
    manifest = validate_all_corner_manifest(run_dir / "manifest.json")
    findings = json.loads(
        (run_dir / "aggregate_stall_cycle_findings.json").read_text()
    )

    assert set(manifest["compact_artifacts"]) == {
        "aggregate_stall_cycles.csv",
        "aggregate_stall_cycle_repeats.csv",
        "aggregate_stall_cycle_findings.json",
        "pc_samples_by_offset.csv",
    }
    assert findings["launch_count"] == 1
    assert findings["expected_corner_count"] == 1
    assert findings["expected_launch_count"] == 1
    assert manifest["expected_corner_count"] == 1
    assert manifest["expected_launch_count"] == 1
    assert findings["operation_totals"][0]["launch_count"] == 1
    assert len(findings["launches"][0]["aggregate_stall_cycles"]["repeats"]) == 3
    assert findings["launches"][0]["pc_sampling"]["repeat_count"] == 3


def test_campaign_json_round_trip(tmp_path):
    campaign = AllCornerCampaign("campaign", "registry", 1, 1, (_spec(),))
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(campaign.to_dict()))

    assert load_all_corner_campaign(path) == campaign


def test_operation_totals_sum_independently_reduced_launches():
    first_context = json.loads(json.dumps(CONTEXT))
    second_context = json.loads(json.dumps(CONTEXT))
    ordered = [
        {"launch_ordinal": 0, "kernel_name": "kernel_a", "subject_identity": dict(IDENTITY)},
        {"launch_ordinal": 1, "kernel_name": "kernel_b", "subject_identity": {**IDENTITY, "kernel_name": "kernel_b"}},
    ]
    first_context["ordered_launches"] = ordered
    second_context.update(
        {"ordered_launches": ordered, "launch_ordinal": 1, "kernel_name": "kernel_b"}
    )
    rows = [
        {
            "measurement_context": first_context,
            "qualification_status": "qualifying",
            "aggregate_stall_cycles": {
                "qualification_status": "qualifying",
                "selected": {
                    "active_warp_cycles": 100.0,
                    "reason_warp_stall_cycles": {"wait": 30.0, "selected": 10.0},
                },
            },
        },
        {
            "measurement_context": second_context,
            "qualification_status": "qualifying",
            "aggregate_stall_cycles": {
                "qualification_status": "qualifying",
                "selected": {
                    "active_warp_cycles": 200.0,
                    "reason_warp_stall_cycles": {"wait": 40.0, "barrier": 20.0},
                },
            },
        },
    ]

    total = _operation_totals(rows)[0]

    assert total["launch_count"] == 2
    assert total["launch_ordinals"] == [0, 1]
    assert total["active_warp_cycles"] == 300.0
    assert total["reason_warp_stall_cycles"] == {
        "barrier": 20.0, "selected": 10.0, "wait": 70.0
    }
    assert total["structural_denominator_warp_cycles"] == 90.0


def test_operation_totals_propagate_final_launch_qualification():
    row = {
        "measurement_context": CONTEXT,
        "qualification_status": "diagnostic",
        "aggregate_stall_cycles": {
            "qualification_status": "qualifying",
            "selected": {
                "active_warp_cycles": 100.0,
                "reason_warp_stall_cycles": {"wait": 30.0},
            },
        },
    }

    assert _operation_totals([row])[0]["qualification_status"] == "diagnostic"


def test_diagnostic_aggregate_automatically_selects_source_counters(tmp_path):
    spec = _spec(source_counters=False)
    campaign = AllCornerCampaign("campaign", "registry", 1, 1, (spec,))
    capabilities = NvidiaCapabilities(ncu_metrics=frozenset(
        metric.rsplit(".", 1)[0]
        for metric in (*CYCLE_METRICS.values(), *REASON_METRICS.values())
    ))
    pc_calls = []

    def aggregate(*args, **kwargs):
        result = _ncu_result(scales=(1.4,))
        report = kwargs["report_path"]
        report.write_bytes(b"aggregate")
        return replace(
            replace(
                result,
                cache_control=kwargs["cache_control"],
                clock_control=kwargs["clock_control"],
            ),
            report_path=report,
            report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
            cache_control=kwargs["cache_control"],
            clock_control=kwargs["clock_control"],
        )

    import hashlib

    def pc(*args, **kwargs):
        pc_calls.append(kwargs)
        report = kwargs["report_path"]
        report.write_bytes(b"pc")
        return NcuPcSamplingResult(
            samples=[
                PcStallSample(
                    0, "kernel_a", "NOP", 1.0, {"wait": 1.0},
                    sass_joined=True, sass_opcode="NOP",
                    sass_instruction="/*0000*/ NOP ;",
                )
            ],
            report_path=report,
            report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
            target_evidence=_evidence(),
            cache_control=kwargs["cache_control"],
            clock_control=kwargs["clock_control"],
        )

    run_dir = execute_all_corner_campaign(
        campaign, capabilities=capabilities, output_root=tmp_path, run_id="run",
        timing_collector=lambda *a, **k: _timing(),
        aggregate_collector=aggregate, pc_collector=pc,
    )
    findings = json.loads(
        (run_dir / "aggregate_stall_cycle_findings.json").read_text()
    )

    assert len(pc_calls) == 3
    assert findings["launches"][0]["source_counter_selection_reasons"] == [
        "failed_gate"
    ]


@pytest.mark.parametrize(
    ("drift_field", "drift_value", "expected_reason"),
    [
        ("cache_control", "all", "pc_repeat_01_cache_control_drift"),
        ("clock_control", "base", "pc_repeat_01_clock_control_drift"),
    ],
)
def test_source_counter_control_drift_makes_launch_diagnostic(
    tmp_path, drift_field, drift_value, expected_reason
):
    import hashlib

    spec = _spec(source_counters=True)
    campaign = AllCornerCampaign("campaign", "registry", 1, 1, (spec,))
    capabilities = NvidiaCapabilities(
        ncu_metrics=frozenset(
            metric.rsplit(".", 1)[0]
            for metric in (*CYCLE_METRICS.values(), *REASON_METRICS.values())
        )
    )

    def aggregate(*args, **kwargs):
        repeat = int(kwargs["environment_overrides"]["AMORA_AGGREGATE_REPEAT_INDEX"])
        report = kwargs["report_path"]
        report.write_bytes(f"aggregate-{repeat}".encode())
        return replace(
            _ncu_result(
                scales=(1.0,),
                repeat=repeat,
                cache_control=kwargs["cache_control"],
                clock_control=kwargs["clock_control"],
            ),
            report_path=report,
            report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        )

    def pc(*args, **kwargs):
        repeat = int(kwargs["environment_overrides"]["AMORA_PC_SAMPLING_REPEAT_INDEX"])
        report = kwargs["report_path"]
        report.write_bytes(f"pc-{repeat}".encode())
        controls = {
            "cache_control": kwargs["cache_control"],
            "clock_control": kwargs["clock_control"],
        }
        if repeat == 0:
            controls[drift_field] = drift_value
        return NcuPcSamplingResult(
            samples=[
                PcStallSample(
                    0,
                    "kernel_a",
                    "NOP",
                    1.0,
                    {"wait": 1.0},
                    sass_joined=True,
                    sass_opcode="NOP",
                    sass_instruction="/*0000*/ NOP ;",
                )
            ],
            report_path=report,
            report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
            target_evidence=_evidence(),
            **controls,
        )

    run_dir = execute_all_corner_campaign(
        campaign,
        capabilities=capabilities,
        output_root=tmp_path,
        run_id=f"run-{drift_field}",
        timing_collector=lambda *a, **k: _timing(),
        aggregate_collector=aggregate,
        pc_collector=pc,
    )
    findings = json.loads(
        (run_dir / "aggregate_stall_cycle_findings.json").read_text()
    )
    launch = findings["launches"][0]

    assert launch["qualification_status"] == "diagnostic"
    assert expected_reason in launch["identity_reasons"]
