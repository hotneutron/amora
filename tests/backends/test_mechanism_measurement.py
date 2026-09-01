"""Tests for physical-mechanism recipes and immutable evidence runs."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.cuda_event_run import (
    CudaEventProcessSample,
    CudaEventTimingResult,
    DeviceInterval,
    DeviceIntervalResult,
)
from amora.backends.nvidia.measurement import CommandMeasurementBundle
from amora.backends.nvidia.mechanism_measurement import (
    InteractionGroup,
    MechanismPoint,
    MechanismRecipe,
    build_mediation_report,
    collect_causal_evidence,
    execute_mechanism_recipe,
    load_mechanism_recipe,
    load_mechanism_manifest,
    validate_mechanism_recipe,
)


IDENTITY = {
    "ttgir_sha256": "ttgir",
    "ptx_sha256": "ptx",
    "sass_sha256": "sass",
    "cubin_sha256": "cubin",
}
DEVICE = {"uuid": "GPU-test", "name": "H100"}
WGMMA_BINDINGS = {
    "footprint_occupancy": (
        "bundle.aggregate_counters.metrics.launch__occupancy_limit_shared_mem",
        "axes.cta_load",
    ),
    "wgmma_issue": (
        "bundle.aggregate_counters.metrics.wgmma_issued",
        "bundle.pc_sampling.by_sass_opcode.HGMMA.support_count",
    ),
    "wgmma_exposed_wait": (
        "bundle.device_intervals.intervals.name=wgmma_exposed_wait.duration_ns",
    ),
    "wgmma_completion": (
        "bundle.device_intervals.intervals.name=wgmma_completion.duration_ns",
    ),
    "stage_buffer_release": (
        "bundle.device_intervals.intervals.name=stage_buffer_release.duration_ns",
    ),
    "total_latency": ("bundle.timing.median_us",),
}
TMA_BINDINGS = {
    "footprint_occupancy": (
        "bundle.aggregate_counters.metrics.launch__occupancy_limit_shared_mem",
        "subject_metadata.shared_memory_bytes",
    ),
    "producer_eligibility": ("bundle.aggregate_counters.metrics.eligible_warps",),
    "tma_issue": (
        "bundle.aggregate_counters.metrics.tma_requests",
        "bundle.pc_sampling.by_sass_opcode.UTMALDG.support_count",
    ),
    "memory_queue_partition": (
        "bundle.aggregate_counters.structural_stall_histogram.reason_fractions.long_scoreboard",
        "axes.compiled_queue_capacity",
    ),
    "memory_route": ("bundle.aggregate_counters.metrics.dram_bytes",),
    "tma_completion": (
        "bundle.device_intervals.intervals.name=tma_completion.duration_ns",
    ),
    "barrier_release": (
        "bundle.device_intervals.intervals.name=barrier_release.duration_ns",
    ),
    "stage_buffer_release": ("bundle.aggregate_counters.metrics.stage_releases",),
    "total_latency": ("bundle.timing.median_us",),
}


def _wgmma_point(
    point_id, *, cta_load=1, split="calibration", spills=0, **axis_overrides
):
    axes = {
        "wgmma_repeats": 16,
        "independent_register_work_gap_ns": 8,
        "outstanding_wgmma_groups": 2,
        "instruction_shape": "m64n128k16",
        "cta_load": cta_load,
    }
    axes.update(axis_overrides)
    return MechanismPoint(
        point_id=point_id,
        split=split,
        axes=axes,
        timing_target=("timing", point_id),
        ncu_target=("ncu", point_id),
        device_interval_target=("intervals", point_id),
        expected_launch_batch_size=4,
        expected_cache_protocol="warm_reuse",
        expected_max_interval_overhead_percent=2.0,
        evidence_bindings=WGMMA_BINDINGS,
        subject_metadata={
            "registers_per_thread": 64,
            "shared_memory_bytes": 32768,
            "spill_count": spills,
        },
    )


def _wgmma_recipe(*, second_spills=0):
    points = [
        _wgmma_point("baseline"),
        _wgmma_point("one_cta_per_sm", cta_load=132, split="held_out"),
        _wgmma_point("repeats_32", wgmma_repeats=32),
        _wgmma_point(
            "gap_16", independent_register_work_gap_ns=16, split="held_out"
        ),
        _wgmma_point("groups_3", outstanding_wgmma_groups=3),
        _wgmma_point(
            "shape_m64n64k16",
            instruction_shape="m64n64k16",
            split="held_out",
        ),
    ]
    if second_spills:
        points.append(
            _wgmma_point(
                "spill_diagnostic",
                split="held_out",
                spills=second_spills,
            )
        )
    return MechanismRecipe(
        recipe_id="wgmma-completion-v1",
        mechanism="wgmma_fixed_completion",
        kernel_name="wgmma_probe",
        aggregate_metrics=(
            "smsp__warp_issue_stalled_wait_per_warp_active.pct",
            "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
            "smsp__warp_issue_stalled_gmma_per_warp_active.pct",
            "wgmma_issued",
        ),
        points=tuple(points),
        interaction_groups=(
            InteractionGroup(
                group_id="cta-load",
                point_ids=("baseline", "one_cta_per_sm"),
                varied_axis="cta_load",
                intervention_node="footprint_occupancy",
            ),
            InteractionGroup(
                group_id="wgmma-repeats",
                point_ids=("baseline", "repeats_32"),
                varied_axis="wgmma_repeats",
                intervention_node="wgmma_issue",
                requires_same_subject_identity=False,
            ),
            InteractionGroup(
                group_id="register-gap",
                point_ids=("baseline", "gap_16"),
                varied_axis="independent_register_work_gap_ns",
                intervention_node="wgmma_completion",
                requires_same_subject_identity=False,
            ),
            InteractionGroup(
                group_id="outstanding-groups",
                point_ids=("baseline", "groups_3"),
                varied_axis="outstanding_wgmma_groups",
                intervention_node="stage_buffer_release",
                requires_same_subject_identity=False,
            ),
            InteractionGroup(
                group_id="instruction-shape",
                point_ids=("baseline", "shape_m64n64k16"),
                varied_axis="instruction_shape",
                intervention_node="wgmma_issue",
                requires_same_subject_identity=False,
            ),
        ),
        repeats=2,
        randomization_seed=7,
    )


def _tma_point(point_id, *, split="calibration", **axis_overrides):
    axes = {
        "request_bytes": 4096,
        "iteration_count": 128,
        "compiled_queue_capacity": 2,
        "rotating_working_set_bytes": 4 * 1024 * 1024,
        "cache_protocol": "disjoint_rotation",
        "cta_concurrency": 132,
        "address_partition_mapping": "round_robin",
        "per_cta_footprint_bytes": 32768,
    }
    axes.update(axis_overrides)
    return MechanismPoint(
        point_id=point_id,
        split=split,
        axes=axes,
        timing_target=("timing", point_id),
        ncu_target=("ncu", point_id),
        device_interval_target=("intervals", point_id),
        evidence_bindings=TMA_BINDINGS,
        subject_metadata={
            "registers_per_thread": 48,
            "shared_memory_bytes": 32768,
            "spill_count": 0,
        },
    )


def _bundle(completion_ns, *, exposed_wait_ns=None, axes=None):
    if exposed_wait_ns is None:
        exposed_wait_ns = completion_ns
    process = CudaEventProcessSample(
        process_index=0,
        samples_us=(20.0,),
        warmup_launches=2,
        launch_batch_size=4,
        cache_protocol="warm_reuse",
        clock_policy="application_clocks",
        device=DEVICE,
        subject_identity=IDENTITY,
        subject_metadata={
            "registers_per_thread": 64,
            "shared_memory_bytes": 32768,
            "spill_count": 0,
        },
        measurement_axes=dict(axes or {}),
        tool_versions={"target": "test"},
        command=("timing",),
        cwd=None,
        environment_overrides={},
        stdout="",
        stderr="",
        returncode=0,
        started_at="start",
        completed_at="end",
    )
    timing = CudaEventTimingResult(
        target_command=("timing",),
        process_samples=(process,),
        median_us=20.0,
        p05_us=20.0,
        p95_us=20.0,
        process_cv=0.0,
        launch_batch_size=4,
        cache_protocol="warm_reuse",
        clock_policy="application_clocks",
        device=DEVICE,
        subject_identity=IDENTITY,
        subject_metadata={
            "registers_per_thread": 64,
            "shared_memory_bytes": 32768,
            "spill_count": 0,
        },
        measurement_axes=dict(axes or {}),
        tool_versions={"target": "test"},
        provenance={},
    )
    intervals = DeviceIntervalResult(
        clock="globaltimer_ns",
        intervals=(
            DeviceInterval(
                "wgmma_exposed_wait", 0.0, exposed_wait_ns, exposed_wait_ns
            ),
            DeviceInterval("wgmma_completion", 0.0, completion_ns, completion_ns),
            DeviceInterval("stage_buffer_release", 0.0, 200.0, 200.0),
        ),
        instrumentation={},
        device=DEVICE,
        subject_identity=IDENTITY,
        subject_metadata={
            "registers_per_thread": 64,
            "shared_memory_bytes": 32768,
            "spill_count": 0,
        },
        measurement_axes=dict(axes or {}),
        tool_versions={"target": "test"},
        evidence_status="qualifying",
        diagnostic_reasons=(),
    )
    return CommandMeasurementBundle(
        timing=timing,
        device_intervals=intervals,
        aggregate_counters={"metrics": {"wgmma_issued": 16.0}},
        profiler_duration={"unit": "ns", "metrics": {"gpu__time_duration.sum": 30_000.0}},
        pc_sampling={
            "raw_unit": "pc_sample_count",
            "support_count": 12.0,
            "instruction_join_fraction": 1.0,
            "by_sass_opcode": {
                "HGMMA": {
                    "support_count": 12.0,
                    "stall_counts": {"wait": 12.0},
                }
            },
        },
        identity_check={"status": "pass", "reasons": []},
        cache_control="none",
        capabilities={},
    )


def test_recipe_requires_controlled_cta_load_interaction():
    recipe = _wgmma_recipe()
    validate_mechanism_recipe(recipe)

    drifting = replace(
        recipe,
        points=(
            recipe.points[0],
            replace(
                recipe.points[1],
                axes={
                    **recipe.points[1].axes,
                    "independent_register_work_gap_ns": 16,
                },
            ),
        ),
    )
    with pytest.raises(ValueError, match="control axis independent_register_work_gap_ns"):
        validate_mechanism_recipe(drifting)


def test_recipe_json_round_trip(tmp_path):
    recipe = _wgmma_recipe()
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe.to_dict()))

    loaded = load_mechanism_recipe(path)

    assert loaded == recipe


def test_tma_recipe_requires_route_reuse_interaction_and_physical_nodes():
    points = (
        _tma_point("baseline"),
        _tma_point("request_8192", request_bytes=8192),
        _tma_point("iterations_256", iteration_count=256),
        _tma_point("queue_3", compiled_queue_capacity=3),
        _tma_point(
            "hbm_route",
            rotating_working_set_bytes=128 * 1024 * 1024,
            split="held_out",
        ),
        _tma_point("warm_reuse", cache_protocol="warm_reuse"),
        _tma_point("one_cta", cta_concurrency=1, split="held_out"),
        _tma_point(
            "partition_zero", address_partition_mapping="partition_zero"
        ),
        _tma_point("footprint_65536", per_cta_footprint_bytes=65536),
    )
    recipe = MechanismRecipe(
        recipe_id="tma-route-v1",
        mechanism="tma_route_offered_load",
        kernel_name="tma_probe",
        aggregate_metrics=(
            "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
            "lts__t_sectors_aperture_device_op_read.sum",
            "lts__t_sectors_aperture_device_op_read_lookup_hit.sum",
            "lts__t_sectors_aperture_device_op_read_lookup_miss.sum",
            "dram__bytes_read.sum",
            "launch__waves_per_multiprocessor",
            "launch__occupancy_limit_shared_mem",
        ),
        points=points,
        interaction_groups=(
            InteractionGroup(
                group_id="reuse-distance",
                point_ids=("baseline", "hbm_route"),
                varied_axis="rotating_working_set_bytes",
                intervention_node="memory_route",
            ),
            InteractionGroup(
                group_id="cta-concurrency",
                point_ids=("baseline", "one_cta"),
                varied_axis="cta_concurrency",
                intervention_node="footprint_occupancy",
            ),
            InteractionGroup(
                group_id="request-bytes",
                point_ids=("baseline", "request_8192"),
                varied_axis="request_bytes",
                intervention_node="tma_issue",
            ),
            InteractionGroup(
                group_id="iteration-count",
                point_ids=("baseline", "iterations_256"),
                varied_axis="iteration_count",
                intervention_node="tma_issue",
            ),
            InteractionGroup(
                group_id="queue-capacity",
                point_ids=("baseline", "queue_3"),
                varied_axis="compiled_queue_capacity",
                intervention_node="memory_queue_partition",
                requires_same_subject_identity=False,
            ),
            InteractionGroup(
                group_id="cache-protocol",
                point_ids=("baseline", "warm_reuse"),
                varied_axis="cache_protocol",
                intervention_node="memory_route",
            ),
            InteractionGroup(
                group_id="partition-mapping",
                point_ids=("baseline", "partition_zero"),
                varied_axis="address_partition_mapping",
                intervention_node="memory_queue_partition",
            ),
            InteractionGroup(
                group_id="cta-footprint",
                point_ids=("baseline", "footprint_65536"),
                varied_axis="per_cta_footprint_bytes",
                intervention_node="footprint_occupancy",
                requires_same_subject_identity=False,
            ),
        ),
    )

    validate_mechanism_recipe(recipe)
    assert recipe.to_dict()["modeling_guard"]["direct_regressors_forbidden"] == [
        "elapsed_duration",
        "K",
        "raw_working_set_bytes",
    ]

    incomplete = replace(
        recipe,
        points=(
            replace(
                points[0],
                evidence_bindings={
                    key: value for key, value in TMA_BINDINGS.items() if key != "memory_route"
                },
            ),
            points[1],
        ),
    )
    with pytest.raises(ValueError, match="memory_route"):
        validate_mechanism_recipe(incomplete)


def test_mediation_classifies_cta_sensitive_completion_as_shared_resource():
    recipe = _wgmma_recipe()
    point_results = {}
    for point in recipe.points:
        completion_ns = 160.0 if point.point_id == "one_cta_per_sm" else 100.0
        point_results[point.point_id] = {
            "axes": point.axes,
                "bundle": _bundle(completion_ns, axes=point.axes).to_dict(),
            "causal_evidence": collect_causal_evidence(
                point, _bundle(completion_ns, axes=point.axes)
            ),
        }

    report = build_mediation_report(recipe, point_results)

    assert report["wgmma_fixed_completion_assessment"]["classification"] == (
        "shared_resource_or_scheduling_interaction"
    )
    assert report["wgmma_gap_knee_assessment"]["classification"] == (
        "fixed_completion_knee_falsified"
    )
    assert report["direct_regressors_emitted"] == []


def test_mediation_recognizes_fixed_completion_gap_knee():
    recipe = _wgmma_recipe()
    point_results = {}
    for point in recipe.points:
        exposed_wait = (
            92.0
            if point.axes["independent_register_work_gap_ns"] == 8
            else 84.0
        )
        bundle = _bundle(100.0, exposed_wait_ns=exposed_wait, axes=point.axes)
        point_results[point.point_id] = {
            "axes": point.axes,
            "bundle": bundle.to_dict(),
            "causal_evidence": collect_causal_evidence(point, bundle),
        }

    report = build_mediation_report(recipe, point_results)

    assert report["wgmma_gap_knee_assessment"]["classification"] == (
        "fixed_completion_knee_candidate"
    )


def test_causal_evidence_resolves_metric_names_with_dots():
    point = replace(
        _wgmma_point("baseline"),
        evidence_bindings={
            **WGMMA_BINDINGS,
            "memory_route": (
                "bundle.aggregate_counters.metrics.dram__bytes_read.sum",
            ),
        },
    )
    bundle = replace(
        _bundle(100.0, axes=point.axes),
        aggregate_counters={"metrics": {"dram__bytes_read.sum": 4096.0}},
    )

    evidence = collect_causal_evidence(point, bundle)

    assert evidence["nodes"]["memory_route"]["measurements"] == {
        "bundle.aggregate_counters.metrics.dram__bytes_read.sum": 4096.0
    }


def test_execute_recipe_writes_immutable_digested_artifacts_and_marks_spills(tmp_path):
    recipe = _wgmma_recipe(second_spills=2)

    def collector(**kwargs):
        return _bundle(
            160.0
            if kwargs["timing_target"][-1] == "one_cta_per_sm"
            else 100.0,
            axes=kwargs["expected_measurement_axes"],
        )

    run_dir = execute_mechanism_recipe(
        recipe,
        capabilities=NvidiaCapabilities(),
        output_root=tmp_path,
        run_id="run-1",
        bundle_collector=collector,
    )
    manifest = load_mechanism_manifest(run_dir / "manifest.json")

    assert manifest["split_counts"] == {"calibration": 3, "held_out": 4}
    assert manifest["execution_order"] != []
    assert manifest["artifacts"]["spill_diagnostic"]["qualification_status"] == "diagnostic"
    with pytest.raises(FileExistsError):
        execute_mechanism_recipe(
            recipe,
            capabilities=NvidiaCapabilities(),
            output_root=tmp_path,
            run_id="run-1",
            bundle_collector=collector,
        )

    bundle_path = run_dir / manifest["artifacts"]["baseline"]["path"]
    bundle_path.write_text("{}\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_mechanism_manifest(run_dir / "manifest.json")
