"""Tests for physical-mechanism recipes and immutable evidence runs."""

from __future__ import annotations

import json
import hashlib
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
    build_barrier_topology_report,
    collect_causal_evidence,
    execute_mechanism_recipe,
    load_mechanism_recipe,
    load_mechanism_manifest,
    localize_pc_regions,
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
BARRIER_BINDINGS = {
    "footprint_occupancy": (
        "bundle.aggregate_counters.metrics.launch__occupancy_limit_shared_mem",
    ),
    "producer_eligibility": (
        "bundle.aggregate_counters.metrics.smsp__eligible_warps_per_scheduler.avg",
    ),
    "tma_issue": (
        "bundle.aggregate_counters.metrics.l1tex__t_bytes_mem_global_op_tma_ld.sum",
        "bundle.pc_sampling.by_sass_opcode.UTMALDG.support_count",
    ),
    "memory_route": (
        "bundle.aggregate_counters.metrics.dram__bytes_read.sum",
    ),
    "tma_completion": (
        "bundle.device_intervals.intervals.name=tma_issue_to_barrier_release.duration_ns",
        "bundle.device_intervals.intervals.name=slow_tma_issue_to_barrier_release.duration_ns",
    ),
    "barrier_release": (
        "bundle.device_intervals.intervals.name=barrier_release_to_consumer_issue.start_ns",
        "bundle.device_intervals.intervals.name=fast_barrier_release_to_consumer_issue.start_ns",
        "bundle.pc_sampling.by_sass_opcode.MBAR.support_count",
    ),
    "consumer_issue": (
        "bundle.device_intervals.intervals.name=barrier_release_to_consumer_issue.end_ns",
        "bundle.device_intervals.intervals.name=fast_barrier_release_to_consumer_issue.end_ns",
        "bundle.pc_sampling.by_sass_opcode.HGMMA.support_count",
    ),
    "wgmma_completion": (
        "bundle.device_intervals.intervals.name=wgmma_issue_to_completion.duration_ns",
    ),
    "stage_buffer_release": (
        "bundle.device_intervals.intervals.name=stage_buffer_release.duration_ns",
    ),
    "total_latency": ("bundle.timing.median_us",),
}
BARRIER_METRICS = (
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_gmma_per_warp_active.pct",
    "smsp__warp_issue_stalled_selected_per_warp_active.pct",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "smsp__inst_executed_op_gmma.sum",
    "l1tex__t_bytes_mem_global_op_tma_ld.sum",
    "sm__warps_active.avg",
    "smsp__eligible_warps_per_scheduler.avg",
    "launch__occupancy_limit_shared_mem",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_read_lookup_hit.sum",
    "lts__t_sectors_op_read_lookup_miss.sum",
    "dram__bytes_read.sum",
)
BARRIER_IDENTITY_FIELDS = (
    "kernel_name",
    "source_sha256",
    "ttgir_sha256",
    "ptx_sha256",
    "sass_sha256",
    "cubin_sha256",
)
BARRIER_ALLOWED_IDENTITY_DIFFERENCES = (
    "source_sha256",
    "ttgir_sha256",
    "ptx_sha256",
    "sass_sha256",
    "cubin_sha256",
)
BARRIER_ALLOWED_SASS_DIFFERENCES = (
    "synchronization",
    "predicate",
    "branch",
    "barrier_address_setup",
)
BARRIER_REGIONS = {
    "tma_issue": (0x20,),
    "mbarrier_poll_fast": (0x40,),
    "mbarrier_poll_retry": (0x60,),
    "wgmma_wait": (0x80,),
    "wgmma_issue": (0xA0,),
    "unrelated_prologue_epilogue": (0xC0,),
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


def _barrier_point(point_id, *, panel, topology, split="calibration"):
    axes = {
        "panel": panel,
        "topology": topology,
        "M": 128,
        "N": 128,
        "K": 256,
        "tile_shape": "128x128x64",
        "grid": "132x1x1",
        "pipeline_depth": 3,
        "producer_asymmetry_level": "transition",
        "cache_protocol": "disjoint_rotation",
        "cta_concurrency": 132,
        "address_partition_mapping": "round_robin",
        "launch_batch_size": 8,
        "clock_policy": "base",
    }
    return MechanismPoint(
        point_id=point_id,
        split=split,
        axes=axes,
        timing_target=("timing", point_id),
        ncu_target=("ncu", point_id),
        device_interval_target=("intervals", point_id),
        evidence_bindings=BARRIER_BINDINGS,
        pc_region_map=BARRIER_REGIONS,
        subject_metadata={
            "registers_per_thread": 64,
            "shared_memory_bytes": 65536,
            "spill_count": 0,
            "total_warps": 8,
            "total_threads": 256,
            "actor_count": 2,
            "compiled_queue_capacity": 3,
            "producer_lead": 2,
            "request_count": 2,
            "request_bytes": 32768,
            "numerical_result_valid": True,
            "numerical_result_fingerprint": "correct-result",
            "useful_operation_fingerprint": {
                "tma": "2x32768-byte-load",
                "hgmma": "m128n128k64",
            },
            "synchronization_instruction_fingerprint": {
                "digest": f"sync-{topology}",
                "non_sync_digest": "same-non-sync",
                "difference_classes": [
                    "synchronization",
                    "predicate",
                    "branch",
                    "barrier_address_setup",
                ],
            },
        },
        expected_launch_batch_size=8,
        expected_cache_protocol="disjoint_rotation",
        expected_max_interval_overhead_percent=5.0,
    )


def _barrier_recipe():
    points = (
        _barrier_point("eq-joint", panel="BT-EQ", topology="joint"),
        _barrier_point("eq-split", panel="BT-EQ", topology="split"),
        _barrier_point(
            "eq-joint-held", panel="BT-EQ", topology="joint", split="held_out"
        ),
        _barrier_point(
            "eq-split-held", panel="BT-EQ", topology="split", split="held_out"
        ),
        _barrier_point(
            "causal-joint", panel="BT-CAUSAL", topology="joint"
        ),
        _barrier_point(
            "causal-pairwise",
            panel="BT-CAUSAL",
            topology="pairwise",
        ),
        _barrier_point(
            "causal-joint-held",
            panel="BT-CAUSAL",
            topology="joint",
            split="held_out",
        ),
        _barrier_point(
            "causal-pairwise-held",
            panel="BT-CAUSAL",
            topology="pairwise",
            split="held_out",
        ),
    )
    groups = tuple(
        InteractionGroup(
            group_id=group_id,
            point_ids=point_ids,
            varied_axis="topology",
            intervention_node="barrier_release",
            requires_same_subject_identity=False,
            invariant_counter_paths=(
                "aggregate_counters.metrics.smsp__inst_executed_op_gmma.sum",
                "aggregate_counters.metrics.l1tex__t_bytes_mem_global_op_tma_ld.sum",
                "aggregate_counters.metrics.launch__occupancy_limit_shared_mem",
            ),
            allowed_identity_differences=BARRIER_ALLOWED_IDENTITY_DIFFERENCES,
            allowed_subject_metadata_differences=(
                "synchronization_instruction_fingerprint",
            ),
            allowed_sass_difference_classes=BARRIER_ALLOWED_SASS_DIFFERENCES,
        )
        for group_id, point_ids in (
            ("eq-pair", ("eq-joint", "eq-split")),
            ("eq-pair-held", ("eq-joint-held", "eq-split-held")),
            ("causal-pair", ("causal-joint", "causal-pairwise")),
            (
                "causal-pair-held",
                ("causal-joint-held", "causal-pairwise-held"),
            ),
        )
    )
    return MechanismRecipe(
        recipe_id="barrier-topology-v1",
        mechanism="barrier_topology_release",
        kernel_name="gluon_barrier_probe",
        aggregate_metrics=BARRIER_METRICS,
        points=points,
        interaction_groups=groups,
        clock_control="base",
        repeats=7,
        pc_sampling_repeats=3,
        required_identity_fields=BARRIER_IDENTITY_FIELDS,
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


def _barrier_bundle(
    point, *, latency_us=20.0, counter_drift=False,
    mediation_drift=False, pc_repeat_count=3
):
    topology = point.axes["topology"]
    identity = {
        "kernel_name": "gluon_barrier_probe",
        "source_sha256": f"source-{topology}",
        "ttgir_sha256": f"ttgir-{topology}",
        "ptx_sha256": f"ptx-{topology}",
        "sass_sha256": f"sass-{topology}",
        "cubin_sha256": f"cubin-{topology}",
    }
    processes = tuple(
        CudaEventProcessSample(
            process_index=index,
            samples_us=(latency_us + index * 0.001,),
            warmup_launches=2,
            launch_batch_size=8,
            cache_protocol="disjoint_rotation",
            clock_policy="base",
            device=DEVICE,
            subject_identity=identity,
            subject_metadata=point.subject_metadata,
            measurement_axes=point.axes,
            tool_versions={"target": "test"},
            command=("timing", point.point_id),
            cwd=None,
            environment_overrides={},
            stdout="",
            stderr="",
            returncode=0,
            started_at="start",
            completed_at="end",
        )
        for index in range(7)
    )
    timing = CudaEventTimingResult(
        target_command=("timing", point.point_id),
        process_samples=processes,
        median_us=latency_us + 0.003,
        p05_us=latency_us,
        p95_us=latency_us + 0.006,
        process_cv=0.001,
        launch_batch_size=8,
        cache_protocol="disjoint_rotation",
        clock_policy="base",
        device=DEVICE,
        subject_identity=identity,
        subject_metadata=point.subject_metadata,
        measurement_axes=point.axes,
        tool_versions={"target": "test"},
        provenance={},
    )
    if point.axes["panel"] == "BT-EQ":
        interval_values = (
            ("tma_issue_to_barrier_release", 0.0, 100.0),
            ("barrier_release_to_consumer_issue", 100.0, 110.0),
            ("wgmma_issue_to_completion", 110.0, 210.0),
            ("stage_buffer_release", 210.0, 220.0),
        )
    else:
        fast_consumer_end = 180.0 if topology == "joint" else 90.0
        interval_values = (
            ("fast_tma_issue_to_barrier_release", 0.0, 80.0),
            ("slow_tma_issue_to_barrier_release", 0.0, 160.0),
            ("fast_barrier_release_to_consumer_issue", 80.0, fast_consumer_end),
            ("slow_barrier_release_to_consumer_issue", 160.0, 180.0),
            ("wgmma_issue_to_completion", 180.0, 280.0),
            ("stage_buffer_release", 280.0, 290.0),
        )
    intervals = DeviceIntervalResult(
        clock="globaltimer_ns",
        intervals=tuple(
            DeviceInterval(name, start, end, end - start)
            for name, start, end in interval_values
        ),
        instrumentation={
            "sass_bracketing_verified": True,
            "instruction_sequence_unchanged": True,
            "unprofiled_event_overhead_percent": 1.0,
            "overhead_threshold_percent": 5.0,
        },
        device=DEVICE,
        subject_identity=identity,
        subject_metadata=point.subject_metadata,
        measurement_axes=point.axes,
        tool_versions={"target": "test"},
        evidence_status="qualifying",
        diagnostic_reasons=(),
    )
    useful_scale = 1.2 if counter_drift else 1.0
    aggregate = {
        "metrics": {
            "smsp__inst_executed_op_gmma.sum": 128.0 * useful_scale,
            "l1tex__t_bytes_mem_global_op_tma_ld.sum": 65536.0,
            "sm__warps_active.avg": 8.0,
            "smsp__eligible_warps_per_scheduler.avg": 2.0,
            "launch__occupancy_limit_shared_mem": 2.0,
            "lts__t_sectors_op_read.sum": 2048.0,
            "lts__t_sectors_op_read_lookup_hit.sum": 1024.0,
            "lts__t_sectors_op_read_lookup_miss.sum": (
                4096.0 if mediation_drift else 1024.0
            ),
            "dram__bytes_read.sum": (
                262144.0 if mediation_drift else 65536.0
            ),
        },
        "missing_metrics": [],
        "structural_stall_histogram": {
            "raw_unit": "pct_per_warp_active_cycle",
            "reason_values": {"barrier": 20.0, "wait": 10.0},
            "denominator": 30.0,
            "reason_fractions": {"barrier": 2 / 3, "wait": 1 / 3},
            "excluded_reasons": ["selected", "not_selected"],
        },
    }
    repeat_rows = [
        {
            "repeat_index": index,
            "samples": [
                {
                    "pc_offset": 0x60,
                    "samples": 40.0,
                    "stalls": {"barrier": 40.0},
                    "sass_joined": True,
                    "sass_opcode": "MBAR",
                    "sass_instruction": "/*0060*/ MBAR.TRY_WAIT ;",
                }
            ],
            "support_count": 40.0,
            "instruction_join_fraction": 1.0,
            "provenance": {"report_sha256": f"report-{index}"},
        }
        for index in range(pc_repeat_count)
    ]
    pc_sampling = {
        "raw_unit": "pc_sample_count",
        "repeat_count": pc_repeat_count,
        "repeats": repeat_rows,
        "samples": [sample for row in repeat_rows for sample in row["samples"]],
        "by_offset": [
            {
                "function": "gluon_barrier_probe",
                "pc_offset": 0x60,
                "sass_opcode": "MBAR",
                "sass_instruction": "/*0060*/ MBAR.TRY_WAIT ;",
                "support_count": 40.0 * pc_repeat_count,
                "stall_counts": {"barrier": 40.0 * pc_repeat_count},
                "repeat_support_counts": [40.0] * pc_repeat_count,
            }
        ],
        "support_count": 40.0 * pc_repeat_count,
        "instruction_join_fraction": 1.0,
        "by_sass_opcode": {
            "MBAR": {
                "support_count": 40.0 * pc_repeat_count,
                "stall_counts": {"barrier": 40.0 * pc_repeat_count},
            }
        },
        "provenance": {"repeat_count": pc_repeat_count, "reports": []},
    }
    return CommandMeasurementBundle(
        timing=timing,
        device_intervals=intervals,
        aggregate_counters=aggregate,
        profiler_duration={
            "unit": "ns",
            "metrics": {"gpu__time_duration.sum": 30_000.0},
            "role": "diagnostic_profiler_perturbation_only",
        },
        pc_sampling=pc_sampling,
        identity_check={
            "status": "pass",
            "reasons": [],
            "subject_identity": identity,
        },
        cache_control="none",
        clock_control="base",
        capabilities={},
    )


def _barrier_point_result(
    point, *, latency_us=20.0, counter_drift=False, mediation_drift=False
):
    bundle = _barrier_bundle(
        point, latency_us=latency_us, counter_drift=counter_drift,
        mediation_drift=mediation_drift
    )
    return {
        "point_id": point.point_id,
        "split": point.split,
        "axes": point.axes,
        "subject_metadata": point.subject_metadata,
        "pc_region_map": {
            region: list(offsets) for region, offsets in point.pc_region_map.items()
        },
        "qualification_status": "qualifying",
        "diagnostic_reasons": [],
        "bundle": bundle.to_dict(),
        "causal_evidence": collect_causal_evidence(point, bundle),
        "pc_localization": localize_pc_regions(
            bundle.pc_sampling, point.pc_region_map
        ),
    }


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


def test_barrier_topology_recipe_declares_pairs_regions_and_repeats(tmp_path):
    recipe = _barrier_recipe()

    validate_mechanism_recipe(recipe)
    ncu_named_eligible_warps = replace(
        recipe,
        aggregate_metrics=tuple(
            "smsp__warps_eligible.avg.per_cycle_active"
            if metric == "smsp__eligible_warps_per_scheduler.avg"
            else metric
            for metric in recipe.aggregate_metrics
        ),
    )
    validate_mechanism_recipe(ncu_named_eligible_warps)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe.to_dict()))
    assert load_mechanism_recipe(path) == recipe

    missing_region = replace(
        recipe,
        points=(
            replace(
                recipe.points[0],
                pc_region_map={
                    key: value
                    for key, value in BARRIER_REGIONS.items()
                    if key != "wgmma_wait"
                },
            ),
            *recipe.points[1:],
        ),
    )
    with pytest.raises(ValueError, match="missing PC regions"):
        validate_mechanism_recipe(missing_region)

    too_few_profiles = replace(
        recipe,
        pc_sampling_repeats=2,
        points=tuple(
            replace(point, pc_sampling_repeats=None) for point in recipe.points
        ),
    )
    with pytest.raises(ValueError, match="three PC-sampling repeats"):
        validate_mechanism_recipe(too_few_profiles)

    mediation_as_invariant = replace(
        recipe,
        interaction_groups=(
            replace(
                recipe.interaction_groups[0],
                invariant_counter_paths=(
                    *recipe.interaction_groups[0].invariant_counter_paths,
                    "aggregate_counters.metrics.dram__bytes_read.sum",
                ),
            ),
            *recipe.interaction_groups[1:],
        ),
    )
    with pytest.raises(ValueError, match="non_hard"):
        validate_mechanism_recipe(mediation_as_invariant)


def test_barrier_recipe_rejects_held_constant_metadata_drift():
    recipe = _barrier_recipe()
    drifting = replace(
        recipe,
        points=(
            recipe.points[0],
            replace(
                recipe.points[1],
                subject_metadata={
                    **recipe.points[1].subject_metadata,
                    "registers_per_thread": 72,
                },
            ),
            *recipe.points[2:],
        ),
    )

    with pytest.raises(ValueError, match="hard metadata differences are invalid"):
        validate_mechanism_recipe(drifting)


def test_pc_region_localization_requires_support_and_repeat_agreement():
    point = _barrier_recipe().points[0]
    bundle = _barrier_bundle(point)

    localization = localize_pc_regions(bundle.pc_sampling, point.pc_region_map)

    assert localization["status"] == "qualifying"
    assert localization["pooled_structural_support"] == 120.0
    assert localization["dominant_region"] == "mbarrier_poll_retry"
    assert localization["dominant_region_repeat_count"] == 3

    low_support = {
        **bundle.pc_sampling,
        "repeats": [
            {
                **row,
                "samples": [
                    {**sample, "samples": 10.0, "stalls": {"barrier": 10.0}}
                    for sample in row["samples"]
                ],
            }
            for row in bundle.pc_sampling["repeats"]
        ],
    }
    assert localize_pc_regions(low_support, point.pc_region_map)["status"] == (
        "diagnostic"
    )


def test_barrier_topology_reduction_supports_partial_order_and_pair_controls():
    recipe = _barrier_recipe()
    results = {
        point.point_id: _barrier_point_result(
            point, latency_us=20.1 if point.point_id == "eq-split" else 20.0
        )
        for point in recipe.points
    }

    report = build_barrier_topology_report(recipe, results)
    by_panel = {pair["panel"]: pair for pair in report["pairs"]}

    assert by_panel["BT-EQ"]["classification"] == "supported"
    assert by_panel["BT-EQ"]["cuda_event_timing"][
        "paired_process_count"
    ] == 7
    assert by_panel["BT-CAUSAL"]["classification"] == "supported"
    assert by_panel["BT-CAUSAL"]["interval_order"]["status"] == "supported"
    assert all(
        edge["status"]
        in {"supported", "falsified", "not_measured", "coupled_fixture"}
        for pair in report["pairs"]
        for edge in pair["causal_edges"]
    )


def test_barrier_topology_reduction_rejects_counter_drift():
    recipe = _barrier_recipe()
    results = {
        point.point_id: _barrier_point_result(
            point, counter_drift=point.point_id == "eq-split"
        )
        for point in recipe.points
    }

    report = build_barrier_topology_report(recipe, results)
    eq_pair = next(pair for pair in report["pairs"] if pair["panel"] == "BT-EQ")

    assert eq_pair["classification"] == "coupled_fixture"
    assert eq_pair["control_scorecard"]["status"] == "coupled_fixture"
    assert any(
        reason.startswith("counter_drift:")
        for reason in eq_pair["control_scorecard"]["drift_reasons"]
    )


def test_barrier_topology_mediation_counter_drift_is_non_blocking():
    recipe = _barrier_recipe()
    results = {
        point.point_id: _barrier_point_result(
            point, mediation_drift=point.point_id == "eq-split"
        )
        for point in recipe.points
    }

    report = build_barrier_topology_report(recipe, results)
    eq_pair = next(pair for pair in report["pairs"] if pair["panel"] == "BT-EQ")
    controls = eq_pair["control_scorecard"]

    assert controls["status"] == "pass"
    assert not controls["drift_reasons"]
    assert all(
        check["role"] == "hard_invariant" and check["blocking"]
        for check in controls["hard_invariant_counter_checks"]
    )
    assert any(
        check["relative_difference"] == pytest.approx(0.75)
        and not check["blocking"]
        and check["role"] == "mediation_evidence"
        for check in controls["mediation_counter_checks"]
    )


def test_barrier_topology_non_hard_diagnostics_do_not_fail_a1():
    recipe = _barrier_recipe()
    results = {
        point.point_id: _barrier_point_result(point)
        for point in recipe.points
    }
    variant = results["eq-split"]
    variant["qualification_status"] = "diagnostic"
    variant["diagnostic_reasons"] = ["instrumentation_overhead_exceeds_threshold"]
    variant["bundle"]["identity_check"] = {
        "status": "invalid",
        "reasons": [
            "pc_sampling_01_subject_metadata_mismatch:"
            "synchronization_instruction_fingerprint"
        ],
    }
    variant["bundle"]["timing"]["subject_metadata"][
        "synchronization_instruction_fingerprint"
    ]["non_sync_digest"] = "different-non-sync-digest"
    report = build_barrier_topology_report(recipe, results)
    eq_pair = next(pair for pair in report["pairs"] if pair["group_id"] == "eq-pair")
    controls = eq_pair["control_scorecard"]

    assert controls["status"] == "pass"
    assert not controls["sass_diagnostics"]["non_sync_digest_matches"]
    assert controls["sass_diagnostics"]["role"] == (
        "non_blocking_compiler_artifact_diagnostic"
    )
    assert controls["cross_lane_diagnostic_reasons"]["eq-split"]


def test_barrier_topology_cross_lane_identity_drift_blocks_a1():
    recipe = _barrier_recipe()
    results = {
        point.point_id: _barrier_point_result(point)
        for point in recipe.points
    }
    results["eq-split"]["bundle"]["identity_check"] = {
        "status": "invalid",
        "reasons": [
            "pc_sampling_02_identity_mismatch:source_sha256",
        ],
    }

    report = build_barrier_topology_report(recipe, results)
    eq_pair = next(pair for pair in report["pairs"] if pair["group_id"] == "eq-pair")
    controls = eq_pair["control_scorecard"]

    assert controls["status"] == "not_measured"
    assert eq_pair["classification"] == "not_measured"
    assert controls["missing_reasons"] == [
        "eq-split:cross_lane:pc_sampling_02_identity_mismatch:source_sha256"
    ]


def test_execute_synthetic_barrier_recipe_writes_compact_handoff(tmp_path):
    original = _barrier_recipe()
    recipe = replace(
        original,
        points=(
            replace(original.points[0], pc_sampling_repeats=4),
            *original.points[1:],
        ),
    )
    points = {point.point_id: point for point in recipe.points}
    observed_repeats = {}

    def collector(**kwargs):
        point = points[kwargs["timing_target"][-1]]
        repeat_count = kwargs["pc_sampling_repeats"]
        observed_repeats[point.point_id] = repeat_count
        bundle = _barrier_bundle(
            point,
            latency_us=20.1 if point.point_id == "eq-split" else 20.0,
            pc_repeat_count=repeat_count,
        )
        report_base = kwargs["pc_report_path"]
        reports = []
        for index in range(repeat_count):
            report_path = report_base.with_name(
                f"pc_sampling.repeat-{index + 1:02d}.ncu-rep"
            )
            report_path.write_bytes(f"synthetic-report-{index}\n".encode())
            reports.append(
                {
                    "report_path": str(report_path),
                    "report_sha256": hashlib.sha256(
                        report_path.read_bytes()
                    ).hexdigest(),
                }
            )
        return replace(
            bundle,
            pc_sampling={
                **bundle.pc_sampling,
                "provenance": {
                    "repeat_count": repeat_count,
                    "reports": reports,
                },
            },
        )

    run_dir = execute_mechanism_recipe(
        recipe,
        capabilities=NvidiaCapabilities(),
        output_root=tmp_path,
        run_id="synthetic-run",
        bundle_collector=collector,
    )
    manifest = load_mechanism_manifest(run_dir / "manifest.json")

    assert manifest["recipe"]["pc_sampling_repeats"] == 3
    assert manifest["recipe"]["points"][0]["pc_sampling_repeats"] == 4
    assert observed_repeats["eq-joint"] == 4
    assert observed_repeats["eq-split"] == 3
    assert set(manifest["compact_artifacts"]) == {
        "cuda_event_timing.csv",
        "device_intervals.csv",
        "aggregate_stalls.csv",
        "pc_samples_by_offset.csv",
        "fixture_control_scorecard.csv",
        "hardware_findings.json",
    }
    findings = json.loads((run_dir / "hardware_findings.json").read_text())
    assert {pair["classification"] for pair in findings["pairs"]} == {
        "supported"
    }
    timing_csv = (run_dir / "cuda_event_timing.csv").read_text()
    assert "process_repeat_count" in timing_csv
    assert "eq-joint" in timing_csv
    for artifact in manifest["compact_artifacts"].values():
        artifact_path = run_dir / artifact["path"]
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact["sha256"]
    assert len(manifest["artifacts"]["eq-joint"]["pc_reports"]) == 4


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
