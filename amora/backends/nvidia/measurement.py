"""Orchestration for independent CUDA timing and NCU mechanism lanes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.cuda_event_run import (
    DEFAULT_IDENTITY_FIELDS,
    CudaEventTimingResult,
    DeviceIntervalResult,
    run_command_cuda_events,
    run_command_device_intervals,
)
from amora.backends.nvidia.ncu_run import (
    NcuPcSamplingResult,
    NcuResult,
    run_command_pc_sampling,
    run_command_profiled,
)
from amora.backends.nvidia.stall_metrics import stall_reason_for_metric
from amora.backends.nvidia.stall_metrics import parse_numeric_cell


NCU_DURATION_PREFIX = "gpu__time_duration."


def _ncu_identity(result: NcuResult | NcuPcSamplingResult) -> dict[str, Any]:
    evidence = result.target_evidence or {}
    identity = evidence.get("subject_identity")
    return dict(identity) if isinstance(identity, Mapping) else {}


def _ncu_device(result: NcuResult | NcuPcSamplingResult) -> dict[str, Any]:
    evidence = result.target_evidence or {}
    device = evidence.get("device")
    return dict(device) if isinstance(device, Mapping) else {}


def _ncu_subject_metadata(result: NcuResult | NcuPcSamplingResult) -> dict[str, Any]:
    evidence = result.target_evidence or {}
    metadata = evidence.get("subject_metadata")
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _ncu_measurement_axes(result: NcuResult | NcuPcSamplingResult) -> dict[str, Any]:
    evidence = result.target_evidence or {}
    axes = evidence.get("measurement_axes")
    return dict(axes) if isinstance(axes, Mapping) else {}


def _ncu_tool_versions(result: NcuResult | NcuPcSamplingResult) -> dict[str, Any]:
    evidence = result.target_evidence or {}
    versions = evidence.get("tool_versions")
    return dict(versions) if isinstance(versions, Mapping) else {}


def _identity_check(
    *,
    timing: CudaEventTimingResult,
    device_intervals: DeviceIntervalResult | None,
    aggregate: NcuResult,
    pc_sampling: NcuPcSamplingResult | None,
    required_fields: tuple[str, ...],
    required_metadata_fields: tuple[str, ...],
    expected_measurement_axes: Mapping[str, Any] | None,
) -> dict[str, Any]:
    lane_identities = {
        "timing": dict(timing.subject_identity),
        "aggregate_counters": _ncu_identity(aggregate),
    }
    lane_devices = {
        "timing": dict(timing.device),
        "aggregate_counters": _ncu_device(aggregate),
    }
    lane_metadata = {
        "timing": dict(timing.subject_metadata),
        "aggregate_counters": _ncu_subject_metadata(aggregate),
    }
    lane_axes = {
        "timing": dict(timing.measurement_axes),
        "aggregate_counters": _ncu_measurement_axes(aggregate),
    }
    lane_tool_versions = {
        "timing": dict(timing.tool_versions),
        "aggregate_counters": _ncu_tool_versions(aggregate),
    }
    if device_intervals is not None:
        lane_identities["device_intervals"] = dict(
            device_intervals.subject_identity
        )
        lane_devices["device_intervals"] = dict(device_intervals.device)
        lane_metadata["device_intervals"] = dict(
            device_intervals.subject_metadata
        )
        lane_axes["device_intervals"] = dict(device_intervals.measurement_axes)
        lane_tool_versions["device_intervals"] = dict(
            device_intervals.tool_versions
        )
    if pc_sampling is not None:
        lane_identities["pc_sampling"] = _ncu_identity(pc_sampling)
        lane_devices["pc_sampling"] = _ncu_device(pc_sampling)
        lane_metadata["pc_sampling"] = _ncu_subject_metadata(pc_sampling)
        lane_axes["pc_sampling"] = _ncu_measurement_axes(pc_sampling)
        lane_tool_versions["pc_sampling"] = _ncu_tool_versions(pc_sampling)

    reasons = []
    canonical = timing.subject_identity
    for lane, identity in lane_identities.items():
        for field in required_fields:
            expected = canonical.get(field)
            actual = identity.get(field)
            if expected is None:
                reasons.append(f"timing_missing_identity:{field}")
            elif actual is None:
                reasons.append(f"{lane}_missing_identity:{field}")
            elif actual != expected:
                reasons.append(f"{lane}_identity_mismatch:{field}")
    expected_uuid = timing.device.get("uuid")
    for lane, device in lane_devices.items():
        actual_uuid = device.get("uuid")
        if not actual_uuid:
            reasons.append(f"{lane}_missing_device_uuid")
        elif actual_uuid != expected_uuid:
            reasons.append(f"{lane}_device_uuid_mismatch")
    canonical_metadata = timing.subject_metadata
    for lane, metadata in lane_metadata.items():
        for field in required_metadata_fields:
            expected = canonical_metadata.get(field)
            actual = metadata.get(field)
            if expected is None:
                reasons.append(f"timing_missing_subject_metadata:{field}")
            elif actual is None:
                reasons.append(f"{lane}_missing_subject_metadata:{field}")
            elif actual != expected:
                reasons.append(f"{lane}_subject_metadata_mismatch:{field}")
    expected_axes = dict(expected_measurement_axes or timing.measurement_axes)
    for lane, axes in lane_axes.items():
        if axes != expected_axes:
            reasons.append(f"{lane}_measurement_axes_mismatch")
    for lane, versions in lane_tool_versions.items():
        if versions != timing.tool_versions:
            reasons.append(f"{lane}_tool_versions_mismatch")
    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "status": "pass" if not unique_reasons else "invalid",
        "required_fields": list(required_fields),
        "subject_identity": {
            field: canonical.get(field) for field in required_fields
        },
        "gpu_uuid": expected_uuid,
        "lane_identities": lane_identities,
        "lane_devices": lane_devices,
        "subject_metadata": {
            field: canonical_metadata.get(field) for field in required_metadata_fields
        },
        "lane_subject_metadata": lane_metadata,
        "measurement_axes": expected_axes,
        "lane_measurement_axes": lane_axes,
        "lane_tool_versions": lane_tool_versions,
        "reasons": unique_reasons,
    }


def _aggregate_payload(result: NcuResult) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(result.raw_rows) != 1:
        raise ValueError(
            "aggregate NCU lane must contain exactly one filtered launch row, "
            f"got {len(result.raw_rows)}"
        )
    selected_row = result.raw_rows[0]
    selected_metrics = {
        name: value
        for name in result.metrics
        for value in [parse_numeric_cell(selected_row.get(name))]
        if value is not None
    }
    profiler_duration = {
        name: value
        for name, value in selected_metrics.items()
        if name.startswith(NCU_DURATION_PREFIX)
    }
    counters = {
        name: value
        for name, value in selected_metrics.items()
        if not name.startswith(NCU_DURATION_PREFIX)
    }
    counter_rows = [
        {
            name: value
            for name, value in row.items()
            if not name.startswith(NCU_DURATION_PREFIX)
        }
        for row in result.raw_rows
    ]
    duration_rows = [
        {
            name: value
            for name, value in row.items()
            if name.startswith(NCU_DURATION_PREFIX)
        }
        for row in result.raw_rows
    ]
    stall_values = {
        reason: value
        for name, value in counters.items()
        for reason in [stall_reason_for_metric(name)]
        if reason is not None
    }
    structural_values = {
        reason: value
        for reason, value in stall_values.items()
        if reason not in {"selected", "not_selected"}
    }
    structural_denominator = sum(structural_values.values())
    structural_histogram = (
        {
            "raw_unit": (
                "pct_per_warp_active_cycle"
                if any("_per_warp_active" in name for name in counters)
                else "ratio_per_issue_active_cycle"
            ),
            "reason_values": structural_values,
            "denominator": structural_denominator,
            "reason_fractions": {
                reason: value / structural_denominator
                for reason, value in structural_values.items()
            },
            "excluded_reasons": ["selected", "not_selected"],
        }
        if structural_denominator > 0.0
        else None
    )
    return (
        {
            "metrics": counters,
            "raw_rows": counter_rows,
            "stall_values": stall_values,
            "structural_stall_histogram": structural_histogram,
            "launch_selection": {
                "strategy": "single_filtered_launch",
                "launches_profiled": 1,
                "selected_index": 0,
            },
            "provenance": result.provenance(),
        },
        {
            "unit": "ns",
            "metrics": profiler_duration,
            "raw_rows": duration_rows,
            "role": "diagnostic_profiler_perturbation_only",
        },
    )


def _pc_payload(result: NcuPcSamplingResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    instruction_join_count = sum(sample.sass_joined for sample in result.samples)
    by_sass_opcode: dict[str, dict[str, Any]] = {}
    for sample in result.samples:
        opcode = sample.sass_opcode or sample.opcode or "unknown"
        entry = by_sass_opcode.setdefault(
            opcode, {"support_count": 0.0, "stall_counts": {}}
        )
        entry["support_count"] += sample.samples
        stall_counts = entry["stall_counts"]
        for reason, count in sample.stalls.items():
            stall_counts[reason] = stall_counts.get(reason, 0.0) + count
    return {
        "raw_unit": "pc_sample_count",
        "samples": [sample.to_dict() for sample in result.samples],
        "support_count": sum(sample.samples for sample in result.samples),
        "instruction_join_count": instruction_join_count,
        "instruction_join_fraction": (
            instruction_join_count / len(result.samples) if result.samples else None
        ),
        "source_instruction_count": sum(
            bool(sample.opcode) for sample in result.samples
        ),
        "by_sass_opcode": by_sass_opcode,
        "provenance": result.provenance(),
    }


@dataclass(frozen=True)
class CommandMeasurementBundle:
    timing: CudaEventTimingResult
    device_intervals: DeviceIntervalResult | None
    aggregate_counters: dict[str, Any]
    profiler_duration: dict[str, Any]
    pc_sampling: dict[str, Any] | None
    identity_check: dict[str, Any]
    cache_control: str
    capabilities: dict[str, Any]
    clock_control: str | None = None

    @property
    def valid_for_comparison(self) -> bool:
        return self.identity_check.get("status") == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "command_measurement_bundle",
            "latency_oracle": "cuda_events",
            "ncu_duration_role": "diagnostic_profiler_perturbation_only",
            "valid_for_comparison": self.valid_for_comparison,
            "timing": self.timing.to_dict(),
            "device_intervals": (
                self.device_intervals.to_dict()
                if self.device_intervals is not None
                else None
            ),
            "aggregate_counters": dict(self.aggregate_counters),
            "profiler_duration": dict(self.profiler_duration),
            "pc_sampling": self.pc_sampling,
            "identity_check": dict(self.identity_check),
            "ncu_cache_control": self.cache_control,
            "ncu_clock_control": self.clock_control,
            "capabilities": dict(self.capabilities),
        }


def collect_command_measurement_bundle(
    *,
    timing_target: tuple[str, ...],
    ncu_target: tuple[str, ...],
    capabilities: NvidiaCapabilities,
    aggregate_metrics: tuple[str, ...],
    kernel_name: str,
    cache_control: str,
    clock_control: str | None = None,
    repeats: int = 7,
    collect_pc_sampling: bool = True,
    device_interval_target: tuple[str, ...] | None = None,
    expected_launch_batch_size: int | None = None,
    expected_cache_protocol: str | None = None,
    expected_max_interval_overhead_percent: float | None = None,
    required_identity_fields: tuple[str, ...] = DEFAULT_IDENTITY_FIELDS,
    required_subject_metadata_fields: tuple[str, ...] = (),
    expected_measurement_axes: Mapping[str, Any] | None = None,
    launch_skip: int | None = None,
    pc_launch_skip: int | None = None,
    sampling_interval: str = "auto",
    timeout: int = 300,
    cwd: str | Path | None = None,
    environment_overrides: Mapping[str, str] | None = None,
    pc_report_path: Path | None = None,
) -> CommandMeasurementBundle:
    """Collect independent timing, interval, and NCU evidence lanes."""

    stall_families = {
        "per_warp_active"
        for metric in aggregate_metrics
        if "warp_issue_stalled_" in metric and "_per_warp_active" in metric
    } | {
        "per_issue_active"
        for metric in aggregate_metrics
        if "average_warps_issue_stalled_" in metric
        and "_per_issue_active" in metric
    }
    if len(stall_families) > 1:
        raise ValueError("aggregate_metrics mixes NCU stall metric families")
    base_environment = dict(environment_overrides or {})
    if expected_cache_protocol is not None:
        base_environment["AMORA_CACHE_PROTOCOL"] = expected_cache_protocol
    timing = run_command_cuda_events(
        timing_target,
        repeats=repeats,
        timeout=timeout,
        expected_launch_batch_size=expected_launch_batch_size,
        expected_cache_protocol=expected_cache_protocol,
        required_identity_fields=required_identity_fields,
        expected_measurement_axes=expected_measurement_axes,
        cwd=cwd,
        environment_overrides=base_environment,
    )
    intervals = (
        run_command_device_intervals(
            device_interval_target,
            timeout=timeout,
            required_identity_fields=required_identity_fields,
            expected_max_overhead_percent=expected_max_interval_overhead_percent,
            expected_measurement_axes=expected_measurement_axes,
            cwd=cwd,
            environment_overrides=base_environment,
        )
        if device_interval_target is not None
        else None
    )
    aggregate = run_command_profiled(
        ncu_target,
        capabilities=capabilities,
        metrics=aggregate_metrics,
        kernel_name=kernel_name,
        launch_skip=launch_skip,
        launch_count=1,
        timeout=timeout,
        cache_control=cache_control,
        clock_control=clock_control,
        cwd=cwd,
        environment_overrides={
            **base_environment,
            "AMORA_MEASUREMENT_LANE": "ncu_aggregate",
        },
    )
    pc_sampling = (
        run_command_pc_sampling(
            ncu_target,
            capabilities=capabilities,
            kernel_name=kernel_name,
            launch_skip=(launch_skip if pc_launch_skip is None else pc_launch_skip),
            launch_count=1,
            timeout=timeout,
            sampling_interval=sampling_interval,
            report_path=pc_report_path,
            cache_control=cache_control,
            clock_control=clock_control,
            cwd=cwd,
            environment_overrides={
                **base_environment,
                "AMORA_MEASUREMENT_LANE": "ncu_pc_sampling",
            },
        )
        if collect_pc_sampling
        else None
    )
    identity_check = _identity_check(
        timing=timing,
        device_intervals=intervals,
        aggregate=aggregate,
        pc_sampling=pc_sampling,
        required_fields=required_identity_fields,
        required_metadata_fields=required_subject_metadata_fields,
        expected_measurement_axes=expected_measurement_axes,
    )
    aggregate_payload, profiler_duration = _aggregate_payload(aggregate)
    return CommandMeasurementBundle(
        timing=timing,
        device_intervals=intervals,
        aggregate_counters=aggregate_payload,
        profiler_duration=profiler_duration,
        pc_sampling=_pc_payload(pc_sampling),
        identity_check=identity_check,
        cache_control=cache_control,
        clock_control=clock_control,
        capabilities=capabilities.to_dict(),
    )
