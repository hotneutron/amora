"""All-corner command measurement in replay-derived warp-stall cycles.

This module extends AMORA's existing arbitrary-command NCU paths. It does not
introduce another profiler framework: every aggregate and SourceCounters lane is
collected through :mod:`amora.backends.nvidia.ncu_run`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.cuda_event_run import (
    CudaEventTimingResult,
    run_command_cuda_events,
)
from amora.backends.nvidia.measurement import pool_pc_sampling_results
from amora.backends.nvidia.ncu_run import (
    NcuPcSamplingResult,
    NcuResult,
    run_command_pc_sampling,
    run_command_profiled,
)
from amora.backends.nvidia.stall_metrics import (
    STALL_REASONS,
    StallCycleMetricSelection,
    parse_numeric_cell,
    resolve_stall_cycle_metrics,
    select_stall_launch_row,
)


RAW_STALL_CYCLE_UNIT = "diagnostic_profiler_replay_warp_cycle"
STRUCTURAL_EXCLUSIONS = frozenset({"selected", "not_selected"})
REQUIRED_CONTEXT_FIELDS = (
    "operation_id",
    "contract",
    "corner_id",
    "launch_ordinal",
    "kernel_name",
    "grid",
    "workgroup",
    "compile_time_axes",
    "runtime_axes",
    "ordered_launches",
)
SOURCE_COUNTER_SELECTION_REASONS = frozenset(
    {
        "minimum",
        "alignment_boundary",
        "first_tail",
        "first_second_tile",
        "first_wave",
        "first_second_wave",
        "compile_time_corner",
        "warm_rotating_pair",
        "failed_gate",
        "dominant_reason_change",
    }
)
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_new_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _repeat_report_path(base: Path, repeat_index: int) -> Path:
    suffix = ".ncu-rep"
    name = base.name[:-len(suffix)] if base.name.endswith(suffix) else base.name
    return base.with_name(f"{name}.repeat-{repeat_index + 1:02d}{suffix}")


def _value(row: Mapping[str, object], metric: str, *, field: str) -> float:
    value = parse_numeric_cell(row.get(metric))
    if value is None or not math.isfinite(value):
        raise ValueError(f"selected NCU row has no finite {field}: {metric}")
    if value < 0.0:
        raise ValueError(f"selected NCU row has negative {field}: {metric}")
    return value


def _normalize_geometry(value: object) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        text = str(value).strip()
        for character in "()[] ":
            text = text.replace(character, "")
        parts = text.lower().replace("x", ",").split(",")
    dimensions = tuple(int(float(part)) for part in parts if str(part))
    if not dimensions:
        raise ValueError("launch geometry must not be empty")
    canonical = list(dimensions)
    while len(canonical) > 1 and canonical[-1] == 1:
        canonical.pop()
    return tuple(canonical)


def _row_geometry(
    row: Mapping[str, object],
    *,
    display_field: str,
    dimension_fields: tuple[str, str, str],
) -> tuple[int, ...] | None:
    display = row.get(display_field)
    if display is not None and str(display).strip():
        return _normalize_geometry(display)
    dimensions = [row.get(field) for field in dimension_fields]
    if any(value is None or not str(value).strip() for value in dimensions):
        return None
    return _normalize_geometry(dimensions)


def validate_launch_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact operation and ordered-launch identity from a target."""

    missing = [field for field in REQUIRED_CONTEXT_FIELDS if field not in context]
    if missing:
        raise ValueError(f"measurement context is missing fields: {missing}")
    for field_name in ("operation_id", "contract", "corner_id", "kernel_name"):
        if not isinstance(context[field_name], str) or not context[field_name]:
            raise ValueError(f"measurement context {field_name} must be a string")
    ordinal = context["launch_ordinal"]
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError("measurement context launch_ordinal must be non-negative")
    for field_name in ("compile_time_axes", "runtime_axes"):
        if not isinstance(context[field_name], Mapping):
            raise ValueError(f"measurement context {field_name} must be an object")
    ordered = context["ordered_launches"]
    if not isinstance(ordered, list) or not ordered:
        raise ValueError("measurement context ordered_launches must be non-empty")
    ordinals = []
    for launch in ordered:
        if not isinstance(launch, Mapping):
            raise ValueError("ordered launch descriptors must be objects")
        launch_ordinal = launch.get("launch_ordinal")
        kernel_name = launch.get("kernel_name")
        subject_identity = launch.get("subject_identity")
        if (
            isinstance(launch_ordinal, bool)
            or not isinstance(launch_ordinal, int)
            or launch_ordinal < 0
            or not isinstance(kernel_name, str)
            or not kernel_name
            or not isinstance(subject_identity, Mapping)
            or not subject_identity
        ):
            raise ValueError(
                "ordered launch descriptors require ordinal, kernel, and identity"
            )
        ordinals.append(launch_ordinal)
    if ordinals != list(range(len(ordered))):
        raise ValueError("ordered launch ordinals must be contiguous and zero-based")
    if ordinal >= len(ordered):
        raise ValueError("launch ordinal is outside ordered_launches")
    selected = ordered[ordinal]
    if selected.get("kernel_name") != context["kernel_name"]:
        raise ValueError("selected launch kernel does not match ordered_launches")
    return json.loads(_canonical_json(dict(context)))


@dataclass(frozen=True)
class StallCycleRecord:
    repeat_index: int
    selected_row_index: int
    metric_family: str
    input_unit: str
    normalized_ratio_unit: str
    raw_unit: str
    active_warp_cycles: float
    active_smsp_cycles: float
    eligible_warp_cycles: float
    issue_state_ratios: dict[str, float]
    issue_state_source_values: dict[str, float]
    reason_warp_stall_cycles: dict[str, float]
    structural_warp_stall_cycles: dict[str, float]
    structural_denominator_warp_cycles: float
    structural_fractions: dict[str, float]
    issue_state_ratio_sum: float
    missing_state_ratio: float
    missing_reasons: tuple[str, ...]
    capability_missing_reasons: tuple[str, ...]
    row_missing_reasons: tuple[str, ...]
    metric_names: dict[str, str]
    profiler_replay_pass_count: float | None
    total_stall_ratio: float
    conservation_status: str
    qualification_status: str
    qualification_reasons: tuple[str, ...]
    target_evidence: dict[str, Any]
    report_path: str | None
    report_sha256: str | None
    ncu_command: tuple[str, ...]
    cache_control: str | None
    clock_control: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repeat_index": self.repeat_index,
            "selected_row_index": self.selected_row_index,
            "metric_family": self.metric_family,
            "input_unit": self.input_unit,
            "normalized_ratio_unit": self.normalized_ratio_unit,
            "raw_unit": self.raw_unit,
            "active_warp_cycles": self.active_warp_cycles,
            "active_smsp_cycles": self.active_smsp_cycles,
            "eligible_warp_cycles": self.eligible_warp_cycles,
            "issue_state_ratios": dict(self.issue_state_ratios),
            "issue_state_source_values": dict(self.issue_state_source_values),
            "reason_warp_stall_cycles": dict(self.reason_warp_stall_cycles),
            "structural_warp_stall_cycles": dict(
                self.structural_warp_stall_cycles
            ),
            "structural_denominator_warp_cycles": (
                self.structural_denominator_warp_cycles
            ),
            "structural_fractions": dict(self.structural_fractions),
            "issue_state_ratio_sum": self.issue_state_ratio_sum,
            "missing_state_ratio": self.missing_state_ratio,
            "missing_reasons": list(self.missing_reasons),
            "capability_missing_reasons": list(
                self.capability_missing_reasons
            ),
            "row_missing_reasons": list(self.row_missing_reasons),
            "metric_names": dict(self.metric_names),
            "profiler_replay_pass_count": self.profiler_replay_pass_count,
            "total_stall_ratio": self.total_stall_ratio,
            "conservation_status": self.conservation_status,
            "qualification_status": self.qualification_status,
            "qualification_reasons": list(self.qualification_reasons),
            "target_evidence": dict(self.target_evidence),
            "report_path": self.report_path,
            "report_sha256": self.report_sha256,
            "ncu_command": list(self.ncu_command),
            "cache_control": self.cache_control,
            "clock_control": self.clock_control,
        }


def reduce_stall_cycle_row(
    result: NcuResult,
    selection: StallCycleMetricSelection,
    *,
    repeat_index: int,
    tolerance: float = 0.05,
) -> StallCycleRecord:
    """Select and reduce one observed NCU launch row."""

    if not selection.available or selection.family is None or selection.input_unit is None:
        raise ValueError("stall-cycle metric selection is incomplete")
    target_context = (result.target_evidence or {}).get("measurement_context") or {}
    eligible_rows: list[tuple[int, Mapping[str, object]]] = []
    for original_index, candidate in enumerate(result.raw_rows):
        kernel = candidate.get("Kernel Name")
        if not kernel or kernel != target_context.get("kernel_name"):
            continue
        mismatch = False
        for row_field, dimension_fields, context_field in (
            (
                "Grid Size",
                (
                    "launch__grid_dim_x",
                    "launch__grid_dim_y",
                    "launch__grid_dim_z",
                ),
                "grid",
            ),
            (
                "Block Size",
                (
                    "launch__block_dim_x",
                    "launch__block_dim_y",
                    "launch__block_dim_z",
                ),
                "workgroup",
            ),
        ):
            observed = _row_geometry(
                candidate,
                display_field=row_field,
                dimension_fields=dimension_fields,
            )
            expected = target_context.get(context_field)
            if expected is not None and (
                observed is None
                or observed != _normalize_geometry(expected)
            ):
                mismatch = True
                break
        if not mismatch:
            eligible_rows.append((original_index, candidate))
    if not eligible_rows:
        raise ValueError("NCU report has no row matching the exact launch context")
    required_row_metrics = (
        *selection.cycle_metrics.values(),
        *selection.reason_to_metric.values(),
    )
    complete_rows = [
        (index, row)
        for index, row in eligible_rows
        if all(
            parse_numeric_cell(row.get(metric)) is not None
            for metric in required_row_metrics
        )
    ]
    selectable_rows = complete_rows or eligible_rows
    local_index, raw_states, total = select_stall_launch_row(
        [row for _, row in selectable_rows],
        selection.reason_to_metric,
        excluded_total_reasons=STRUCTURAL_EXCLUSIONS,
    )
    if local_index is None or total is None:
        raise ValueError("NCU report has no selectable issue-state row")
    row_index = selectable_rows[local_index][0]
    row = result.raw_rows[row_index]
    active_warp_cycles = _value(
        row, selection.cycle_metrics["active_warp_cycles"], field="active warp cycles"
    )
    if active_warp_cycles <= 0.0:
        raise ValueError("active warp cycles must be positive")
    active_smsp_cycles = _value(
        row, selection.cycle_metrics["active_smsp_cycles"], field="active SMSP cycles"
    )
    eligible_warp_cycles = _value(
        row,
        selection.cycle_metrics["eligible_warp_cycles"],
        field="eligible warp cycles",
    )
    scale = 0.01 if selection.input_unit == "pct" else 1.0
    ratios = {reason: value * scale for reason, value in raw_states.items()}
    invalid = [
        reason
        for reason, value in ratios.items()
        if not math.isfinite(value) or value < 0.0 or value > 1.0 + tolerance
    ]
    cycles = {
        reason: active_warp_cycles * ratio for reason, ratio in ratios.items()
    }
    structural_cycles = {
        reason: value
        for reason, value in cycles.items()
        if reason not in STRUCTURAL_EXCLUSIONS
    }
    structural_denominator = sum(structural_cycles.values())
    structural_fractions = (
        {
            reason: value / structural_denominator
            for reason, value in structural_cycles.items()
        }
        if structural_denominator > 0.0
        else {}
    )
    ratio_sum = sum(ratios.values())
    missing_state_ratio = max(0.0, 1.0 - ratio_sum)
    row_missing_reasons = tuple(
        reason
        for reason in selection.reason_to_metric
        if reason not in raw_states
    )
    missing_reasons = tuple(
        dict.fromkeys((*selection.missing_reasons, *row_missing_reasons))
    )
    reasons = []
    if invalid:
        reasons.append(f"invalid_issue_state_ratios:{','.join(sorted(invalid))}")
    if any(value > active_warp_cycles * (1.0 + tolerance) for value in cycles.values()):
        reasons.append("reason_cycles_exceed_active_warp_cycles")
    if ratio_sum > 1.0 + tolerance:
        conservation_status = "fail_overcoverage"
        reasons.append("issue_state_ratio_sum_exceeds_tolerance")
    elif abs(ratio_sum - 1.0) <= tolerance:
        conservation_status = "pass"
    elif missing_reasons:
        conservation_status = "incomplete_missing_state_coverage_quantified"
    else:
        conservation_status = "fail_unexplained_undercoverage"
        reasons.append("issue_state_ratio_sum_below_tolerance_without_missing_states")
    if "selected" not in ratios or "not_selected" not in ratios:
        reasons.append("ready_issue_states_missing")
    if row_missing_reasons:
        reasons.append(
            "selected_row_missing_issue_states:"
            + ",".join(row_missing_reasons)
        )
    if result.report_path is None or result.report_sha256 is None:
        reasons.append("aggregate_report_not_retained")
    elif not result.report_path.is_file():
        reasons.append("aggregate_report_missing")
    elif hashlib.sha256(result.report_path.read_bytes()).hexdigest() != result.report_sha256:
        reasons.append("aggregate_report_hash_mismatch")
    qualification = "qualifying" if not reasons else "diagnostic"
    replay_count = next(
        (
            parsed
            for name, value in row.items()
            if name.startswith("profiler__replayer_passes")
            for parsed in [parse_numeric_cell(value)]
            if parsed is not None
        ),
        None,
    )
    return StallCycleRecord(
        repeat_index=repeat_index,
        selected_row_index=row_index,
        metric_family=selection.family,
        input_unit=selection.input_unit,
        normalized_ratio_unit="ratio",
        raw_unit=RAW_STALL_CYCLE_UNIT,
        active_warp_cycles=active_warp_cycles,
        active_smsp_cycles=active_smsp_cycles,
        eligible_warp_cycles=eligible_warp_cycles,
        issue_state_ratios=ratios,
        issue_state_source_values=dict(raw_states),
        reason_warp_stall_cycles=cycles,
        structural_warp_stall_cycles=structural_cycles,
        structural_denominator_warp_cycles=structural_denominator,
        structural_fractions=structural_fractions,
        issue_state_ratio_sum=ratio_sum,
        missing_state_ratio=missing_state_ratio,
        missing_reasons=missing_reasons,
        capability_missing_reasons=selection.missing_reasons,
        row_missing_reasons=row_missing_reasons,
        metric_names={
            **selection.cycle_metrics,
            **{f"reason:{reason}": metric for reason, metric in selection.reason_to_metric.items()},
        },
        profiler_replay_pass_count=replay_count,
        total_stall_ratio=sum(
            ratio
            for reason, ratio in ratios.items()
            if reason not in STRUCTURAL_EXCLUSIONS
        ),
        conservation_status=conservation_status,
        qualification_status=qualification,
        qualification_reasons=tuple(reasons),
        target_evidence=dict(result.target_evidence or {}),
        report_path=str(result.report_path) if result.report_path else None,
        report_sha256=result.report_sha256,
        ncu_command=result.command,
        cache_control=result.cache_control,
        clock_control=result.clock_control,
    )


def _signature(record: StallCycleRecord) -> str:
    return _evidence_signature(record.target_evidence)


@dataclass(frozen=True)
class StallCycleRepeatResult:
    repeats: tuple[StallCycleRecord, ...]
    selected_repeat_index: int
    identity_status: str
    identity_reasons: tuple[str, ...]

    @property
    def selected(self) -> StallCycleRecord:
        return self.repeats[self.selected_repeat_index]

    @property
    def qualification_status(self) -> str:
        if self.identity_status != "pass":
            return "diagnostic"
        if any(record.qualification_status != "qualifying" for record in self.repeats):
            return "diagnostic"
        return "qualifying"

    def to_dict(self) -> dict[str, Any]:
        return {
            "repeat_count": len(self.repeats),
            "selected_repeat_index": self.selected_repeat_index,
            "selection_strategy": "median_total_stall_observed_repeat",
            "identity_status": self.identity_status,
            "identity_reasons": list(self.identity_reasons),
            "qualification_status": self.qualification_status,
            "selected": self.selected.to_dict(),
            "repeats": [record.to_dict() for record in self.repeats],
        }


def reduce_stall_cycle_repeats(
    repeats: Sequence[StallCycleRecord],
) -> StallCycleRepeatResult:
    """Choose the median-total observed repeat while retaining all inputs."""

    if len(repeats) != 3:
        raise ValueError("stall-cycle reduction requires exactly three reports")
    signatures = {_signature(record) for record in repeats}
    identity_reasons = () if len(signatures) == 1 else ("repeat_identity_drift",)
    ordered = sorted(
        enumerate(repeats), key=lambda item: (item[1].total_stall_ratio, item[0])
    )
    selected_index = ordered[len(ordered) // 2][0]
    return StallCycleRepeatResult(
        repeats=tuple(repeats),
        selected_repeat_index=selected_index,
        identity_status="pass" if not identity_reasons else "invalid",
        identity_reasons=identity_reasons,
    )


@dataclass(frozen=True)
class AllCornerLaunchSpec:
    point_id: str
    target: tuple[str, ...]
    kernel_name: str
    measurement_axes: dict[str, Any]
    measurement_context: dict[str, Any]
    cache_control: str = "none"
    clock_control: str | None = None
    launch_skip: int | None = None
    profile_launch_count: int = 3
    timing_repeats: int = 1
    collect_source_counters: bool = False
    source_counter_selection_reasons: tuple[str, ...] = ()
    pc_launch_skip: int | None = None
    sampling_interval: str = "auto"
    ncu_kernel_filter: str | None = None
    operation_target: tuple[str, ...] | None = None
    operation_measurement_context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "target": list(self.target),
            "kernel_name": self.kernel_name,
            "measurement_axes": dict(self.measurement_axes),
            "measurement_context": dict(self.measurement_context),
            "cache_control": self.cache_control,
            "clock_control": self.clock_control,
            "launch_skip": self.launch_skip,
            "profile_launch_count": self.profile_launch_count,
            "timing_repeats": self.timing_repeats,
            "collect_source_counters": self.collect_source_counters,
            "source_counter_selection_reasons": list(
                self.source_counter_selection_reasons
            ),
            "pc_launch_skip": self.pc_launch_skip,
            "sampling_interval": self.sampling_interval,
            "ncu_kernel_filter": self.ncu_kernel_filter,
            "operation_target": (
                list(self.operation_target)
                if self.operation_target is not None
                else None
            ),
            "operation_measurement_context": (
                dict(self.operation_measurement_context)
                if self.operation_measurement_context is not None
                else None
            ),
        }


DEFAULT_CAMPAIGN_IDENTITY_FIELDS = (
    "kernel_name",
    "ttgir_sha256",
    "ptx_sha256",
    "sass_sha256",
    "cubin_sha256",
)
DEFAULT_CAMPAIGN_METADATA_FIELDS = (
    "registers_per_thread",
    "shared_memory_bytes",
    "total_warps",
    "spill_count",
)


@dataclass(frozen=True)
class AllCornerCampaign:
    campaign_id: str
    corner_registry_digest: str
    expected_corner_count: int
    expected_launch_count: int
    launches: tuple[AllCornerLaunchSpec, ...]
    aggregate_repeats: int = 3
    pc_sampling_repeats: int = 3
    required_identity_fields: tuple[str, ...] = DEFAULT_CAMPAIGN_IDENTITY_FIELDS
    required_subject_metadata_fields: tuple[str, ...] = (
        DEFAULT_CAMPAIGN_METADATA_FIELDS
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "all_corner_stall_cycle_campaign",
            "campaign_id": self.campaign_id,
            "corner_registry_digest": self.corner_registry_digest,
            "expected_corner_count": self.expected_corner_count,
            "expected_launch_count": self.expected_launch_count,
            "aggregate_repeats": self.aggregate_repeats,
            "pc_sampling_repeats": self.pc_sampling_repeats,
            "required_identity_fields": list(self.required_identity_fields),
            "required_subject_metadata_fields": list(
                self.required_subject_metadata_fields
            ),
            "launches": [launch.to_dict() for launch in self.launches],
        }


def load_all_corner_campaign(path: str | Path) -> AllCornerCampaign:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    launches = tuple(
        AllCornerLaunchSpec(
            point_id=item["point_id"],
            target=tuple(item["target"]),
            kernel_name=item["kernel_name"],
            measurement_axes=dict(item["measurement_axes"]),
            measurement_context=validate_launch_context(item["measurement_context"]),
            cache_control=item.get("cache_control", "none"),
            clock_control=item.get("clock_control"),
            launch_skip=item.get("launch_skip"),
            profile_launch_count=int(item.get("profile_launch_count", 3)),
            timing_repeats=int(item.get("timing_repeats", 1)),
            collect_source_counters=bool(item.get("collect_source_counters", False)),
            source_counter_selection_reasons=tuple(
                item.get("source_counter_selection_reasons") or ()
            ),
            pc_launch_skip=item.get("pc_launch_skip"),
            sampling_interval=str(item.get("sampling_interval", "auto")),
            ncu_kernel_filter=item.get("ncu_kernel_filter"),
            operation_target=(
                tuple(item["operation_target"])
                if item.get("operation_target") is not None
                else None
            ),
            operation_measurement_context=(
                dict(item["operation_measurement_context"])
                if item.get("operation_measurement_context") is not None
                else None
            ),
        )
        for item in data.get("launches") or ()
    )
    campaign = AllCornerCampaign(
        campaign_id=data["campaign_id"],
        corner_registry_digest=data["corner_registry_digest"],
        expected_corner_count=int(data["expected_corner_count"]),
        expected_launch_count=int(data["expected_launch_count"]),
        launches=launches,
        aggregate_repeats=int(data.get("aggregate_repeats", 3)),
        pc_sampling_repeats=int(data.get("pc_sampling_repeats", 3)),
        required_identity_fields=tuple(
            data.get("required_identity_fields")
            or DEFAULT_CAMPAIGN_IDENTITY_FIELDS
        ),
        required_subject_metadata_fields=tuple(
            data.get("required_subject_metadata_fields")
            or DEFAULT_CAMPAIGN_METADATA_FIELDS
        ),
    )
    validate_all_corner_campaign(campaign)
    return campaign


def validate_all_corner_campaign(campaign: AllCornerCampaign) -> None:
    if not _SAFE_ID.fullmatch(campaign.campaign_id):
        raise ValueError("campaign_id must be filesystem-safe")
    if (
        not campaign.campaign_id
        or not campaign.corner_registry_digest
        or not campaign.launches
    ):
        raise ValueError(
            "campaign_id, corner_registry_digest, and launches are required"
        )
    if campaign.expected_launch_count != len(campaign.launches):
        raise ValueError("campaign launch count does not match corner registry")
    if campaign.aggregate_repeats != 3 or campaign.pc_sampling_repeats != 3:
        raise ValueError("all-corner campaigns require exactly three NCU repeats")
    point_ids = [launch.point_id for launch in campaign.launches]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("all-corner point IDs must be unique")
    for launch in campaign.launches:
        if not _SAFE_ID.fullmatch(launch.point_id):
            raise ValueError("point_id must be filesystem-safe")
        if not launch.target or not launch.point_id or not launch.kernel_name:
            raise ValueError("launch point, target, and kernel are required")
        context = validate_launch_context(launch.measurement_context)
        if context["kernel_name"] != launch.kernel_name:
            raise ValueError(f"{launch.point_id} kernel differs from context")
        combined_axes = {
            **context["compile_time_axes"],
            **context["runtime_axes"],
        }
        if combined_axes != launch.measurement_axes:
            raise ValueError(f"{launch.point_id} compile/runtime axes do not match")
        cache_protocol = launch.measurement_axes.get("cache_protocol")
        if cache_protocol not in {
            "warm_reuse",
            "disjoint_rotation",
            "cold_flush",
        }:
            raise ValueError(
                f"{launch.point_id} must declare a semantic cache protocol"
            )
        clock_policy = launch.measurement_axes.get("clock_policy")
        if not isinstance(clock_policy, str) or not clock_policy:
            raise ValueError(f"{launch.point_id} must declare a clock policy")
        if launch.profile_launch_count <= 0 or launch.timing_repeats <= 0:
            raise ValueError(f"{launch.point_id} repeat counts must be positive")
        reasons = set(launch.source_counter_selection_reasons)
        if not reasons <= SOURCE_COUNTER_SELECTION_REASONS:
            raise ValueError(f"{launch.point_id} has unknown SourceCounters reasons")
        if launch.collect_source_counters != bool(reasons):
            raise ValueError(
                f"{launch.point_id} SourceCounters selection needs explicit reasons"
            )
        operation_context = launch.operation_measurement_context
        if (launch.operation_target is None) != (operation_context is None):
            raise ValueError(
                f"{launch.point_id} operation timing target and context must "
                "be supplied together"
            )
        if operation_context is not None:
            if operation_context.get("scope") != "application_operation":
                raise ValueError(
                    f"{launch.point_id} operation timing context has wrong scope"
                )
            if operation_context.get("launch_ordinal") != -1:
                raise ValueError(
                    f"{launch.point_id} operation timing context must use "
                    "launch_ordinal=-1"
                )
            for field_name in ("operation_id", "contract", "corner_id"):
                if operation_context.get(field_name) != context[field_name]:
                    raise ValueError(
                        f"{launch.point_id} operation timing {field_name} drift"
                    )
            if operation_context.get("ordered_launches") != context["ordered_launches"]:
                raise ValueError(
                    f"{launch.point_id} operation timing launch graph drift"
                )
        if len(context["ordered_launches"]) > 1 and (
            not launch.ncu_kernel_filter
            or launch.launch_skip is None
            or launch.pc_launch_skip is None
        ):
            raise ValueError(
                f"{launch.point_id} multi-launch profiling requires an NCU kernel "
                "filter plus aggregate and SourceCounters launch skips"
            )
    by_operation: dict[tuple[str, str, str], list[AllCornerLaunchSpec]] = {}
    for launch in campaign.launches:
        context = launch.measurement_context
        key = (context["operation_id"], context["contract"], context["corner_id"])
        by_operation.setdefault(key, []).append(launch)
    if campaign.expected_corner_count != len(by_operation):
        raise ValueError("campaign corner count does not match corner registry")
    for key, launches in by_operation.items():
        ordered = launches[0].measurement_context["ordered_launches"]
        if any(
            launch.measurement_context["ordered_launches"] != ordered
            for launch in launches
        ):
            raise ValueError(f"operation {key} has inconsistent ordered launch descriptors")
        observed_list = [
            (
                launch.measurement_context["launch_ordinal"],
                launch.kernel_name,
            )
            for launch in launches
        ]
        if len(observed_list) != len(set(observed_list)):
            raise ValueError(f"operation {key} has duplicate launch ordinals")
        observed = set(observed_list)
        expected = {
            (descriptor["launch_ordinal"], descriptor["kernel_name"])
            for descriptor in ordered
        }
        if observed != expected:
            raise ValueError(
                f"operation {key} does not contain every ordered launch"
            )
        operation_specs = {
            _canonical_json(
                {
                    "target": launch.operation_target,
                    "context": launch.operation_measurement_context,
                    "timing_repeats": launch.timing_repeats,
                }
            )
            for launch in launches
        }
        if len(operation_specs) != 1:
            raise ValueError(
                f"operation {key} has inconsistent operation timing specs"
            )


def _evidence_signature(evidence: Mapping[str, Any]) -> str:
    return _canonical_json(
        {
            "identity": evidence.get("subject_identity"),
            "device": (evidence.get("device") or {}).get("uuid"),
            "metadata": evidence.get("subject_metadata"),
            "axes": evidence.get("measurement_axes"),
            "context": evidence.get("measurement_context"),
            "tools": evidence.get("tool_versions"),
        }
    )


def _operation_timing_identity_reasons(
    timing: CudaEventTimingResult,
    aggregate: StallCycleRecord,
    spec: AllCornerLaunchSpec,
    *,
    required_subject_metadata_fields: tuple[str, ...],
) -> list[str]:
    evidence = aggregate.target_evidence
    context = spec.measurement_context
    ordinal = context["launch_ordinal"]
    ordered = timing.measurement_context.get("ordered_launches") or ()
    reasons: list[str] = []
    if ordinal >= len(ordered):
        return ["operation_timing_missing_launch_descriptor"]
    descriptor = ordered[ordinal]
    if not isinstance(descriptor, Mapping):
        return ["operation_timing_invalid_launch_descriptor"]
    if descriptor.get("subject_identity") != evidence.get("subject_identity"):
        reasons.append("timing_aggregate_launch_identity_drift")
    if timing.device.get("uuid") != (evidence.get("device") or {}).get("uuid"):
        reasons.append("timing_aggregate_gpu_drift")
    if timing.measurement_axes != evidence.get("measurement_axes"):
        reasons.append("timing_aggregate_axes_drift")
    if timing.tool_versions != evidence.get("tool_versions"):
        reasons.append("timing_aggregate_tools_drift")
    launch_metadata = timing.subject_metadata.get("ordered_launches") or ()
    selected_metadata: Mapping[str, Any] = {}
    if ordinal < len(launch_metadata):
        candidate = launch_metadata[ordinal]
        if isinstance(candidate, Mapping):
            value = candidate.get("metadata")
            if isinstance(value, Mapping):
                selected_metadata = value
    aggregate_metadata = evidence.get("subject_metadata") or {}
    if any(
        selected_metadata.get(field_name) != aggregate_metadata.get(field_name)
        for field_name in required_subject_metadata_fields
    ):
        reasons.append("timing_aggregate_launch_metadata_drift")
    return reasons


def collect_stall_cycle_repeats(
    spec: AllCornerLaunchSpec,
    *,
    capabilities: NvidiaCapabilities,
    selection: StallCycleMetricSelection,
    report_dir: Path,
    timeout: int,
    cwd: str | Path | None,
    environment_overrides: Mapping[str, str] | None,
    required_identity_fields: tuple[str, ...] = DEFAULT_CAMPAIGN_IDENTITY_FIELDS,
    required_subject_metadata_fields: tuple[str, ...] = (
        DEFAULT_CAMPAIGN_METADATA_FIELDS
    ),
    aggregate_collector: Callable[..., NcuResult] = run_command_profiled,
) -> StallCycleRepeatResult:
    records = []
    for repeat_index in range(3):
        report_path = report_dir / f"aggregate.repeat-{repeat_index + 1:02d}.ncu-rep"
        result = aggregate_collector(
            spec.target,
            capabilities=capabilities,
            metrics=selection.metrics,
            kernel_name=spec.ncu_kernel_filter,
            launch_skip=spec.launch_skip,
            launch_count=spec.profile_launch_count,
            timeout=timeout,
            cache_control=spec.cache_control,
            clock_control=spec.clock_control,
            cwd=cwd,
            environment_overrides={
                **dict(environment_overrides or {}),
                "AMORA_MEASUREMENT_LANE": "ncu_aggregate_stall_cycles",
                "AMORA_AGGREGATE_REPEAT_INDEX": str(repeat_index),
                "AMORA_PROFILE_LAUNCH_COUNT": str(
                    spec.profile_launch_count
                ),
                "AMORA_CACHE_PROTOCOL": str(
                    spec.measurement_axes["cache_protocol"]
                ),
            },
            report_path=report_path,
        )
        evidence = result.target_evidence or {}
        if result.cache_control != spec.cache_control:
            raise ValueError(f"{spec.point_id} aggregate cache-control drift")
        if result.clock_control != spec.clock_control:
            raise ValueError(f"{spec.point_id} aggregate clock-control drift")
        if (
            result.report_path is None
            or result.report_sha256 is None
            or not result.report_path.is_file()
        ):
            raise ValueError(f"{spec.point_id} aggregate report was not retained")
        identity = evidence.get("subject_identity") or {}
        metadata = evidence.get("subject_metadata") or {}
        missing_identity = [
            name for name in required_identity_fields if not identity.get(name)
        ]
        missing_metadata = [
            name
            for name in required_subject_metadata_fields
            if name not in metadata
        ]
        if missing_identity or missing_metadata:
            raise ValueError(
                f"{spec.point_id} aggregate target is missing identity/metadata: "
                f"identity={missing_identity}, metadata={missing_metadata}"
            )
        if evidence.get("measurement_axes") != spec.measurement_axes:
            raise ValueError(f"{spec.point_id} aggregate runtime-axis drift")
        if evidence.get("measurement_context") != spec.measurement_context:
            raise ValueError(f"{spec.point_id} aggregate launch-context drift")
        records.append(
            reduce_stall_cycle_row(
                result, selection, repeat_index=repeat_index
            )
        )
    return reduce_stall_cycle_repeats(records)


def _operation_totals(launches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_operation: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for launch in launches:
        context = launch["measurement_context"]
        key = (context["operation_id"], context["contract"], context["corner_id"])
        by_operation.setdefault(key, []).append(launch)
    totals = []
    for (operation_id, contract, corner_id), rows in sorted(by_operation.items()):
        ordered = sorted(rows, key=lambda row: row["measurement_context"]["launch_ordinal"])
        reasons = sorted(
            {
                reason
                for row in ordered
                for reason in row["aggregate_stall_cycles"]["selected"][
                    "reason_warp_stall_cycles"
                ]
            }
        )
        reason_cycles = {
            reason: sum(
                row["aggregate_stall_cycles"]["selected"][
                    "reason_warp_stall_cycles"
                ].get(reason, 0.0)
                for row in ordered
            )
            for reason in reasons
        }
        structural = {
            reason: value
            for reason, value in reason_cycles.items()
            if reason not in STRUCTURAL_EXCLUSIONS
        }
        denominator = sum(structural.values())
        totals.append(
            {
                "operation_id": operation_id,
                "contract": contract,
                "corner_id": corner_id,
                "launch_count": len(ordered),
                "launch_ordinals": [
                    row["measurement_context"]["launch_ordinal"]
                    for row in ordered
                ],
                "active_warp_cycles": sum(
                    row["aggregate_stall_cycles"]["selected"]["active_warp_cycles"]
                    for row in ordered
                ),
                "reason_warp_stall_cycles": reason_cycles,
                "structural_denominator_warp_cycles": denominator,
                "structural_fractions": (
                    {reason: value / denominator for reason, value in structural.items()}
                    if denominator > 0.0
                    else {}
                ),
                "qualification_status": (
                    "qualifying"
                    if all(
                        row["qualification_status"] == "qualifying"
                        for row in ordered
                    )
                    else "diagnostic"
                ),
            }
        )
    return totals


def _reason_rows(
    launch_rows: Sequence[Mapping[str, Any]], *, repeats: bool
) -> list[dict[str, Any]]:
    rows = []
    for launch in launch_rows:
        context = launch["measurement_context"]
        aggregate = launch["aggregate_stall_cycles"]
        records = aggregate["repeats"] if repeats else [aggregate["selected"]]
        for record in records:
            for reason in STALL_REASONS:
                if reason not in record["issue_state_ratios"]:
                    continue
                rows.append(
                    {
                        "point_id": launch["point_id"],
                        "contract": context["contract"],
                        "corner_id": context["corner_id"],
                        "cache_protocol": launch["cache_protocol"],
                        "clock_policy": launch["clock_policy"],
                        "ncu_cache_control": record["cache_control"],
                        "ncu_clock_control": record["clock_control"],
                        "launch_ordinal": context["launch_ordinal"],
                        "kernel_name": context["kernel_name"],
                        "metric_family": record["metric_family"],
                        "reason": reason,
                        "reason_ratio": record["issue_state_ratios"][reason],
                        "source_metric_value": record["issue_state_source_values"][reason],
                        "source_metric_unit": record["input_unit"],
                        "ratio_unit": record["normalized_ratio_unit"],
                        "raw_cycle_unit": record["raw_unit"],
                        "reason_metric": record["metric_names"].get(
                            f"reason:{reason}"
                        ),
                        "active_warp_cycles": record["active_warp_cycles"],
                        "active_smsp_cycles": record["active_smsp_cycles"],
                        "eligible_warp_cycles": record["eligible_warp_cycles"],
                        "reason_warp_stall_cycles": record["reason_warp_stall_cycles"][reason],
                        "structural_denominator_warp_cycles": record["structural_denominator_warp_cycles"],
                        "structural_fraction": record["structural_fractions"].get(reason),
                        "excluded_ready_state": reason in STRUCTURAL_EXCLUSIONS,
                        "selected_row_index": record["selected_row_index"],
                        "selected_repeat_index": aggregate[
                            "selected_repeat_index"
                        ],
                        "repeat_index": record["repeat_index"],
                        "profiler_replay_pass_count": record[
                            "profiler_replay_pass_count"
                        ],
                        "issue_state_ratio_sum": record[
                            "issue_state_ratio_sum"
                        ],
                        "missing_state_ratio": record["missing_state_ratio"],
                        "missing_reasons": json.dumps(record["missing_reasons"]),
                        "dimensional_conservation_status": record["conservation_status"],
                        "qualification_status": launch["qualification_status"],
                    }
                )
    return rows


AGGREGATE_FIELDS = (
    "point_id", "contract", "corner_id", "cache_protocol",
    "clock_policy", "ncu_cache_control", "ncu_clock_control",
    "launch_ordinal", "kernel_name", "metric_family", "reason",
    "reason_ratio", "source_metric_value", "source_metric_unit",
    "ratio_unit", "raw_cycle_unit", "reason_metric",
    "active_warp_cycles", "active_smsp_cycles", "eligible_warp_cycles",
    "reason_warp_stall_cycles",
    "structural_denominator_warp_cycles", "structural_fraction",
    "excluded_ready_state", "selected_row_index", "selected_repeat_index",
    "repeat_index", "profiler_replay_pass_count", "issue_state_ratio_sum",
    "missing_state_ratio", "missing_reasons",
    "dimensional_conservation_status", "qualification_status",
)


def execute_all_corner_campaign(
    campaign: AllCornerCampaign,
    *,
    capabilities: NvidiaCapabilities,
    output_root: str | Path,
    run_id: str,
    timeout: int = 300,
    cwd: str | Path | None = None,
    environment_overrides: Mapping[str, str] | None = None,
    timing_collector: Callable[..., CudaEventTimingResult] = run_command_cuda_events,
    aggregate_collector: Callable[..., NcuResult] = run_command_profiled,
    pc_collector: Callable[..., NcuPcSamplingResult] = run_command_pc_sampling,
) -> Path:
    validate_all_corner_campaign(campaign)
    if not _SAFE_ID.fullmatch(run_id):
        raise ValueError("run_id must be filesystem-safe")
    selection = resolve_stall_cycle_metrics(capabilities.ncu_metrics)
    if not selection.available:
        raise ValueError(f"stall-cycle capability unavailable: {selection.to_dict()}")
    run_dir = Path(output_root) / campaign.campaign_id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    frozen_path = run_dir / "campaign.json"
    _write_new_json(frozen_path, campaign.to_dict())
    launch_rows: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    operation_timings: dict[tuple[str, str, str], CudaEventTimingResult] = {}
    for spec in campaign.launches:
        point_dir = run_dir / "points" / spec.point_id
        point_dir.mkdir(parents=True, exist_ok=False)
        operation_key = (
            spec.measurement_context["operation_id"],
            spec.measurement_context["contract"],
            spec.measurement_context["corner_id"],
        )
        operation_timing = spec.operation_target is not None
        if operation_timing and operation_key in operation_timings:
            timing = operation_timings[operation_key]
        else:
            timing = timing_collector(
                spec.operation_target or spec.target,
                repeats=spec.timing_repeats,
                timeout=timeout,
                expected_cache_protocol=spec.measurement_axes.get(
                    "cache_protocol"
                ),
                required_identity_fields=campaign.required_identity_fields,
                expected_measurement_axes=spec.measurement_axes,
                expected_measurement_context=(
                    spec.operation_measurement_context
                    if operation_timing
                    else spec.measurement_context
                ),
                cwd=cwd,
                environment_overrides=environment_overrides,
            )
            if operation_timing:
                operation_timings[operation_key] = timing
        aggregate = collect_stall_cycle_repeats(
            spec,
            capabilities=capabilities,
            selection=selection,
            report_dir=point_dir,
            timeout=timeout,
            cwd=cwd,
            environment_overrides=environment_overrides,
            required_identity_fields=campaign.required_identity_fields,
            required_subject_metadata_fields=(
                campaign.required_subject_metadata_fields
            ),
            aggregate_collector=aggregate_collector,
        )
        timing_evidence = {
            "subject_identity": timing.subject_identity,
            "device": timing.device,
            "subject_metadata": timing.subject_metadata,
            "measurement_axes": timing.measurement_axes,
            "measurement_context": timing.measurement_context,
            "tool_versions": timing.tool_versions,
        }
        identity_reasons = list(aggregate.identity_reasons)
        if timing.clock_policy != spec.measurement_axes["clock_policy"]:
            identity_reasons.append("timing_clock_policy_drift")
        if operation_timing:
            identity_reasons.extend(
                _operation_timing_identity_reasons(
                    timing,
                    aggregate.selected,
                    spec,
                    required_subject_metadata_fields=(
                        campaign.required_subject_metadata_fields
                    ),
                )
            )
        elif _evidence_signature(timing_evidence) != _signature(aggregate.selected):
            identity_reasons.append("timing_aggregate_identity_drift")
        selected_reason = (
            max(
                aggregate.selected.structural_fractions.items(),
                key=lambda item: item[1],
            )[0]
            if aggregate.selected.structural_fractions
            else None
        )
        previous_dominant_reason = spec.measurement_context.get(
            "previous_dominant_reason"
        )
        auto_selection_reasons = set(spec.source_counter_selection_reasons)
        if aggregate.qualification_status != "qualifying":
            auto_selection_reasons.add("failed_gate")
        if spec.measurement_context.get("latency_gate_status") == "fail":
            auto_selection_reasons.add("failed_gate")
        if (
            previous_dominant_reason is not None
            and selected_reason != previous_dominant_reason
        ):
            auto_selection_reasons.add("dominant_reason_change")
        collect_pc = spec.collect_source_counters or bool(auto_selection_reasons)
        pc_payload = None
        if collect_pc:
            pc_results = []
            for repeat_index in range(campaign.pc_sampling_repeats):
                report_path = point_dir / f"pc_sampling.repeat-{repeat_index + 1:02d}.ncu-rep"
                result = pc_collector(
                    spec.target,
                    capabilities=capabilities,
                    kernel_name=spec.ncu_kernel_filter,
                    launch_skip=(
                        spec.launch_skip
                        if spec.pc_launch_skip is None
                        else spec.pc_launch_skip
                    ),
                    launch_count=1,
                    timeout=timeout,
                    sampling_interval=spec.sampling_interval,
                    report_path=report_path,
                    cache_control=spec.cache_control,
                    clock_control=spec.clock_control,
                    cwd=cwd,
                    environment_overrides={
                        **dict(environment_overrides or {}),
                        "AMORA_MEASUREMENT_LANE": "ncu_pc_sampling",
                        "AMORA_PC_SAMPLING_REPEAT_INDEX": str(repeat_index),
                        "AMORA_PROFILE_LAUNCH_COUNT": "1",
                        "AMORA_CACHE_PROTOCOL": str(
                            spec.measurement_axes["cache_protocol"]
                        ),
                    },
                )
                if (
                    result.report_path is None
                    or result.report_sha256 is None
                    or not result.report_path.is_file()
                ):
                    raise ValueError(
                        f"{spec.point_id} SourceCounters report was not retained"
                    )
                if result.cache_control != spec.cache_control:
                    identity_reasons.append(
                        f"pc_repeat_{repeat_index + 1:02d}_cache_control_drift"
                    )
                if result.clock_control != spec.clock_control:
                    identity_reasons.append(
                        f"pc_repeat_{repeat_index + 1:02d}_clock_control_drift"
                    )
                if _evidence_signature(result.target_evidence or {}) != _signature(
                    aggregate.selected
                ):
                    identity_reasons.append(
                        f"pc_repeat_{repeat_index + 1:02d}_identity_drift"
                    )
                pc_results.append(result)
            pc_payload = pool_pc_sampling_results(tuple(pc_results))
        qualification = (
            "qualifying"
            if aggregate.qualification_status == "qualifying"
            and not identity_reasons
            and (
                pc_payload is None
                or pc_payload.get("instruction_join_fraction") == 1.0
            )
            else "diagnostic"
        )
        payload = {
            "schema_version": 1,
            "kind": "all_corner_launch_stall_cycles",
            "point_id": spec.point_id,
            "cache_protocol": spec.measurement_axes.get("cache_protocol"),
            "clock_policy": spec.measurement_axes.get("clock_policy"),
            "measurement_axes": spec.measurement_axes,
            "measurement_context": spec.measurement_context,
            "timing": timing.to_dict(),
            "aggregate_stall_cycles": aggregate.to_dict(),
            "pc_sampling": pc_payload,
            "source_counter_selection_reasons": list(
                sorted(auto_selection_reasons)
            ),
            "identity_reasons": list(dict.fromkeys(identity_reasons)),
            "qualification_status": qualification,
        }
        path = point_dir / "stall_cycles.json"
        _write_new_json(path, payload)
        launch_rows.append(payload)
        artifacts[spec.point_id] = {
            "path": str(path.relative_to(run_dir)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "qualification_status": qualification,
        }
        aggregate_reports: list[dict[str, Any]] = []
        for record in aggregate.repeats:
            if record.report_path and record.report_sha256:
                aggregate_report_path = Path(record.report_path)
                if aggregate_report_path.is_file():
                    aggregate_reports.append(
                        {
                            "path": str(aggregate_report_path.relative_to(run_dir)),
                            "sha256": record.report_sha256,
                            "repeat_index": record.repeat_index,
                        }
                    )
        if aggregate_reports:
            artifacts[spec.point_id]["aggregate_reports"] = aggregate_reports
        if pc_payload is not None:
            pc_reports: list[dict[str, str]] = []
            report_entries = (pc_payload.get("provenance") or {}).get("reports") or ()
            for report_entry in report_entries:
                if not isinstance(report_entry, Mapping):
                    continue
                pc_report_path_value = report_entry.get("report_path")
                report_hash = report_entry.get("report_sha256")
                if isinstance(pc_report_path_value, str) and isinstance(report_hash, str):
                    candidate = Path(pc_report_path_value)
                    if candidate.is_file():
                        pc_reports.append(
                            {
                                "path": str(candidate.relative_to(run_dir)),
                                "sha256": report_hash,
                            }
                        )
            if pc_reports:
                artifacts[spec.point_id]["pc_reports"] = pc_reports
    operation_totals = _operation_totals(launch_rows)
    summary_rows = _reason_rows(launch_rows, repeats=False)
    repeat_rows = _reason_rows(launch_rows, repeats=True)
    summary_path = run_dir / "aggregate_stall_cycles.csv"
    repeats_path = run_dir / "aggregate_stall_cycle_repeats.csv"
    findings_path = run_dir / "aggregate_stall_cycle_findings.json"
    _write_new_csv(summary_path, summary_rows, AGGREGATE_FIELDS)
    _write_new_csv(repeats_path, repeat_rows, AGGREGATE_FIELDS)
    findings = {
        "schema_version": 1,
        "kind": "all_corner_stall_cycle_findings",
        "corner_registry_digest": campaign.corner_registry_digest,
        "expected_corner_count": campaign.expected_corner_count,
        "expected_launch_count": campaign.expected_launch_count,
        "raw_unit": RAW_STALL_CYCLE_UNIT,
        "latency_oracle": "cuda_events",
        "ncu_duration_role": "diagnostic_profiler_perturbation_only",
        "capability": selection.to_dict(),
        "launch_count": len(launch_rows),
        "qualifying_launch_count": sum(
            row["qualification_status"] == "qualifying" for row in launch_rows
        ),
        "operation_totals": operation_totals,
        "launches": launch_rows,
    }
    _write_new_json(findings_path, findings)
    compact_paths = [summary_path, repeats_path, findings_path]
    pc_rows: list[dict[str, Any]] = []
    for launch in launch_rows:
        context_value = launch["measurement_context"]
        if not isinstance(context_value, Mapping):
            continue
        context = context_value
        pc_value = launch.get("pc_sampling")
        pc = pc_value if isinstance(pc_value, Mapping) else {}
        for row in pc.get("by_offset") or ():
            if not isinstance(row, Mapping):
                continue
            pc_rows.append(
                {
                    "point_id": launch["point_id"],
                    "contract": context["contract"],
                    "corner_id": context["corner_id"],
                    "launch_ordinal": context["launch_ordinal"],
                    "kernel_name": context["kernel_name"],
                    "function": row["function"],
                    "pc_offset": row["pc_offset"],
                    "sass_opcode": row["sass_opcode"],
                    "sass_instruction": row["sass_instruction"],
                    "support_count": row["support_count"],
                    "repeat_support_counts": json.dumps(row["repeat_support_counts"]),
                    "stall_counts": json.dumps(row["stall_counts"], sort_keys=True),
                    "sass_joined_in_all_repeats": row["sass_joined_in_all_repeats"],
                }
            )
    if pc_rows:
        pc_path = run_dir / "pc_samples_by_offset.csv"
        _write_new_csv(pc_path, pc_rows, tuple(pc_rows[0]))
        compact_paths.append(pc_path)
    compact = {
        path.name: {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in compact_paths
    }
    manifest = {
        "schema_version": 1,
        "kind": "all_corner_stall_cycle_run",
        "campaign_id": campaign.campaign_id,
        "run_id": run_id,
        "corner_registry_digest": campaign.corner_registry_digest,
        "expected_corner_count": campaign.expected_corner_count,
        "expected_launch_count": campaign.expected_launch_count,
        "campaign_digest": _digest(campaign.to_dict()),
        "frozen_campaign": {
            "path": frozen_path.name,
            "sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        },
        "artifacts": artifacts,
        "compact_artifacts": compact,
        "capabilities": capabilities.to_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest["run_digest"] = _digest(manifest)
    _write_new_json(run_dir / "manifest.json", manifest)
    return run_dir


def validate_all_corner_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("run_digest")
    payload = dict(manifest)
    payload.pop("run_digest", None)
    if not isinstance(expected, str) or _digest(payload) != expected:
        raise ValueError("all-corner manifest digest mismatch")
    entries = [manifest["frozen_campaign"]]
    for point in manifest.get("artifacts", {}).values():
        entries.append(point)
        entries.extend(point.get("aggregate_reports") or ())
        entries.extend(point.get("pc_reports") or ())
    entries.extend(manifest.get("compact_artifacts", {}).values())
    for entry in entries:
        artifact = manifest_path.parent / entry["path"]
        if not artifact.is_file():
            raise ValueError(f"missing all-corner artifact: {entry['path']}")
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"all-corner artifact hash mismatch: {entry['path']}")
    return manifest
