"""Physical-mechanism recipes and immutable measurement artifacts.

The structures in this module preserve interventions and measured causal links.
They deliberately do not fit elapsed time, problem size, or raw working-set size
as direct latency corrections. Model interpretation remains with the consumer.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.measurement import (
    CommandMeasurementBundle,
    collect_command_measurement_bundle,
)


MECHANISMS = frozenset({"wgmma_fixed_completion", "tma_route_offered_load"})
SPLITS = frozenset({"calibration", "held_out"})
CAUSAL_EDGES = (
    ("footprint_occupancy", "producer_eligibility"),
    ("producer_eligibility", "tma_issue"),
    ("tma_issue", "memory_queue_partition"),
    ("memory_queue_partition", "memory_route"),
    ("memory_route", "tma_completion"),
    ("tma_completion", "barrier_release"),
    ("barrier_release", "wgmma_issue"),
    ("wgmma_issue", "wgmma_completion"),
    ("wgmma_completion", "stage_buffer_release"),
    ("stage_buffer_release", "tma_issue"),
)
REQUIRED_AXES = {
    "wgmma_fixed_completion": frozenset(
        {
            "wgmma_repeats",
            "independent_register_work_gap_ns",
            "outstanding_wgmma_groups",
            "instruction_shape",
            "cta_load",
        }
    ),
    "tma_route_offered_load": frozenset(
        {
            "request_bytes",
            "iteration_count",
            "compiled_queue_capacity",
            "rotating_working_set_bytes",
            "cache_protocol",
            "cta_concurrency",
            "address_partition_mapping",
            "per_cta_footprint_bytes",
        }
    ),
}
REQUIRED_EVIDENCE_NODES = {
    "wgmma_fixed_completion": frozenset(
        {
            "footprint_occupancy",
            "wgmma_issue",
            "wgmma_exposed_wait",
            "wgmma_completion",
            "stage_buffer_release",
            "total_latency",
        }
    ),
    "tma_route_offered_load": frozenset(
        {
            "footprint_occupancy",
            "producer_eligibility",
            "tma_issue",
            "memory_queue_partition",
            "memory_route",
            "tma_completion",
            "barrier_release",
            "stage_buffer_release",
            "total_latency",
        }
    ),
}
DIRECT_EVIDENCE_PREFIXES = {
    "wgmma_fixed_completion": {
        "wgmma_issue": (
            "bundle.aggregate_counters.",
            "bundle.pc_sampling.",
        ),
        "wgmma_exposed_wait": ("bundle.device_intervals.",),
        "wgmma_completion": ("bundle.device_intervals.",),
        "stage_buffer_release": (
            "bundle.device_intervals.",
            "bundle.aggregate_counters.",
            "bundle.pc_sampling.",
        ),
        "total_latency": ("bundle.timing.",),
    },
    "tma_route_offered_load": {
        "footprint_occupancy": ("bundle.aggregate_counters.",),
        "producer_eligibility": ("bundle.aggregate_counters.",),
        "tma_issue": (
            "bundle.aggregate_counters.",
            "bundle.pc_sampling.",
        ),
        "memory_queue_partition": (
            "bundle.aggregate_counters.",
            "bundle.pc_sampling.",
        ),
        "memory_route": ("bundle.aggregate_counters.",),
        "tma_completion": ("bundle.device_intervals.",),
        "barrier_release": ("bundle.device_intervals.",),
        "stage_buffer_release": (
            "bundle.device_intervals.",
            "bundle.aggregate_counters.",
            "bundle.pc_sampling.",
        ),
        "total_latency": ("bundle.timing.",),
    },
}
REQUIRED_SUBJECT_METADATA = frozenset(
    {"registers_per_thread", "shared_memory_bytes", "spill_count"}
)
WGMMA_REQUIRED_STALL_REASONS = frozenset(
    {"wait", "math_pipe_throttle", "barrier", "warpgroup_arrive"}
)
DEFAULT_IDENTITY_FIELDS = (
    "ttgir_sha256",
    "ptx_sha256",
    "sass_sha256",
    "cubin_sha256",
)
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
PHYSICAL_TESTS = {
    "wgmma_fixed_completion": {
        "equation": "exposed_wait(D) = max(0, L_complete - D)",
        "discriminator": (
            "If inferred completion changes with CTA load, classify it as a "
            "shared-resource or scheduling interaction."
        ),
    },
    "tma_route_offered_load": {
        "request_latency": "L_request = sum_r p_r * L_r + transaction_overhead",
        "memory_service": "S_memory = max_r(bytes_r / bandwidth_r)",
        "initiation_interval": "II_memory = max(S_issue, S_memory, L_request / Q)",
        "total_time": (
            "T(N) = T_fill + (N - 1) * max(II_compute, II_memory) + T_drain"
        ),
        "offered_load": (
            "rho = offered_bytes_per_cycle / sustainable_bytes_per_cycle; "
            "L_effective = L_request + Q_delay(rho)"
        ),
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MechanismPoint:
    point_id: str
    split: str
    axes: dict[str, Any]
    timing_target: tuple[str, ...]
    ncu_target: tuple[str, ...]
    evidence_bindings: dict[str, tuple[str, ...]]
    subject_metadata: dict[str, Any]
    device_interval_target: tuple[str, ...] | None = None
    expected_launch_batch_size: int | None = None
    expected_cache_protocol: str | None = None
    expected_max_interval_overhead_percent: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "split": self.split,
            "axes": dict(self.axes),
            "timing_target": list(self.timing_target),
            "ncu_target": list(self.ncu_target),
            "device_interval_target": (
                list(self.device_interval_target)
                if self.device_interval_target is not None
                else None
            ),
            "evidence_bindings": {
                node: list(paths) for node, paths in self.evidence_bindings.items()
            },
            "subject_metadata": dict(self.subject_metadata),
            "expected_launch_batch_size": self.expected_launch_batch_size,
            "expected_cache_protocol": self.expected_cache_protocol,
            "expected_max_interval_overhead_percent": (
                self.expected_max_interval_overhead_percent
            ),
        }


@dataclass(frozen=True)
class InteractionGroup:
    group_id: str
    point_ids: tuple[str, ...]
    varied_axis: str
    intervention_node: str
    change_threshold_fraction: float = 0.05
    requires_same_subject_identity: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "point_ids": list(self.point_ids),
            "varied_axis": self.varied_axis,
            "intervention_node": self.intervention_node,
            "change_threshold_fraction": self.change_threshold_fraction,
            "requires_same_subject_identity": self.requires_same_subject_identity,
        }


@dataclass(frozen=True)
class MechanismRecipe:
    recipe_id: str
    mechanism: str
    kernel_name: str
    aggregate_metrics: tuple[str, ...]
    points: tuple[MechanismPoint, ...]
    interaction_groups: tuple[InteractionGroup, ...]
    cache_control: str = "none"
    clock_control: str | None = None
    repeats: int = 7
    collect_pc_sampling: bool = True
    randomization_seed: int = 0
    launch_skip: int | None = None
    pc_launch_skip: int | None = None
    sampling_interval: str = "auto"
    required_identity_fields: tuple[str, ...] = DEFAULT_IDENTITY_FIELDS
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "physical_mechanism_recipe",
            "recipe_id": self.recipe_id,
            "mechanism": self.mechanism,
            "kernel_name": self.kernel_name,
            "aggregate_metrics": list(self.aggregate_metrics),
            "points": [point.to_dict() for point in self.points],
            "interaction_groups": [
                group.to_dict() for group in self.interaction_groups
            ],
            "cache_control": self.cache_control,
            "clock_control": self.clock_control,
            "repeats": self.repeats,
            "collect_pc_sampling": self.collect_pc_sampling,
            "randomization_seed": self.randomization_seed,
            "launch_skip": self.launch_skip,
            "pc_launch_skip": self.pc_launch_skip,
            "sampling_interval": self.sampling_interval,
            "required_identity_fields": list(self.required_identity_fields),
            "provenance": dict(self.provenance),
            "modeling_guard": {
                "direct_regressors_forbidden": [
                    "elapsed_duration",
                    "K",
                    "raw_working_set_bytes",
                ],
                "required_interpretation": "coupled_resource_and_dependency_graph",
            },
            "physical_test": dict(PHYSICAL_TESTS[self.mechanism]),
        }


def load_mechanism_recipe(path: str | Path) -> MechanismRecipe:
    """Load and validate a command-target mechanism recipe from JSON."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("mechanism recipe must be an object")
    points = tuple(
        MechanismPoint(
            point_id=item["point_id"],
            split=item["split"],
            axes=dict(item.get("axes") or {}),
            timing_target=tuple(item.get("timing_target") or ()),
            ncu_target=tuple(item.get("ncu_target") or ()),
            device_interval_target=(
                tuple(item["device_interval_target"])
                if item.get("device_interval_target") is not None
                else None
            ),
            evidence_bindings={
                node: tuple(paths)
                for node, paths in (item.get("evidence_bindings") or {}).items()
            },
            subject_metadata=dict(item.get("subject_metadata") or {}),
            expected_launch_batch_size=item.get("expected_launch_batch_size"),
            expected_cache_protocol=item.get("expected_cache_protocol"),
            expected_max_interval_overhead_percent=item.get(
                "expected_max_interval_overhead_percent"
            ),
        )
        for item in data.get("points") or ()
    )
    groups = tuple(
        InteractionGroup(
            group_id=item["group_id"],
            point_ids=tuple(item.get("point_ids") or ()),
            varied_axis=item["varied_axis"],
            intervention_node=item["intervention_node"],
            change_threshold_fraction=float(
                item.get("change_threshold_fraction", 0.05)
            ),
            requires_same_subject_identity=bool(
                item.get("requires_same_subject_identity", True)
            ),
        )
        for item in data.get("interaction_groups") or ()
    )
    recipe = MechanismRecipe(
        recipe_id=data["recipe_id"],
        mechanism=data["mechanism"],
        kernel_name=data["kernel_name"],
        aggregate_metrics=tuple(data.get("aggregate_metrics") or ()),
        points=points,
        interaction_groups=groups,
        cache_control=data.get("cache_control", "none"),
        clock_control=data.get("clock_control"),
        repeats=int(data.get("repeats", 7)),
        collect_pc_sampling=bool(data.get("collect_pc_sampling", True)),
        randomization_seed=int(data.get("randomization_seed", 0)),
        launch_skip=data.get("launch_skip"),
        pc_launch_skip=data.get("pc_launch_skip"),
        sampling_interval=str(data.get("sampling_interval", "auto")),
        required_identity_fields=tuple(
            data.get("required_identity_fields") or DEFAULT_IDENTITY_FIELDS
        ),
        provenance=dict(data.get("provenance") or {}),
    )
    validate_mechanism_recipe(recipe)
    return recipe


def validate_mechanism_recipe(recipe: MechanismRecipe) -> None:
    """Fail closed when a recipe cannot support the requested causal test."""

    if not _SAFE_ID.fullmatch(recipe.recipe_id):
        raise ValueError("recipe_id must be filesystem-safe")
    if recipe.mechanism not in MECHANISMS:
        raise ValueError(f"unsupported mechanism: {recipe.mechanism}")
    if not recipe.kernel_name or not recipe.aggregate_metrics:
        raise ValueError("kernel_name and aggregate_metrics are required")
    if recipe.cache_control not in {"all", "none"}:
        raise ValueError("cache_control must be 'all' or 'none'")
    if recipe.clock_control not in {None, "base", "boost", "none"}:
        raise ValueError("clock_control must be 'base', 'boost', 'none', or null")
    if recipe.repeats <= 0:
        raise ValueError("repeats must be positive")
    if isinstance(recipe.randomization_seed, bool) or not isinstance(
        recipe.randomization_seed, int
    ):
        raise ValueError("randomization_seed must be an integer")
    for name, value in (
        ("launch_skip", recipe.launch_skip),
        ("pc_launch_skip", recipe.pc_launch_skip),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer or null")
    if not recipe.points:
        raise ValueError("recipe must contain points")
    point_ids = [point.point_id for point in recipe.points]
    if len(set(point_ids)) != len(point_ids):
        raise ValueError("point IDs must be unique")
    required_axes = REQUIRED_AXES[recipe.mechanism]
    required_nodes = REQUIRED_EVIDENCE_NODES[recipe.mechanism]
    for point in recipe.points:
        if not _SAFE_ID.fullmatch(point.point_id):
            raise ValueError(f"point_id must be filesystem-safe: {point.point_id!r}")
        if point.split not in SPLITS:
            raise ValueError(f"unsupported split for {point.point_id}: {point.split}")
        missing_axes = sorted(required_axes - set(point.axes))
        if missing_axes:
            raise ValueError(f"{point.point_id} is missing axes: {missing_axes}")
        missing_nodes = sorted(required_nodes - set(point.evidence_bindings))
        if missing_nodes:
            raise ValueError(
                f"{point.point_id} is missing evidence nodes: {missing_nodes}"
            )
        for node, prefixes in DIRECT_EVIDENCE_PREFIXES[recipe.mechanism].items():
            paths = point.evidence_bindings[node]
            if not any(path.startswith(prefixes) for path in paths):
                raise ValueError(
                    f"{point.point_id} node {node} lacks direct evidence binding"
                )
        missing_metadata = sorted(REQUIRED_SUBJECT_METADATA - set(point.subject_metadata))
        if missing_metadata:
            raise ValueError(
                f"{point.point_id} is missing subject metadata: {missing_metadata}"
            )
        for name in REQUIRED_SUBJECT_METADATA:
            value = point.subject_metadata[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(
                    f"{point.point_id} subject metadata {name} must be non-negative"
                )
        if not point.timing_target or not point.ncu_target:
            raise ValueError(f"{point.point_id} has an empty target command")
        if point.expected_launch_batch_size is not None and (
            isinstance(point.expected_launch_batch_size, bool)
            or not isinstance(point.expected_launch_batch_size, int)
            or point.expected_launch_batch_size <= 0
        ):
            raise ValueError(
                f"{point.point_id} expected_launch_batch_size must be positive"
            )
        if point.expected_cache_protocol is not None and (
            point.expected_cache_protocol
            not in {"warm_reuse", "disjoint_rotation", "cold_flush"}
        ):
            raise ValueError(f"{point.point_id} has an invalid cache protocol")
        declared_cache_protocol = point.expected_cache_protocol or point.axes.get(
            "cache_protocol"
        )
        if declared_cache_protocol not in {
            "warm_reuse",
            "disjoint_rotation",
            "cold_flush",
        }:
            raise ValueError(
                f"{point.point_id} must declare a semantic cache protocol"
            )
        if point.expected_max_interval_overhead_percent is not None and (
            not math.isfinite(point.expected_max_interval_overhead_percent)
            or point.expected_max_interval_overhead_percent < 0.0
        ):
            raise ValueError(f"{point.point_id} has an invalid overhead threshold")
        bound_paths = {
            path for paths in point.evidence_bindings.values() for path in paths
        }
        if recipe.collect_pc_sampling and not any(
            path.startswith("bundle.pc_sampling.") for path in bound_paths
        ):
            raise ValueError(
                f"{point.point_id} does not bind SourceCounters PC evidence"
            )
    if {point.split for point in recipe.points} != SPLITS:
        raise ValueError("recipe must predeclare calibration and held_out points")

    points = {point.point_id: point for point in recipe.points}
    for group in recipe.interaction_groups:
        if not _SAFE_ID.fullmatch(group.group_id):
            raise ValueError("interaction group ID must be filesystem-safe")
        if len(group.point_ids) < 2 or len(set(group.point_ids)) != len(group.point_ids):
            raise ValueError(f"{group.group_id} requires at least two unique points")
        if group.intervention_node not in {node for edge in CAUSAL_EDGES for node in edge}:
            raise ValueError(f"{group.group_id} has an unknown intervention node")
        if (
            not math.isfinite(group.change_threshold_fraction)
            or group.change_threshold_fraction < 0
        ):
            raise ValueError(f"{group.group_id} has an invalid change threshold")
        try:
            members = [points[point_id] for point_id in group.point_ids]
        except KeyError as exc:
            raise ValueError(f"{group.group_id} references an unknown point") from exc
        if group.varied_axis not in members[0].axes:
            raise ValueError(f"{group.group_id} varied axis is absent")
        if len({canonical_json(point.axes[group.varied_axis]) for point in members}) < 2:
            raise ValueError(f"{group.group_id} does not vary {group.varied_axis}")
        control_axes = set(members[0].axes) - {group.varied_axis}
        for axis in control_axes:
            values = {canonical_json(point.axes.get(axis)) for point in members}
            if len(values) != 1:
                raise ValueError(
                    f"{group.group_id} changes control axis {axis} with "
                    f"{group.varied_axis}"
                )
        if group.requires_same_subject_identity:
            metadata_keys = set(members[0].subject_metadata)
            if any(set(point.subject_metadata) != metadata_keys for point in members):
                raise ValueError(f"{group.group_id} subject metadata fields drift")
            for field_name in metadata_keys:
                values = {
                    canonical_json(point.subject_metadata[field_name])
                    for point in members
                }
                if len(values) != 1:
                    raise ValueError(
                        f"{group.group_id} changes subject metadata {field_name}"
                    )
    varied_axes = {group.varied_axis for group in recipe.interaction_groups}
    missing_interventions = sorted(required_axes - varied_axes)
    if missing_interventions:
        raise ValueError(
            f"{recipe.mechanism} recipe is missing controlled intervention groups: "
            f"{missing_interventions}"
        )
    if recipe.mechanism == "wgmma_fixed_completion" and not any(
        group.varied_axis == "cta_load" for group in recipe.interaction_groups
    ):
        raise ValueError("WGMMA recipe requires a CTA-load interaction group")
    if recipe.mechanism == "wgmma_fixed_completion":
        from amora.backends.nvidia.stall_metrics import stall_reason_for_metric

        present = {
            reason
            for metric in recipe.aggregate_metrics
            for reason in [stall_reason_for_metric(metric)]
            if reason is not None
        }
        missing = sorted(WGMMA_REQUIRED_STALL_REASONS - present)
        if missing:
            raise ValueError(f"WGMMA recipe is missing stall reasons: {missing}")
    if recipe.mechanism == "tma_route_offered_load" and not any(
        group.varied_axis in {"rotating_working_set_bytes", "cache_protocol"}
        for group in recipe.interaction_groups
    ):
        raise ValueError("TMA recipe requires a route/reuse interaction group")
    if recipe.mechanism == "tma_route_offered_load" and not any(
        group.varied_axis == "cta_concurrency"
        for group in recipe.interaction_groups
    ):
        raise ValueError("TMA recipe requires a CTA-concurrency interaction group")
    if recipe.mechanism == "tma_route_offered_load":
        lowered_metrics = tuple(metric.lower() for metric in recipe.aggregate_metrics)
        metric_requirements = {
            "TMA L2-to-L1TEX traffic": lambda metric: (
                "l1tex" in metric
                and "tma" in metric
                and ("byte" in metric or "sector" in metric)
            ),
            "L2 read sectors": lambda metric: (
                "lts__" in metric and "op_read" in metric
            ),
            "L2 hit sectors": lambda metric: (
                "lts__" in metric and "lookup_hit" in metric
            ),
            "L2 miss sectors": lambda metric: (
                "lts__" in metric and "lookup_miss" in metric
            ),
            "DRAM read traffic": lambda metric: (
                "dram__" in metric
                and "read" in metric
                and ("byte" in metric or "sector" in metric)
            ),
            "launch waves": lambda metric: "waves_per_multiprocessor" in metric,
            "occupancy context": lambda metric: "occupancy" in metric,
        }
        missing_metrics = [
            name
            for name, predicate in metric_requirements.items()
            if not any(predicate(metric) for metric in lowered_metrics)
        ]
        if missing_metrics:
            raise ValueError(
                f"TMA recipe is missing physical metric groups: {missing_metrics}"
            )


def _resolve_path(value: Any, path: str) -> Any:
    current = value
    parts = path.split(".")
    index = 0
    while index < len(parts):
        part = parts[index]
        if isinstance(current, Mapping):
            matched = False
            # NCU metric names contain dots. Prefer the longest remaining key
            # present in the current mapping before treating dots as nesting.
            for end in range(len(parts), index, -1):
                candidate = ".".join(parts[index:end])
                if candidate in current:
                    current = current[candidate]
                    index = end
                    matched = True
                    break
            if not matched:
                return None
        elif isinstance(current, list) and part.startswith("name="):
            name = part.split("=", 1)[1]
            current = next(
                (
                    item
                    for item in current
                    if isinstance(item, Mapping) and item.get("name") == name
                ),
                None,
            )
            index += 1
        else:
            return None
        if current is None:
            return None
    return current


def collect_causal_evidence(
    point: MechanismPoint, bundle: CommandMeasurementBundle
) -> dict[str, Any]:
    """Group same-case measured values under causal graph nodes."""

    evidence = {
        "bundle": bundle.to_dict(),
        "axes": dict(point.axes),
        "subject_metadata": dict(point.subject_metadata),
    }
    nodes = {}
    for node, paths in point.evidence_bindings.items():
        measurements = {path: _resolve_path(evidence, path) for path in paths}
        nodes[node] = {
            "status": (
                "measured"
                if any(value is not None for value in measurements.values())
                else "not_measured"
            ),
            "measurements": measurements,
        }
    histogram = evidence["bundle"]["aggregate_counters"].get(
        "structural_stall_histogram"
    )
    reason_fractions = (
        histogram.get("reason_fractions", {})
        if isinstance(histogram, Mapping)
        else {}
    )
    dominant_reason = (
        max(reason_fractions, key=reason_fractions.get) if reason_fractions else None
    )
    return {
        "graph": {
            "nodes": sorted({node for edge in CAUSAL_EDGES for node in edge}),
            "edges": [
                {"source": source, "target": target}
                for source, target in CAUSAL_EDGES
            ],
        },
        "nodes": nodes,
        "binding_observation": {
            "status": "measured" if dominant_reason is not None else "not_measured",
            "dominant_hardware_stall_reason": dominant_reason,
            "dominant_fraction": (
                reason_fractions.get(dominant_reason)
                if dominant_reason is not None
                else None
            ),
            "interpretation": "hardware_reason_only",
        },
    }


def _numeric_measurements(node: Mapping[str, Any] | None) -> dict[str, float]:
    if not node:
        return {}
    values = node.get("measurements")
    if not isinstance(values, Mapping):
        return {}
    return {
        str(path): float(value)
        for path, value in values.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }


def _changed(
    baseline: Mapping[str, float],
    variant: Mapping[str, float],
    threshold: float,
) -> bool | None:
    common = set(baseline) & set(variant)
    if not common:
        return None
    for key in common:
        left = baseline[key]
        right = variant[key]
        scale = max(abs(left), abs(right), 1e-12)
        if abs(right - left) / scale > threshold:
            return True
    return False


def build_mediation_report(
    recipe: MechanismRecipe,
    point_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Report measured edge mediation for controlled intervention groups."""

    groups: list[dict[str, Any]] = []
    for group in recipe.interaction_groups:
        baseline_id = group.point_ids[0]
        baseline_nodes = point_results[baseline_id]["causal_evidence"]["nodes"]
        baseline_identity = point_results[baseline_id]["bundle"]["identity_check"].get(
            "subject_identity"
        )
        baseline_qualifying = (
            point_results[baseline_id].get("qualification_status", "qualifying")
            == "qualifying"
        )
        comparisons = []
        for point_id in group.point_ids[1:]:
            variant_nodes = point_results[point_id]["causal_evidence"]["nodes"]
            variant_identity = point_results[point_id]["bundle"]["identity_check"].get(
                "subject_identity"
            )
            controls_valid = (
                baseline_qualifying
                and point_results[point_id].get(
                    "qualification_status", "qualifying"
                )
                == "qualifying"
                and (
                not group.requires_same_subject_identity
                or baseline_identity == variant_identity
                )
            )
            edge_rows = []
            for source, target in CAUSAL_EDGES:
                source_changed = _changed(
                    _numeric_measurements(baseline_nodes.get(source)),
                    _numeric_measurements(variant_nodes.get(source)),
                    group.change_threshold_fraction,
                )
                target_changed = _changed(
                    _numeric_measurements(baseline_nodes.get(target)),
                    _numeric_measurements(variant_nodes.get(target)),
                    group.change_threshold_fraction,
                )
                if not controls_valid:
                    status = "invalid_control_subject_identity_changed"
                elif source_changed is None or target_changed is None:
                    status = "not_measured"
                elif source_changed and target_changed:
                    status = "co_changed_under_controlled_intervention"
                elif source_changed and not target_changed:
                    status = "falsified_at_declared_threshold"
                else:
                    status = "upstream_unchanged"
                edge_rows.append(
                    {
                        "source": source,
                        "target": target,
                        "source_changed": source_changed,
                        "target_changed": target_changed,
                        "status": status,
                    }
                )
            comparisons.append(
                {
                    "baseline_point_id": baseline_id,
                    "variant_point_id": point_id,
                    "varied_axis": group.varied_axis,
                    "baseline_axis_value": point_results[baseline_id]["axes"][group.varied_axis],
                    "variant_axis_value": point_results[point_id]["axes"][group.varied_axis],
                    "control_status": "pass" if controls_valid else "invalid",
                    "binding_transition": {
                        "baseline": point_results[baseline_id]["causal_evidence"][
                            "binding_observation"
                        ],
                        "variant": point_results[point_id]["causal_evidence"][
                            "binding_observation"
                        ],
                    },
                    "edges": edge_rows,
                }
            )
        groups.append({**group.to_dict(), "comparisons": comparisons})

    wgmma_assessment = None
    wgmma_gap_knee = None
    if recipe.mechanism == "wgmma_fixed_completion":
        statuses: list[bool | None] = []
        for group_result in groups:
            if group_result["varied_axis"] != "cta_load":
                continue
            for comparison in group_result["comparisons"]:
                if comparison["control_status"] != "pass":
                    statuses.append(None)
                    continue
                completion = next(
                    (
                        edge
                        for edge in comparison["edges"]
                        if edge["target"] == "wgmma_completion"
                    ),
                    None,
                )
                statuses.append(
                    completion.get("target_changed") if completion else None
                )
        if any(status is True for status in statuses):
            classification = "shared_resource_or_scheduling_interaction"
        elif statuses and all(status is False for status in statuses):
            classification = "fixed_completion_candidate"
        else:
            classification = "not_measured"
        wgmma_assessment = {
            "classification": classification,
            "basis": "wgmma_completion sensitivity to cta_load",
        }
        gap_rows: list[dict[str, float | str]] = []
        for group_result in groups:
            if group_result["varied_axis"] != "independent_register_work_gap_ns":
                continue
            point_ids = group_result["point_ids"]
            for point_id in point_ids:
                if (
                    point_results[point_id].get(
                        "qualification_status", "qualifying"
                    )
                    != "qualifying"
                ):
                    continue
                exposed_wait_node = point_results[point_id]["causal_evidence"][
                    "nodes"
                ].get("wgmma_exposed_wait")
                exposed_wait_values = _numeric_measurements(exposed_wait_node)
                if exposed_wait_values:
                    gap = float(
                        point_results[point_id]["axes"][
                            "independent_register_work_gap_ns"
                        ]
                    )
                    exposed_wait = next(iter(exposed_wait_values.values()))
                    gap_rows.append(
                        {
                            "point_id": point_id,
                            "independent_register_work_gap_ns": gap,
                            "exposed_wait": exposed_wait,
                            "inferred_completion": gap + exposed_wait,
                        }
                    )
        if len(gap_rows) < 2:
            gap_classification = "not_measured"
        else:
            ordered = sorted(
                gap_rows,
                key=lambda row: float(row["independent_register_work_gap_ns"]),
            )
            waits_decrease = all(
                float(right["exposed_wait"]) <= float(left["exposed_wait"])
                for left, right in zip(ordered, ordered[1:])
            )
            inferred = [float(row["inferred_completion"]) for row in ordered]
            inferred_span = max(inferred) - min(inferred)
            inferred_scale = max(max(abs(value) for value in inferred), 1e-12)
            stable_completion = (
                inferred_span / inferred_scale
                <= max(
                    group.change_threshold_fraction
                    for group in recipe.interaction_groups
                        if group.varied_axis == "independent_register_work_gap_ns"
                )
            )
            gap_classification = (
                "fixed_completion_knee_candidate"
                if waits_decrease and stable_completion
                else "fixed_completion_knee_falsified"
            )
        wgmma_gap_knee = {
            "classification": gap_classification,
            "physical_test": "exposed_wait(D) = max(0, L_complete - D)",
            "observations": gap_rows,
        }
    return {
        "schema_version": 1,
        "kind": "physical_mechanism_mediation",
        "causal_graph_edges": [
            {"source": source, "target": target}
            for source, target in CAUSAL_EDGES
        ],
        "interaction_groups": groups,
        "wgmma_fixed_completion_assessment": wgmma_assessment,
        "wgmma_gap_knee_assessment": wgmma_gap_knee,
        "interpretation_boundary": (
            "Measured co-change or falsification under declared controls; "
            "portable model selection belongs to the consumer."
        ),
        "direct_regressors_emitted": [],
    }


def _write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_mechanism_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the content digest of a persisted mechanism-run manifest."""

    expected = manifest.get("run_digest")
    if not isinstance(expected, str) or not expected:
        raise ValueError("mechanism manifest has no run digest")
    payload = dict(manifest)
    payload.pop("run_digest", None)
    if _digest(payload) != expected:
        raise ValueError("mechanism manifest digest does not match its contents")


def load_mechanism_manifest(path: str | Path) -> dict[str, Any]:
    """Load and content-validate an immutable mechanism-run manifest."""

    manifest_path = Path(path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("mechanism manifest must be an object")
    validate_mechanism_manifest(value)
    artifact_entries = []
    for point_artifact in (value.get("artifacts") or {}).values():
        if isinstance(point_artifact, Mapping):
            artifact_entries.append(point_artifact)
            if isinstance(point_artifact.get("pc_report"), Mapping):
                artifact_entries.append(point_artifact["pc_report"])
    if isinstance(value.get("frozen_recipe"), Mapping):
        artifact_entries.append(value["frozen_recipe"])
    if isinstance(value.get("mediation"), Mapping):
        artifact_entries.append(value["mediation"])
    for artifact in artifact_entries:
        if not isinstance(artifact, Mapping):
            raise ValueError("mechanism manifest artifact must be an object")
        relative = artifact.get("path")
        expected_hash = artifact.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError("mechanism manifest artifact lacks path or sha256")
        artifact_path = manifest_path.parent / relative
        if not artifact_path.is_file():
            raise ValueError(f"mechanism artifact is missing: {relative}")
        actual_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"mechanism artifact hash mismatch: {relative}")
    return value


def execute_mechanism_recipe(
    recipe: MechanismRecipe,
    *,
    capabilities: NvidiaCapabilities,
    output_root: str | Path,
    run_id: str,
    timeout: int = 300,
    cwd: str | Path | None = None,
    environment_overrides: Mapping[str, str] | None = None,
    bundle_collector: Callable[
        ..., CommandMeasurementBundle
    ] = collect_command_measurement_bundle,
) -> Path:
    """Execute a recipe into a unique immutable run directory."""

    validate_mechanism_recipe(recipe)
    if not _SAFE_ID.fullmatch(run_id):
        raise ValueError("run_id must be filesystem-safe")
    run_dir = Path(output_root) / recipe.recipe_id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    frozen_recipe_path = run_dir / "recipe.json"
    _write_new_json(frozen_recipe_path, recipe.to_dict())
    point_results: dict[str, dict[str, Any]] = {}
    artifacts = {}
    ordered_points = list(recipe.points)
    random.Random(recipe.randomization_seed).shuffle(ordered_points)
    for point in ordered_points:
        point_dir = run_dir / "points" / point.point_id
        point_dir.mkdir(parents=True, exist_ok=False)
        pc_report = point_dir / "pc_sampling.ncu-rep"
        bundle = bundle_collector(
            timing_target=point.timing_target,
            ncu_target=point.ncu_target,
            device_interval_target=point.device_interval_target,
            capabilities=capabilities,
            aggregate_metrics=recipe.aggregate_metrics,
            kernel_name=recipe.kernel_name,
            cache_control=recipe.cache_control,
            clock_control=recipe.clock_control,
            repeats=recipe.repeats,
            collect_pc_sampling=recipe.collect_pc_sampling,
            expected_launch_batch_size=point.expected_launch_batch_size,
            expected_cache_protocol=(
                point.expected_cache_protocol or point.axes.get("cache_protocol")
            ),
            expected_max_interval_overhead_percent=(
                point.expected_max_interval_overhead_percent
            ),
            expected_measurement_axes=point.axes,
            required_identity_fields=recipe.required_identity_fields,
            required_subject_metadata_fields=tuple(
                sorted(REQUIRED_SUBJECT_METADATA)
            ),
            timeout=timeout,
            launch_skip=recipe.launch_skip,
            pc_launch_skip=recipe.pc_launch_skip,
            sampling_interval=recipe.sampling_interval,
            cwd=cwd,
            environment_overrides=environment_overrides,
            pc_report_path=pc_report,
        )
        causal = collect_causal_evidence(point, bundle)
        diagnostic_reasons: list[str] = []
        if not bundle.valid_for_comparison:
            diagnostic_reasons.extend(bundle.identity_check.get("reasons") or ())
        interval_identity = (
            bundle.device_intervals.subject_identity
            if bundle.device_intervals is not None
            else None
        )
        interval_metadata = (
            bundle.device_intervals.subject_metadata
            if bundle.device_intervals is not None
            else None
        )
        if interval_identity is not None:
            for field_name in recipe.required_identity_fields:
                if (
                    point.axes.get(field_name) is not None
                    and interval_identity.get(field_name) != point.axes[field_name]
                ):
                    diagnostic_reasons.append(
                        f"recipe_interval_identity_mismatch:{field_name}"
                    )
        measured_metadata = bundle.timing.subject_metadata
        for field_name in REQUIRED_SUBJECT_METADATA:
            if measured_metadata.get(field_name) != point.subject_metadata.get(field_name):
                diagnostic_reasons.append(
                    f"recipe_subject_metadata_mismatch:{field_name}"
                )
            if (
                interval_metadata is not None
                and interval_metadata.get(field_name)
                != point.subject_metadata.get(field_name)
            ):
                diagnostic_reasons.append(
                    f"recipe_interval_metadata_mismatch:{field_name}"
                )
        if point.subject_metadata.get("spill_count", 0):
            diagnostic_reasons.append("subject_contains_spills")
        pc_payload = bundle.pc_sampling
        if recipe.collect_pc_sampling and (
            pc_payload is None
            or pc_payload.get("instruction_join_fraction") != 1.0
        ):
            diagnostic_reasons.append("pc_to_sass_join_incomplete")
        if (
            bundle.device_intervals is not None
            and bundle.device_intervals.evidence_status == "diagnostic"
        ):
            diagnostic_reasons.extend(bundle.device_intervals.diagnostic_reasons)
        missing_nodes = [
            node
            for node in REQUIRED_EVIDENCE_NODES[recipe.mechanism]
            if causal["nodes"][node]["status"] == "not_measured"
        ]
        diagnostic_reasons.extend(f"not_measured:{node}" for node in missing_nodes)
        point_payload = {
            "schema_version": 1,
            "kind": "physical_mechanism_point",
            "point_id": point.point_id,
            "split": point.split,
            "axes": dict(point.axes),
            "subject_metadata": dict(point.subject_metadata),
            "qualification_status": (
                "diagnostic" if diagnostic_reasons else "qualifying"
            ),
            "diagnostic_reasons": list(dict.fromkeys(diagnostic_reasons)),
            "bundle": bundle.to_dict(),
            "causal_evidence": causal,
        }
        bundle_path = point_dir / "bundle.json"
        _write_new_json(bundle_path, point_payload)
        point_results[point.point_id] = point_payload
        artifacts[point.point_id] = {
            "path": str(bundle_path.relative_to(run_dir)),
            "sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
            "qualification_status": point_payload["qualification_status"],
            "split": point.split,
        }
        if bundle.pc_sampling is not None:
            pc_provenance = bundle.pc_sampling.get("provenance") or {}
            report_hash = pc_provenance.get("report_sha256")
            if pc_report.is_file() and isinstance(report_hash, str):
                artifacts[point.point_id]["pc_report"] = {
                    "path": str(pc_report.relative_to(run_dir)),
                    "sha256": report_hash,
                }

    mediation = build_mediation_report(recipe, point_results)
    mediation_path = run_dir / "mediation.json"
    _write_new_json(mediation_path, mediation)
    manifest = {
        "schema_version": 1,
        "kind": "physical_mechanism_run",
        "run_id": run_id,
        "recipe": recipe.to_dict(),
        "recipe_digest": _digest(recipe.to_dict()),
        "frozen_recipe": {
            "path": str(frozen_recipe_path.relative_to(run_dir)),
            "sha256": hashlib.sha256(frozen_recipe_path.read_bytes()).hexdigest(),
        },
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "execution_order": [point.point_id for point in ordered_points],
        "randomization_seed": recipe.randomization_seed,
        "artifacts": artifacts,
        "mediation": {
            "path": str(mediation_path.relative_to(run_dir)),
            "sha256": hashlib.sha256(mediation_path.read_bytes()).hexdigest(),
        },
        "split_counts": {
            split: sum(point.split == split for point in recipe.points)
            for split in sorted(SPLITS)
        },
        "capabilities": capabilities.to_dict(),
    }
    manifest["run_digest"] = _digest(manifest)
    _write_new_json(run_dir / "manifest.json", manifest)
    return run_dir
