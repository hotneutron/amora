"""Physical-mechanism recipes and immutable measurement artifacts.

The structures in this module preserve interventions and measured causal links.
They deliberately do not fit elapsed time, problem size, or raw working-set size
as direct latency corrections. Model interpretation remains with the consumer.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.measurement import (
    CommandMeasurementBundle,
    collect_command_measurement_bundle,
)


MECHANISMS = frozenset(
    {
        "wgmma_fixed_completion",
        "tma_route_offered_load",
        "barrier_topology_release",
    }
)
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
BARRIER_TOPOLOGY_CAUSAL_EDGES = (
    ("footprint_occupancy", "producer_eligibility"),
    ("producer_eligibility", "tma_issue"),
    ("tma_issue", "memory_route"),
    ("memory_route", "tma_completion"),
    ("tma_completion", "barrier_release"),
    ("barrier_release", "consumer_issue"),
    ("consumer_issue", "wgmma_completion"),
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
    "barrier_topology_release": frozenset(
        {
            "panel",
            "topology",
            "M",
            "N",
            "K",
            "tile_shape",
            "grid",
            "pipeline_depth",
            "producer_asymmetry_level",
            "cache_protocol",
            "cta_concurrency",
            "address_partition_mapping",
            "launch_batch_size",
            "clock_policy",
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
    "barrier_topology_release": frozenset(
        {
            "footprint_occupancy",
            "producer_eligibility",
            "tma_issue",
            "memory_route",
            "tma_completion",
            "barrier_release",
            "consumer_issue",
            "wgmma_completion",
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
    "barrier_topology_release": {
        "footprint_occupancy": ("bundle.aggregate_counters.",),
        "producer_eligibility": ("bundle.aggregate_counters.",),
        "tma_issue": (
            "bundle.aggregate_counters.",
            "bundle.pc_sampling.",
        ),
        "memory_route": ("bundle.aggregate_counters.",),
        "tma_completion": ("bundle.device_intervals.",),
        "barrier_release": (
            "bundle.device_intervals.",
            "bundle.pc_sampling.",
        ),
        "consumer_issue": (
            "bundle.device_intervals.",
            "bundle.pc_sampling.",
        ),
        "wgmma_completion": ("bundle.device_intervals.",),
        "stage_buffer_release": ("bundle.device_intervals.",),
        "total_latency": ("bundle.timing.",),
    },
}
REQUIRED_SUBJECT_METADATA = frozenset(
    {"registers_per_thread", "shared_memory_bytes", "spill_count"}
)
BARRIER_REQUIRED_SUBJECT_METADATA = frozenset(
    {
        "registers_per_thread",
        "shared_memory_bytes",
        "spill_count",
        "total_warps",
        "total_threads",
        "actor_count",
        "compiled_queue_capacity",
        "producer_lead",
        "request_count",
        "request_bytes",
        "numerical_result_valid",
        "numerical_result_fingerprint",
        "useful_operation_fingerprint",
        "synchronization_instruction_fingerprint",
    }
)
BARRIER_REQUIRED_IDENTITY_FIELDS = frozenset(
    {
        "kernel_name",
        "source_sha256",
        "ttgir_sha256",
        "ptx_sha256",
        "sass_sha256",
        "cubin_sha256",
    }
)
BARRIER_PANELS = {
    "BT-EQ": frozenset({"joint", "split"}),
    "BT-CAUSAL": frozenset({"joint", "pairwise"}),
}
BARRIER_PC_REGIONS = frozenset(
    {
        "tma_issue",
        "mbarrier_poll_fast",
        "mbarrier_poll_retry",
        "wgmma_wait",
        "wgmma_issue",
        "unrelated_prologue_epilogue",
    }
)
BARRIER_ALLOWED_SASS_DIFFERENCE_CLASSES = frozenset(
    {"synchronization", "predicate", "branch", "barrier_address_setup"}
)
BARRIER_HARD_INVARIANT_COUNTER_REQUIREMENTS = {
    "GMMA instruction count": lambda path: (
        "gmma" in path and ("inst" in path or "instruction" in path)
    ),
    "TMA global-load bytes": lambda path: (
        "tma" in path and "global" in path and "byte" in path
    ),
    "shared-memory occupancy limit": lambda path: (
        "occupancy_limit_shared_mem" in path
    ),
}
BARRIER_HARD_INVARIANT_METADATA_FIELDS = frozenset(
    {
        "useful_operation_fingerprint",
        "request_count",
        "request_bytes",
        "registers_per_thread",
        "shared_memory_bytes",
        "spill_count",
    }
)
BARRIER_MEDIATION_COUNTER_REQUIREMENTS = {
    "L2 read sectors": lambda path: (
        "lts__" in path
        and "op_read" in path
        and "lookup_hit" not in path
        and "lookup_miss" not in path
    ),
    "L2 hit sectors": lambda path: (
        "lts__" in path and "lookup_hit" in path
    ),
    "L2 miss sectors": lambda path: (
        "lts__" in path and "lookup_miss" in path
    ),
    "DRAM read bytes": lambda path: (
        "dram__" in path and "read" in path and "byte" in path
    ),
}
WGMMA_REQUIRED_STALL_REASONS = frozenset(
    {"wait", "math_pipe_throttle", "barrier", "warpgroup_arrive"}
)
BARRIER_REQUIRED_STALL_REASONS = frozenset(
    {
        "long_scoreboard",
        "barrier",
        "wait",
        "mio_throttle",
        "warpgroup_arrive",
        "selected",
        "not_selected",
    }
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
    "barrier_topology_release": {
        "portable_test": (
            "BT-EQ preserves release at max(A.complete, B.complete); "
            "BT-CAUSAL pairwise topology permits the fast consumer to issue "
            "before the slow producer completes."
        ),
        "classification_boundary": (
            "Hardware evidence measures the partial order and localized "
            "protocol cost; portable P3 interpretation remains external."
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
    pc_region_map: dict[str, tuple[int, ...]] = field(default_factory=dict)
    pc_sampling_repeats: int | None = None
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
            "pc_region_map": {
                region: list(offsets)
                for region, offsets in self.pc_region_map.items()
            },
            "pc_sampling_repeats": self.pc_sampling_repeats,
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
    invariant_counter_paths: tuple[str, ...] = ()
    allowed_identity_differences: tuple[str, ...] = ()
    allowed_subject_metadata_differences: tuple[str, ...] = ()
    allowed_sass_difference_classes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "point_ids": list(self.point_ids),
            "varied_axis": self.varied_axis,
            "intervention_node": self.intervention_node,
            "change_threshold_fraction": self.change_threshold_fraction,
            "requires_same_subject_identity": self.requires_same_subject_identity,
            "invariant_counter_paths": list(self.invariant_counter_paths),
            "allowed_identity_differences": list(
                self.allowed_identity_differences
            ),
            "allowed_subject_metadata_differences": list(
                self.allowed_subject_metadata_differences
            ),
            "allowed_sass_difference_classes": list(
                self.allowed_sass_difference_classes
            ),
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
    pc_sampling_repeats: int = 1
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
            "pc_sampling_repeats": self.pc_sampling_repeats,
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
            pc_region_map={
                region: tuple(offsets)
                for region, offsets in (item.get("pc_region_map") or {}).items()
            },
            pc_sampling_repeats=item.get("pc_sampling_repeats"),
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
            invariant_counter_paths=tuple(
                item.get("invariant_counter_paths") or ()
            ),
            allowed_identity_differences=tuple(
                item.get("allowed_identity_differences") or ()
            ),
            allowed_subject_metadata_differences=tuple(
                item.get("allowed_subject_metadata_differences") or ()
            ),
            allowed_sass_difference_classes=tuple(
                item.get("allowed_sass_difference_classes") or ()
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
        pc_sampling_repeats=int(data.get("pc_sampling_repeats", 1)),
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
    if (
        isinstance(recipe.pc_sampling_repeats, bool)
        or not isinstance(recipe.pc_sampling_repeats, int)
        or recipe.pc_sampling_repeats <= 0
    ):
        raise ValueError("pc_sampling_repeats must be positive")
    if recipe.mechanism == "barrier_topology_release" and recipe.repeats < 7:
        raise ValueError("barrier topology recipes require at least seven timing repeats")
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
    required_metadata = (
        BARRIER_REQUIRED_SUBJECT_METADATA
        if recipe.mechanism == "barrier_topology_release"
        else REQUIRED_SUBJECT_METADATA
    )
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
        missing_metadata = sorted(required_metadata - set(point.subject_metadata))
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
        if point.pc_sampling_repeats is not None and (
            isinstance(point.pc_sampling_repeats, bool)
            or not isinstance(point.pc_sampling_repeats, int)
            or point.pc_sampling_repeats <= 0
        ):
            raise ValueError(
                f"{point.point_id} pc_sampling_repeats must be positive"
            )
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
        if recipe.mechanism == "barrier_topology_release":
            panel = point.axes["panel"]
            topology = point.axes["topology"]
            if panel not in BARRIER_PANELS:
                raise ValueError(f"{point.point_id} has an invalid topology panel")
            if topology not in BARRIER_PANELS[panel]:
                raise ValueError(
                    f"{point.point_id} topology {topology!r} is invalid for {panel}"
                )
            if point.subject_metadata["numerical_result_valid"] is not True:
                raise ValueError(
                    f"{point.point_id} does not declare a valid numerical result"
                )
            if (
                point.pc_sampling_repeats or recipe.pc_sampling_repeats
            ) < 3:
                raise ValueError(
                    f"{point.point_id} requires at least three PC-sampling repeats"
                )
            missing_identity = sorted(
                BARRIER_REQUIRED_IDENTITY_FIELDS
                - set(recipe.required_identity_fields)
            )
            if missing_identity:
                raise ValueError(
                    "barrier topology required_identity_fields omit: "
                    f"{missing_identity}"
                )
            missing_regions = sorted(BARRIER_PC_REGIONS - set(point.pc_region_map))
            if recipe.collect_pc_sampling and missing_regions:
                raise ValueError(
                    f"{point.point_id} is missing PC regions: {missing_regions}"
                )
            offsets = [
                offset
                for region_offsets in point.pc_region_map.values()
                for offset in region_offsets
            ]
            if any(
                isinstance(offset, bool)
                or not isinstance(offset, int)
                or offset < 0
                for offset in offsets
            ):
                raise ValueError(
                    f"{point.point_id} PC region offsets must be non-negative integers"
                )
            if len(offsets) != len(set(offsets)):
                raise ValueError(
                    f"{point.point_id} PC region offsets must map to one region"
                )
            if point.expected_launch_batch_size != point.axes["launch_batch_size"]:
                raise ValueError(
                    f"{point.point_id} launch batch declaration does not match axes"
                )
            if declared_cache_protocol != point.axes["cache_protocol"]:
                raise ValueError(
                    f"{point.point_id} cache protocol declaration does not match axes"
                )
            if recipe.clock_control != point.axes["clock_policy"]:
                raise ValueError(
                    f"{point.point_id} clock policy declaration does not match recipe"
                )
            if (
                point.expected_max_interval_overhead_percent is None
                or point.expected_max_interval_overhead_percent > 5.0
            ):
                raise ValueError(
                    f"{point.point_id} interval overhead threshold must be at most 5%"
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
        control_axes = (
            set().union(*(set(point.axes) for point in members))
            - {group.varied_axis}
        )
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
        if recipe.mechanism == "barrier_topology_release":
            if group.varied_axis != "topology":
                raise ValueError(
                    f"{group.group_id} must vary only topology for a topology pair"
                )
            if len(group.point_ids) != 2:
                raise ValueError(f"{group.group_id} must contain exactly two points")
            if len({point.split for point in members}) != 1:
                raise ValueError(
                    f"{group.group_id} cannot pair calibration and held-out points"
                )
            if {point.axes["topology"] for point in members} != BARRIER_PANELS[
                members[0].axes["panel"]
            ]:
                raise ValueError(
                    f"{group.group_id} does not contain the required topology pair"
                )
            if group.requires_same_subject_identity:
                raise ValueError(
                    f"{group.group_id} must allow declared topology binary differences"
                )
            required_allowed_identity = BARRIER_REQUIRED_IDENTITY_FIELDS - {
                "kernel_name"
            }
            if set(group.allowed_identity_differences) != required_allowed_identity:
                raise ValueError(
                    f"{group.group_id} must declare only compiler-artifact identity "
                    "differences"
                )
            if set(group.allowed_subject_metadata_differences) != {
                "synchronization_instruction_fingerprint"
            }:
                raise ValueError(
                    f"{group.group_id} must allow only the synchronization "
                    "instruction fingerprint to differ"
                )
            if set(group.allowed_sass_difference_classes) != (
                BARRIER_ALLOWED_SASS_DIFFERENCE_CLASSES
            ):
                raise ValueError(
                    f"{group.group_id} must limit SASS differences to "
                    "synchronization/control classes"
                )
            if not group.invariant_counter_paths:
                raise ValueError(
                    f"{group.group_id} must declare invariant counter paths"
                )
            lowered_counter_paths = tuple(
                path.lower() for path in group.invariant_counter_paths
            )
            matched_invariants = {
                name: [
                    path
                    for path in lowered_counter_paths
                    if predicate(path)
                ]
                for name, predicate in (
                    BARRIER_HARD_INVARIANT_COUNTER_REQUIREMENTS.items()
                )
            }
            missing_invariants = [
                name for name, paths in matched_invariants.items() if not paths
            ]
            if missing_invariants:
                raise ValueError(
                    f"{group.group_id} is missing invariant counter groups: "
                    f"{missing_invariants}"
                )
            duplicate_invariants = {
                name: paths
                for name, paths in matched_invariants.items()
                if len(paths) > 1
            }
            non_hard_invariants = [
                path
                for path in lowered_counter_paths
                if not any(
                    predicate(path)
                    for predicate in (
                        BARRIER_HARD_INVARIANT_COUNTER_REQUIREMENTS.values()
                    )
                )
            ]
            if duplicate_invariants or non_hard_invariants:
                raise ValueError(
                    f"{group.group_id} must declare exactly one counter path "
                    "for each hard invariant group; "
                    f"duplicates={duplicate_invariants}, "
                    f"non_hard={non_hard_invariants}"
                )
            metadata_differences = {
                field_name
                for field_name in BARRIER_REQUIRED_SUBJECT_METADATA
                if canonical_json(members[0].subject_metadata.get(field_name))
                != canonical_json(members[1].subject_metadata.get(field_name))
            }
            hard_metadata_differences = (
                metadata_differences & BARRIER_HARD_INVARIANT_METADATA_FIELDS
            )
            if hard_metadata_differences:
                raise ValueError(
                    f"{group.group_id} topology pair hard metadata differences "
                    f"are invalid: {sorted(hard_metadata_differences)}"
                )
            for member in members:
                useful_fingerprint = member.subject_metadata[
                    "useful_operation_fingerprint"
                ]
                if not isinstance(useful_fingerprint, Mapping) or not {
                    "tma", "hgmma"
                } <= set(useful_fingerprint):
                    raise ValueError(
                        f"{member.point_id} useful-operation fingerprint must "
                        "declare TMA and HGMMA components"
                    )
                fingerprint = member.subject_metadata[
                    "synchronization_instruction_fingerprint"
                ]
                if not isinstance(fingerprint, Mapping):
                    raise ValueError(
                        f"{member.point_id} synchronization fingerprint must "
                        "be an object"
                    )
                difference_classes = fingerprint.get("difference_classes")
                if not isinstance(difference_classes, list) or not all(
                    isinstance(value, str) for value in difference_classes
                ):
                    raise ValueError(
                        f"{member.point_id} synchronization fingerprint must "
                        "declare difference_classes"
                    )
                if not fingerprint.get("non_sync_digest"):
                    raise ValueError(
                        f"{member.point_id} synchronization fingerprint must "
                        "declare non_sync_digest"
                    )
                if not set(difference_classes) <= set(
                    group.allowed_sass_difference_classes
                ):
                    raise ValueError(
                        f"{member.point_id} declares a disallowed SASS difference"
                    )
    varied_axes = {group.varied_axis for group in recipe.interaction_groups}
    if recipe.mechanism == "barrier_topology_release":
        represented_panels = {
            points[group.point_ids[0]].axes["panel"]
            for group in recipe.interaction_groups
        }
        if represented_panels != set(BARRIER_PANELS):
            raise ValueError(
                "barrier topology recipe must contain BT-EQ and BT-CAUSAL pairs"
            )
        represented_panel_splits = {
            (points[group.point_ids[0]].axes["panel"], points[group.point_ids[0]].split)
            for group in recipe.interaction_groups
        }
        required_panel_splits = {
            (panel, split) for panel in BARRIER_PANELS for split in SPLITS
        }
        if represented_panel_splits != required_panel_splits:
            raise ValueError(
                "barrier topology recipe must pair calibration and held-out "
                "points in both panels"
            )
    else:
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
    if recipe.mechanism == "barrier_topology_release":
        from amora.backends.nvidia.stall_metrics import stall_reason_for_metric

        stall_metrics = [
            metric for metric in recipe.aggregate_metrics
            if stall_reason_for_metric(metric) is not None
        ]
        families = {
            "per_warp_active"
            for metric in stall_metrics
            if "_per_warp_active" in metric
        } | {
            "per_issue_active"
            for metric in stall_metrics
            if "_per_issue_active" in metric
        }
        if families != {"per_warp_active"}:
            raise ValueError(
                "barrier topology stalls require one per_warp_active family"
            )
        present_reasons = {
            reason
            for metric in stall_metrics
            for reason in [stall_reason_for_metric(metric)]
            if reason is not None
        }
        missing_reasons = sorted(
            BARRIER_REQUIRED_STALL_REASONS - present_reasons
        )
        if missing_reasons:
            raise ValueError(
                f"barrier topology recipe is missing stall reasons: {missing_reasons}"
            )
        lowered_metrics = tuple(metric.lower() for metric in recipe.aggregate_metrics)
        metric_requirements = {
            "GMMA instruction count": lambda metric: (
                "gmma" in metric and ("inst" in metric or "instruction" in metric)
            ),
            "TMA global-load bytes": lambda metric: (
                "tma" in metric and "global" in metric and "byte" in metric
            ),
            "active warps": lambda metric: (
                "active_warps" in metric or "warps_active" in metric
            ),
            "eligible warps": lambda metric: (
                "eligible_warps" in metric or "warps_eligible" in metric
            ),
            "shared-memory occupancy limit": lambda metric: (
                "occupancy_limit_shared_mem" in metric
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
            "DRAM read bytes": lambda metric: (
                "dram__" in metric and "read" in metric and "byte" in metric
            ),
        }
        missing_metrics = [
            name
            for name, predicate in metric_requirements.items()
            if not any(predicate(metric) for metric in lowered_metrics)
        ]
        if missing_metrics:
            raise ValueError(
                "barrier topology recipe is missing physical metric groups: "
                f"{missing_metrics}"
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
    graph_edges = (
        BARRIER_TOPOLOGY_CAUSAL_EDGES
        if set(point.evidence_bindings) >= REQUIRED_EVIDENCE_NODES[
            "barrier_topology_release"
        ]
        else CAUSAL_EDGES
    )
    return {
        "graph": {
            "nodes": sorted({node for edge in graph_edges for node in edge}),
            "edges": [
                {"source": source, "target": target}
                for source, target in graph_edges
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


def _structural_sample_count(sample: Mapping[str, Any]) -> float:
    stalls = sample.get("stalls")
    if not isinstance(stalls, Mapping):
        return 0.0
    return sum(
        float(value)
        for reason, value in stalls.items()
        if reason not in {"selected", "not_selected"}
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def localize_pc_regions(
    pc_sampling: Mapping[str, Any] | None,
    region_map: Mapping[str, tuple[int, ...]],
) -> dict[str, Any]:
    """Apply a consumer-reviewed offset map and assess repeat agreement."""

    if not pc_sampling:
        return {
            "status": "diagnostic",
            "reason": "pc_sampling_not_measured",
            "raw_unit": "pc_sample_count",
            "pooled_structural_support": 0.0,
            "dominant_region": None,
            "dominant_region_repeat_count": 0,
            "repeats": [],
            "regions": {},
        }
    offset_to_region = {
        offset: region
        for region, offsets in region_map.items()
        for offset in offsets
    }
    repeat_rows = []
    pooled = {region: 0.0 for region in region_map}
    unmapped_offsets: set[int] = set()
    for repeat in pc_sampling.get("repeats") or ():
        regions = {region: 0.0 for region in region_map}
        for sample in repeat.get("samples") or ():
            support = _structural_sample_count(sample)
            if support <= 0.0:
                continue
            offset = sample.get("pc_offset")
            region = offset_to_region.get(offset)
            if region is None:
                if isinstance(offset, int):
                    unmapped_offsets.add(offset)
                continue
            regions[region] += support
            pooled[region] += support
        supported = {region: count for region, count in regions.items() if count > 0.0}
        dominant = max(supported, key=lambda region: supported[region]) if supported else None
        repeat_rows.append(
            {
                "repeat_index": repeat.get("repeat_index"),
                "structural_support_by_region": regions,
                "structural_support_count": sum(regions.values()),
                "dominant_region": dominant,
            }
        )
    pooled_supported = {region: count for region, count in pooled.items() if count > 0.0}
    dominant_region = (
        max(pooled_supported, key=lambda region: pooled_supported[region])
        if pooled_supported
        else None
    )
    dominant_repeats = sum(
        row["dominant_region"] == dominant_region for row in repeat_rows
    )
    pooled_support = sum(pooled.values())
    qualifying = (
        not unmapped_offsets
        and pooled_support >= 100.0
        and dominant_region is not None
        and dominant_repeats >= 2
    )
    reasons = []
    if unmapped_offsets:
        reasons.append("semantic_region_map_incomplete")
    if pooled_support < 100.0:
        reasons.append("pooled_structural_support_below_100")
    if dominant_region is None or dominant_repeats < 2:
        reasons.append("dominant_region_not_reproduced_in_two_profiles")
    return {
        "status": "qualifying" if qualifying else "diagnostic",
        "raw_unit": "pc_sample_count",
        "pooled_structural_support": pooled_support,
        "regions": pooled,
        "dominant_region": dominant_region,
        "dominant_region_repeat_count": dominant_repeats,
        "required_repeat_agreement": 2,
        "unmapped_offsets": sorted(unmapped_offsets),
        "repeats": repeat_rows,
        "diagnostic_reasons": reasons,
    }


def _relative_difference(left: float, right: float) -> float:
    return abs(right - left) / max(abs(left), abs(right), 1e-12)


def _paired_timing_difference(
    baseline: Mapping[str, Any], variant: Mapping[str, Any]
) -> dict[str, Any]:
    left = [
        float(row["median_us"])
        for row in baseline.get("process_samples") or ()
    ]
    right = [
        float(row["median_us"])
        for row in variant.get("process_samples") or ()
    ]
    count = min(len(left), len(right))
    differences = [right[index] - left[index] for index in range(count)]
    if not differences:
        return {
            "status": "not_measured",
            "unit": "us_per_launch",
            "paired_process_count": 0,
        }
    mean_difference = statistics.fmean(differences)
    t_critical_95 = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        24: 2.064,
        29: 2.045,
    }
    degrees_of_freedom = count - 1
    if degrees_of_freedom <= 20:
        critical = t_critical_95[degrees_of_freedom]
    elif degrees_of_freedom <= 24:
        critical = t_critical_95[24]
    elif degrees_of_freedom <= 29:
        critical = t_critical_95[29]
    else:
        critical = 1.96
    half_width = (
        critical * statistics.stdev(differences) / math.sqrt(count)
        if count > 1
        else None
    )
    baseline_mean = statistics.fmean(left[:count])
    return {
        "status": "measured",
        "unit": "us_per_launch",
        "paired_process_count": count,
        "paired_differences": differences,
        "mean_difference": mean_difference,
        "mean_difference_percent": (
            mean_difference / baseline_mean * 100.0 if baseline_mean else None
        ),
        "confidence_level": 0.95,
        "confidence_interval_method": "paired_student_t",
        "confidence_interval": (
            [mean_difference - half_width, mean_difference + half_width]
            if half_width is not None
            else None
        ),
        "confidence_half_width": half_width,
        "baseline_mean_us": baseline_mean,
    }


def _interval_by_name(point_result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    interval_payload = point_result.get("bundle", {}).get("device_intervals")
    intervals = (
        interval_payload.get("intervals", [])
        if isinstance(interval_payload, Mapping)
        else []
    )
    return {
        str(interval.get("name")): interval
        for interval in intervals or ()
        if isinstance(interval, Mapping)
    }


def _interval_order_assessment(
    panel: str, point_results: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if panel != "BT-CAUSAL":
        required = {
            "tma_issue_to_barrier_release",
            "barrier_release_to_consumer_issue",
            "wgmma_issue_to_completion",
            "stage_buffer_release",
        }
        missing = {
            point_id: sorted(required - set(_interval_by_name(result)))
            for point_id, result in point_results.items()
        }
        missing = {point_id: names for point_id, names in missing.items() if names}
        diagnostic = {
            point_id: list(
                (result["bundle"].get("device_intervals") or {}).get(
                    "diagnostic_reasons"
                ) or ()
            )
            for point_id, result in point_results.items()
            if (result["bundle"].get("device_intervals") or {}).get(
                "evidence_status"
            ) != "qualifying"
        }
        return {
            "status": (
                "not_measured" if missing or diagnostic else "supported"
            ),
            "required_intervals": sorted(required),
            "missing_intervals": missing,
            "diagnostic_intervals": diagnostic,
        }

    required = {
        "fast_tma_issue_to_barrier_release",
        "slow_tma_issue_to_barrier_release",
        "fast_barrier_release_to_consumer_issue",
        "slow_barrier_release_to_consumer_issue",
        "wgmma_issue_to_completion",
        "stage_buffer_release",
    }
    observations = {}
    missing = {}
    diagnostic = {}
    for point_id, result in point_results.items():
        interval_payload = result["bundle"].get("device_intervals") or {}
        if interval_payload.get("evidence_status") != "qualifying":
            diagnostic[point_id] = list(
                interval_payload.get("diagnostic_reasons") or ()
            )
            continue
        intervals = _interval_by_name(result)
        absent = sorted(required - set(intervals))
        if absent:
            missing[point_id] = absent
            continue
        fast_consumer_issue = float(
            intervals["fast_barrier_release_to_consumer_issue"]["end_ns"]
        )
        slow_producer_release = float(
            intervals["slow_tma_issue_to_barrier_release"]["end_ns"]
        )
        observations[point_id] = {
            "topology": result["axes"]["topology"],
            "fast_consumer_issue_ns": fast_consumer_issue,
            "slow_producer_release_ns": slow_producer_release,
            "fast_consumer_precedes_slow_producer_release": (
                fast_consumer_issue < slow_producer_release
            ),
        }
    if diagnostic or missing or len(observations) != 2:
        status = "not_measured"
    else:
        by_topology = {row["topology"]: row for row in observations.values()}
        status = (
            "supported"
            if by_topology["pairwise"][
                "fast_consumer_precedes_slow_producer_release"
            ]
            and not by_topology["joint"][
                "fast_consumer_precedes_slow_producer_release"
            ]
            else "falsified"
        )
    return {
        "status": status,
        "required_intervals": sorted(required),
        "missing_intervals": missing,
        "diagnostic_intervals": diagnostic,
        "observations": observations,
    }


def _pair_control_scorecard(
    group: InteractionGroup,
    baseline: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    aggregate_metrics: tuple[str, ...] = (),
) -> dict[str, Any]:
    drift_reasons = []
    missing_reasons = []
    left_bundle = baseline["bundle"]
    right_bundle = variant["bundle"]
    axis_differences = sorted(
        axis
        for axis in set(baseline["axes"]) | set(variant["axes"])
        if canonical_json(baseline["axes"].get(axis))
        != canonical_json(variant["axes"].get(axis))
    )
    if axis_differences != [group.varied_axis]:
        drift_reasons.append("runtime_axis_drift")

    left_identity = left_bundle["timing"].get("subject_identity", {})
    right_identity = right_bundle["timing"].get("subject_identity", {})
    identity_differences = sorted(
        field
        for field in set(left_identity) | set(right_identity)
        if left_identity.get(field) != right_identity.get(field)
    )
    unexpected_identity = sorted(
        set(identity_differences) - set(group.allowed_identity_differences)
    )
    if unexpected_identity:
        drift_reasons.append("unexpected_subject_identity_drift")

    left_metadata = left_bundle["timing"].get("subject_metadata", {})
    right_metadata = right_bundle["timing"].get("subject_metadata", {})
    metadata_differences = sorted(
        field
        for field in set(left_metadata) | set(right_metadata)
        if canonical_json(left_metadata.get(field))
        != canonical_json(right_metadata.get(field))
    )
    hard_metadata_differences = sorted(
        set(metadata_differences) & BARRIER_HARD_INVARIANT_METADATA_FIELDS
    )
    unexpected_metadata = sorted(
        set(metadata_differences)
        - set(group.allowed_subject_metadata_differences)
        - BARRIER_HARD_INVARIANT_METADATA_FIELDS
    )
    if hard_metadata_differences:
        drift_reasons.append("hard_subject_metadata_drift")
    if left_metadata.get("spill_count") or right_metadata.get("spill_count"):
        drift_reasons.append("subject_contains_spills")
    sync_field = "synchronization_instruction_fingerprint"
    if left_metadata.get(sync_field) == right_metadata.get(sync_field):
        drift_reasons.append("topology_intervention_not_observed")
    left_sync = left_metadata.get(sync_field)
    right_sync = right_metadata.get(sync_field)
    if not isinstance(left_sync, Mapping) or not isinstance(right_sync, Mapping):
        drift_reasons.append("invalid_synchronization_fingerprint")
    else:
        observed_classes = set(left_sync.get("difference_classes") or ()) | set(
            right_sync.get("difference_classes") or ()
        )
        if not observed_classes <= set(group.allowed_sass_difference_classes):
            drift_reasons.append("disallowed_sass_difference_class")

    counter_checks = []
    for path in group.invariant_counter_paths:
        left_value = _resolve_path(left_bundle, path)
        right_value = _resolve_path(right_bundle, path)
        if not (
            isinstance(left_value, (int, float))
            and not isinstance(left_value, bool)
            and isinstance(right_value, (int, float))
            and not isinstance(right_value, bool)
        ):
            status = "not_measured"
            relative_difference = None
            missing_reasons.append(f"counter_not_measured:{path}")
        else:
            relative_difference = _relative_difference(
                float(left_value), float(right_value)
            )
            status = (
                "pass"
                if relative_difference <= group.change_threshold_fraction
                else "drift"
            )
            if status == "drift":
                drift_reasons.append(f"counter_drift:{path}")
        counter_checks.append(
            {
                "path": path,
                "role": "hard_invariant",
                "blocking": True,
                "baseline": left_value,
                "variant": right_value,
                "relative_difference": relative_difference,
                "threshold_fraction": group.change_threshold_fraction,
                "status": status,
            }
        )
    mediation_counter_checks = []
    for metric in aggregate_metrics:
        lowered_metric = metric.lower()
        groups = [
            name
            for name, predicate in BARRIER_MEDIATION_COUNTER_REQUIREMENTS.items()
            if predicate(lowered_metric)
        ]
        if not groups:
            continue
        path = f"aggregate_counters.metrics.{metric}"
        left_value = _resolve_path(left_bundle, path)
        right_value = _resolve_path(right_bundle, path)
        measured = (
            isinstance(left_value, (int, float))
            and not isinstance(left_value, bool)
            and isinstance(right_value, (int, float))
            and not isinstance(right_value, bool)
        )
        mediation_counter_checks.append(
            {
                "path": path,
                "groups": groups,
                "role": "mediation_evidence",
                "blocking": False,
                "baseline": left_value,
                "variant": right_value,
                "relative_difference": (
                    _relative_difference(float(left_value), float(right_value))
                    if measured
                    else None
                ),
                "status": "measured" if measured else "not_measured",
            }
        )
    cross_lane_diagnostic_reasons = {}
    for point_id, bundle in (
        (baseline["point_id"], left_bundle),
        (variant["point_id"], right_bundle),
    ):
        reasons = list(bundle.get("identity_check", {}).get("reasons") or ())
        blocking_reasons = [
            reason
            for reason in reasons
            if not reason.endswith(
                "subject_metadata_mismatch:synchronization_instruction_fingerprint"
            )
        ]
        if blocking_reasons:
            missing_reasons.extend(
                f"{point_id}:cross_lane:{reason}" for reason in blocking_reasons
            )
        if reasons:
            cross_lane_diagnostic_reasons[point_id] = reasons

    sass_diagnostics = {
        "baseline_non_sync_digest": (
            left_sync.get("non_sync_digest")
            if isinstance(left_sync, Mapping)
            else None
        ),
        "variant_non_sync_digest": (
            right_sync.get("non_sync_digest")
            if isinstance(right_sync, Mapping)
            else None
        ),
        "non_sync_digest_matches": (
            left_sync.get("non_sync_digest")
            == right_sync.get("non_sync_digest")
            if isinstance(left_sync, Mapping)
            and isinstance(right_sync, Mapping)
            else None
        ),
        "role": "non_blocking_compiler_artifact_diagnostic",
    }

    if drift_reasons:
        status = "coupled_fixture"
    elif missing_reasons:
        status = "not_measured"
    else:
        status = "pass"
    return {
        "status": status,
        "baseline_point_id": baseline["point_id"],
        "variant_point_id": variant["point_id"],
        "axis_differences": axis_differences,
        "identity_differences": identity_differences,
        "allowed_identity_differences": list(
            group.allowed_identity_differences
        ),
        "unexpected_identity_differences": unexpected_identity,
        "metadata_differences": metadata_differences,
        "hard_subject_metadata_differences": hard_metadata_differences,
        "allowed_subject_metadata_differences": list(
            group.allowed_subject_metadata_differences
        ),
        "unexpected_subject_metadata_differences": unexpected_metadata,
        "counter_checks": counter_checks,
        "hard_invariant_counter_checks": counter_checks,
        "mediation_counter_checks": mediation_counter_checks,
        "cross_lane_diagnostic_reasons": cross_lane_diagnostic_reasons,
        "sass_diagnostics": sass_diagnostics,
        "drift_reasons": list(dict.fromkeys(drift_reasons)),
        "missing_reasons": list(dict.fromkeys(missing_reasons)),
    }


def build_barrier_topology_report(
    recipe: MechanismRecipe,
    point_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Reduce topology pairs without mixing counter and PC-sample units."""

    if recipe.mechanism != "barrier_topology_release":
        raise ValueError("barrier topology report requires a topology recipe")
    pairs = []
    for group in recipe.interaction_groups:
        baseline = point_results[group.point_ids[0]]
        variant = point_results[group.point_ids[1]]
        controls = _pair_control_scorecard(
            group,
            baseline,
            variant,
            aggregate_metrics=recipe.aggregate_metrics,
        )
        timing = _paired_timing_difference(
            baseline["bundle"]["timing"], variant["bundle"]["timing"]
        )
        timing_quality_reasons = []
        for point in (baseline, variant):
            point_timing = point["bundle"]["timing"]
            if len(point_timing.get("process_samples") or ()) < 7:
                timing_quality_reasons.append(
                    f"{point['point_id']}:fewer_than_seven_process_repeats"
                )
            if float(point_timing.get("process_cv", math.inf)) > 0.02:
                timing_quality_reasons.append(
                    f"{point['point_id']}:process_cv_exceeds_2_percent"
                )
        timing["quality_status"] = (
            "qualifying" if not timing_quality_reasons else "diagnostic"
        )
        timing["diagnostic_reasons"] = timing_quality_reasons
        panel = str(baseline["axes"]["panel"])
        intervals = _interval_order_assessment(
            panel, {point["point_id"]: point for point in (baseline, variant)}
        )
        pc_localization = {
            point["point_id"]: point.get("pc_localization")
            for point in (baseline, variant)
        }

        if controls["status"] == "coupled_fixture":
            classification = "coupled_fixture"
        elif controls["status"] != "pass":
            classification = "not_measured"
        elif timing["quality_status"] != "qualifying":
            classification = "not_measured"
        elif panel == "BT-CAUSAL":
            classification = intervals["status"]
        else:
            baseline_scale = float(timing.get("baseline_mean_us") or 0.0)
            half_width = float(timing.get("confidence_half_width") or 0.0)
            allowed = max(0.02 * baseline_scale, half_width)
            difference = abs(float(timing.get("mean_difference") or 0.0))
            if difference <= allowed:
                classification = "supported"
            elif any(
                localization
                and localization.get("status") == "qualifying"
                and localization.get("dominant_region")
                in {"mbarrier_poll_fast", "mbarrier_poll_retry", "wgmma_wait"}
                for localization in pc_localization.values()
            ):
                classification = "missing_sync_protocol_service"
            else:
                classification = "falsified"

        edge_rows = []
        left_nodes = baseline["causal_evidence"]["nodes"]
        right_nodes = variant["causal_evidence"]["nodes"]
        for source, target in BARRIER_TOPOLOGY_CAUSAL_EDGES:
            source_changed = _changed(
                _numeric_measurements(left_nodes.get(source)),
                _numeric_measurements(right_nodes.get(source)),
                group.change_threshold_fraction,
            )
            target_changed = _changed(
                _numeric_measurements(left_nodes.get(target)),
                _numeric_measurements(right_nodes.get(target)),
                group.change_threshold_fraction,
            )
            if controls["status"] == "coupled_fixture":
                status = "coupled_fixture"
            elif controls["status"] != "pass":
                status = "not_measured"
            elif source_changed is None or target_changed is None:
                status = "not_measured"
            elif source_changed and target_changed:
                status = "supported"
            elif source_changed:
                status = "falsified"
            else:
                status = "not_measured"
            edge_rows.append(
                {
                    "source": source,
                    "target": target,
                    "source_changed": source_changed,
                    "target_changed": target_changed,
                    "status": status,
                }
            )
        pairs.append(
            {
                "group_id": group.group_id,
                "panel": panel,
                "point_ids": list(group.point_ids),
                "control_scorecard": controls,
                "cuda_event_timing": timing,
                "interval_order": intervals,
                "aggregate_stalls": {
                    point["point_id"]: point["bundle"][
                        "aggregate_counters"
                    ].get("structural_stall_histogram")
                    for point in (baseline, variant)
                },
                "pc_localization": pc_localization,
                "causal_edges": edge_rows,
                "classification": classification,
            }
        )
    return {
        "schema_version": 1,
        "kind": "barrier_topology_hardware_findings",
        "latency_oracle": "cuda_events",
        "aggregate_stall_unit": "coherent_ncu_metric_family",
        "pc_localization_unit": "pc_sample_count",
        "pairs": pairs,
        "interpretation_boundary": (
            "Amora reports measured topology evidence; portable P3 semantic "
            "interpretation belongs to the consumer."
        ),
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

    if recipe.mechanism == "barrier_topology_release":
        return build_barrier_topology_report(recipe, point_results)

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


def _write_new_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def _topology_consumer_rows(
    point_results: Mapping[str, Mapping[str, Any]],
    findings: Mapping[str, Any],
) -> dict[str, tuple[list[dict[str, Any]], tuple[str, ...]]]:
    timing_rows = []
    interval_rows = []
    aggregate_rows = []
    pc_rows = []
    control_rows = []
    for point_id, result in point_results.items():
        timing = result["bundle"]["timing"]
        timing_rows.append(
            {
                "point_id": point_id,
                "split": result["split"],
                "panel": result["axes"]["panel"],
                "topology": result["axes"]["topology"],
                "process_repeat_count": len(timing.get("process_samples") or ()),
                "median_us": timing.get("median_us"),
                "p05_us": timing.get("p05_us"),
                "p95_us": timing.get("p95_us"),
                "process_cv": timing.get("process_cv"),
                "qualification_status": result["qualification_status"],
            }
        )
        interval_payload = result["bundle"].get("device_intervals") or {}
        for interval in interval_payload.get("intervals") or ():
            interval_rows.append(
                {
                    "point_id": point_id,
                    "panel": result["axes"]["panel"],
                    "topology": result["axes"]["topology"],
                    "name": interval.get("name"),
                    "start_ns": interval.get("start_ns"),
                    "end_ns": interval.get("end_ns"),
                    "duration_ns": interval.get("duration_ns"),
                    "evidence_status": interval_payload.get("evidence_status"),
                }
            )
        histogram = result["bundle"]["aggregate_counters"].get(
            "structural_stall_histogram"
        ) or {}
        reason_values = histogram.get("reason_values") or {}
        reason_fractions = histogram.get("reason_fractions") or {}
        for reason in sorted(reason_values):
            aggregate_rows.append(
                {
                    "point_id": point_id,
                    "panel": result["axes"]["panel"],
                    "topology": result["axes"]["topology"],
                    "reason": reason,
                    "raw_unit": histogram.get("raw_unit"),
                    "reason_value": reason_values[reason],
                    "structural_denominator": histogram.get("denominator"),
                    "structural_fraction": reason_fractions.get(reason),
                }
            )
        localization = result.get("pc_localization") or {}
        region_by_offset = {
            offset: region
            for region, offsets in result.get("pc_region_map", {}).items()
            for offset in offsets
        }
        pc_payload = result["bundle"].get("pc_sampling") or {}
        for offset_row in pc_payload.get("by_offset") or ():
            pc_rows.append(
                {
                    "point_id": point_id,
                    "panel": result["axes"]["panel"],
                    "topology": result["axes"]["topology"],
                    "function": offset_row.get("function"),
                    "pc_offset": offset_row.get("pc_offset"),
                    "sass_opcode": offset_row.get("sass_opcode"),
                    "sass_instruction": offset_row.get("sass_instruction"),
                    "region": region_by_offset.get(offset_row.get("pc_offset")),
                    "support_count": offset_row.get("support_count"),
                    "repeat_support_counts": canonical_json(
                        offset_row.get("repeat_support_counts") or []
                    ),
                    "stall_counts": canonical_json(
                        offset_row.get("stall_counts") or {}
                    ),
                    "localization_status": localization.get("status"),
                }
            )
    for pair in findings.get("pairs") or ():
        controls = pair["control_scorecard"]
        control_rows.append(
            {
                "group_id": pair["group_id"],
                "panel": pair["panel"],
                "baseline_point_id": controls["baseline_point_id"],
                "variant_point_id": controls["variant_point_id"],
                "status": controls["status"],
                "classification": pair["classification"],
                "drift_reasons": canonical_json(controls["drift_reasons"]),
                "missing_reasons": canonical_json(controls["missing_reasons"]),
            }
        )
    return {
        "cuda_event_timing.csv": (
            timing_rows,
            (
                "point_id", "split", "panel", "topology",
                "process_repeat_count", "median_us", "p05_us", "p95_us",
                "process_cv", "qualification_status",
            ),
        ),
        "device_intervals.csv": (
            interval_rows,
            (
                "point_id", "panel", "topology", "name", "start_ns",
                "end_ns", "duration_ns", "evidence_status",
            ),
        ),
        "aggregate_stalls.csv": (
            aggregate_rows,
            (
                "point_id", "panel", "topology", "reason", "raw_unit",
                "reason_value", "structural_denominator", "structural_fraction",
            ),
        ),
        "pc_samples_by_offset.csv": (
            pc_rows,
            (
                "point_id", "panel", "topology", "function", "pc_offset",
                "sass_opcode", "sass_instruction", "region", "support_count",
                "repeat_support_counts", "stall_counts", "localization_status",
            ),
        ),
        "fixture_control_scorecard.csv": (
            control_rows,
            (
                "group_id", "panel", "baseline_point_id", "variant_point_id",
                "status", "classification", "drift_reasons", "missing_reasons",
            ),
        ),
    }


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
            for pc_report in point_artifact.get("pc_reports") or ():
                artifact_entries.append(pc_report)
    if isinstance(value.get("frozen_recipe"), Mapping):
        artifact_entries.append(value["frozen_recipe"])
    if isinstance(value.get("mediation"), Mapping):
        artifact_entries.append(value["mediation"])
    for compact_artifact in (value.get("compact_artifacts") or {}).values():
        artifact_entries.append(compact_artifact)
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
        required_subject_metadata = (
            BARRIER_REQUIRED_SUBJECT_METADATA
            if recipe.mechanism == "barrier_topology_release"
            else REQUIRED_SUBJECT_METADATA
        )
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
            pc_sampling_repeats=(
                point.pc_sampling_repeats or recipe.pc_sampling_repeats
            ),
            expected_launch_batch_size=point.expected_launch_batch_size,
            expected_cache_protocol=(
                point.expected_cache_protocol or point.axes.get("cache_protocol")
            ),
            expected_max_interval_overhead_percent=(
                point.expected_max_interval_overhead_percent
            ),
            expected_measurement_axes=point.axes,
            expected_measurement_context={},
            required_identity_fields=recipe.required_identity_fields,
            required_subject_metadata_fields=tuple(
                sorted(required_subject_metadata)
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
        pc_localization = (
            localize_pc_regions(bundle.pc_sampling, point.pc_region_map)
            if recipe.mechanism == "barrier_topology_release"
            else None
        )
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
        for field_name in required_subject_metadata:
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
            "pc_region_map": {
                region: list(offsets)
                for region, offsets in point.pc_region_map.items()
            },
            "qualification_status": (
                "diagnostic" if diagnostic_reasons else "qualifying"
            ),
            "diagnostic_reasons": list(dict.fromkeys(diagnostic_reasons)),
            "bundle": bundle.to_dict(),
            "causal_evidence": causal,
            "pc_localization": pc_localization,
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
            report_entries = []
            for provenance in pc_provenance.get("reports") or ():
                report_path = provenance.get("report_path")
                report_hash = provenance.get("report_sha256")
                if not isinstance(report_path, str) or not isinstance(report_hash, str):
                    continue
                candidate = Path(report_path)
                if candidate.is_file():
                    report_entries.append(
                        {
                            "path": str(candidate.relative_to(run_dir)),
                            "sha256": report_hash,
                        }
                    )
            if report_entries:
                artifacts[point.point_id]["pc_reports"] = report_entries

    mediation = build_mediation_report(recipe, point_results)
    mediation_path = run_dir / "mediation.json"
    _write_new_json(mediation_path, mediation)
    compact_artifacts = {}
    if recipe.mechanism == "barrier_topology_release":
        findings_path = run_dir / "hardware_findings.json"
        _write_new_json(findings_path, mediation)
        compact_artifacts[findings_path.name] = {
            "path": findings_path.name,
            "sha256": hashlib.sha256(findings_path.read_bytes()).hexdigest(),
        }
        for filename, (rows, fields) in _topology_consumer_rows(
            point_results, mediation
        ).items():
            compact_path = run_dir / filename
            _write_new_csv(compact_path, rows, fields)
            compact_artifacts[filename] = {
                "path": filename,
                "sha256": hashlib.sha256(compact_path.read_bytes()).hexdigest(),
            }
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
        "compact_artifacts": compact_artifacts,
        "split_counts": {
            split: sum(point.split == split for point in recipe.points)
            for split in sorted(SPLITS)
        },
        "capabilities": capabilities.to_dict(),
    }
    manifest["run_digest"] = _digest(manifest)
    _write_new_json(run_dir / "manifest.json", manifest)
    return run_dir
