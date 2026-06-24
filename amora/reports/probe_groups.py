"""Pluggable probe-grouping registry.

The report summary groups probes into a small number of *probe groups* and
renders one table per (family, group). Groups combine probes across priority
tiers and target roughly 6-10 probes each so the per-group tables stay readable
while the total table count stays small.

Grouping is vendor-specific and pluggable: each vendor registers an ordered list
of :class:`ProbeGroup` definitions. When a vendor has no registration (or a probe
is not listed), grouping falls back to the prefix of the ``probe_id`` before the
first ``.``.

To add a vendor: call :func:`register_vendor_groups`. To add a probe to an
existing vendor: append its ``probe_id`` to the relevant group's ``probes``
tuple.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProbeGroup:
    """A named group of related probes (spanning priority tiers)."""

    key: str            # stable slug, e.g. "onchip_memory"
    label: str          # human label, e.g. "On-chip Memory"
    probes: tuple[str, ...] = field(default_factory=tuple)


# vendor -> ordered probe groups
_VENDOR_GROUPS: dict[str, tuple[ProbeGroup, ...]] = {}


def register_vendor_groups(vendor: str, groups: tuple[ProbeGroup, ...]) -> None:
    _VENDOR_GROUPS[vendor] = groups


def vendor_groups(vendor: str) -> tuple[ProbeGroup, ...]:
    return _VENDOR_GROUPS.get(vendor, ())


def group_for_probe(vendor: str, probe_id: str) -> tuple[str, str]:
    """Return (group_key, group_label) for a probe.

    Falls back to the prefix before the first ``.`` when the vendor has no
    registration or the probe is not explicitly listed.
    """
    for group in _VENDOR_GROUPS.get(vendor, ()):
        if probe_id in group.probes:
            return group.key, group.label
    prefix = probe_id.split(".", 1)[0]
    return prefix, prefix.replace("_", " ").title()


# --------------------------------------------------------------------------- #
# NVIDIA registration
#
# 30 probes (baseline P0 + P1 + P2 + P3) combined across tiers into 5 thematic
# groups of 6-8 probes each.
# --------------------------------------------------------------------------- #


register_vendor_groups(
    "nvidia",
    (
        ProbeGroup(
            "compute_scheduling", "Compute & Scheduling",
            (
                # topology / occupancy
                "topology.device_attributes",
                "topology.occupancy",
                "topology.persistent_cta",
                # arithmetic
                "arithmetic_latency.dependent_chain",
                "arithmetic_throughput.independent_chains",
                # scheduler / issue
                "scheduler_policy.ready_warps",
                "scheduler_policy.mixed_issue",
                "scheduler_policy.analyze",
            ),
        ),
        ProbeGroup(
            "register_tensor_sync", "Register, Tensor & Sync",
            (
                "register_file.register_bank_sweep",
                "register_file.register_latency",
                "register_file.analyze",
                "tensor_core.mma_latency",
                "tensor_core.mma_throughput",
                "synchronization.barrier_latency",
                "synchronization.fence_latency",
            ),
        ),
        ProbeGroup(
            "onchip_memory", "On-chip Memory",
            (
                "shared_memory.pointer_chase",
                "shared_memory.bank_stride",
                "shared_memory.analyze",
                "l1_cache.pointer_chase",
                "l1_cache.working_set",
                "l1_cache.conflict_sets",
                "l1_cache.analyze",
                "l2_cache.pointer_chase",
            ),
        ),
        ProbeGroup(
            "global_memory", "Global Memory & DRAM",
            (
                "memory_pipeline.lane_patterns",
                "memory_pipeline.outstanding_requests",
                "memory_pipeline.analyze",
                "global_memory.streaming",
                "global_memory.partition_sweep",
                "global_memory.row_policy_sweep",
                "global_memory.analyze",
            ),
        ),
        ProbeGroup(
            "transfer_fabric", "Transfer & Interconnect",
            (
                "tma_copy.async_copy_latency",
                "tma_copy.tma_transfer_sweep",
                "tma_copy.analyze",
                "interconnect.address_mapping",
                "interconnect.injection_rate",
                "interconnect.analyze",
            ),
        ),
    ),
)
