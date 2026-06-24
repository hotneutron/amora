"""Curated published NVIDIA device facts and capability gating.

A small trust-and-verify table of per-architecture published specifications,
keyed by the device-name patterns AMORA already uses for report grouping. These
facts are *anchors*: probes can cross-check runtime metadata against them, and
feature flags (e.g. tensor cores, TMA) let architecture-specific probes gate
cleanly on older hardware instead of producing meaningless results.

The table is intentionally conservative — only widely published, stable specs
are included. Unknown devices resolve to ``None`` and gating then defaults to
"allow" (probes fall back to their own evidence) so an unseen GPU is never
silently mis-gated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArchFacts:
    """Published facts for one GPU architecture / SKU family."""

    family: str                       # e.g. "hopper"
    model: str                        # e.g. "h100"
    compute_capability: tuple[int, int]
    sm_count: int | None = None
    l2_cache_mb: float | None = None
    memory_bandwidth_gbps: float | None = None
    shared_memory_per_sm_kb: int | None = None
    # Feature flags consumed by capability gating.
    features: frozenset[str] = field(default_factory=frozenset)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "model": self.model,
            "compute_capability": f"{self.compute_capability[0]}.{self.compute_capability[1]}",
            "sm_count": self.sm_count,
            "l2_cache_mb": self.l2_cache_mb,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "shared_memory_per_sm_kb": self.shared_memory_per_sm_kb,
            "features": sorted(self.features),
        }


# Feature tokens.
TENSOR_CORE = "tensor_core"
ASYNC_COPY = "async_copy"      # cp.async / LDGSTS (Ampere+)
TMA = "tma"                    # Tensor Memory Accelerator (Hopper+)
FP8 = "fp8"                    # FP8 tensor cores (Hopper+)


# (device-name regex) -> ArchFacts. First match wins; order specific -> generic.
_FACTS: list[tuple[re.Pattern[str], ArchFacts]] = [
    (re.compile(r"\bB200\b", re.I), ArchFacts(
        "blackwell", "b200", (10, 0), sm_count=148, l2_cache_mb=50.0,
        memory_bandwidth_gbps=8000.0, shared_memory_per_sm_kb=228,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, TMA, FP8}))),
    (re.compile(r"\bH200\b", re.I), ArchFacts(
        "hopper", "h200", (9, 0), sm_count=132, l2_cache_mb=50.0,
        memory_bandwidth_gbps=4800.0, shared_memory_per_sm_kb=228,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, TMA, FP8}))),
    (re.compile(r"\bH100\b", re.I), ArchFacts(
        "hopper", "h100", (9, 0), sm_count=132, l2_cache_mb=50.0,
        memory_bandwidth_gbps=3350.0, shared_memory_per_sm_kb=228,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, TMA, FP8}))),
    (re.compile(r"\bH20\b", re.I), ArchFacts(
        "hopper", "h20", (9, 0), sm_count=78, l2_cache_mb=60.0,
        memory_bandwidth_gbps=4000.0, shared_memory_per_sm_kb=228,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, TMA, FP8}))),
    (re.compile(r"\bL40S?\b", re.I), ArchFacts(
        "ada", "l40s", (8, 9), sm_count=142, l2_cache_mb=96.0,
        memory_bandwidth_gbps=864.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, FP8}))),
    (re.compile(r"\bA100\b", re.I), ArchFacts(
        "ampere", "a100", (8, 0), sm_count=108, l2_cache_mb=40.0,
        memory_bandwidth_gbps=2039.0, shared_memory_per_sm_kb=164,
        features=frozenset({TENSOR_CORE, ASYNC_COPY}))),
    (re.compile(r"\bA800\b", re.I), ArchFacts(
        "ampere", "a800", (8, 0), sm_count=108, l2_cache_mb=40.0,
        memory_bandwidth_gbps=2039.0, shared_memory_per_sm_kb=164,
        features=frozenset({TENSOR_CORE, ASYNC_COPY}))),
    (re.compile(r"\bA30\b", re.I), ArchFacts(
        "ampere", "a30", (8, 0), sm_count=56, l2_cache_mb=24.0,
        memory_bandwidth_gbps=933.0, shared_memory_per_sm_kb=164,
        features=frozenset({TENSOR_CORE, ASYNC_COPY}))),
    (re.compile(r"\bA10\b", re.I), ArchFacts(
        "ampere", "a10", (8, 6), sm_count=72, l2_cache_mb=6.0,
        memory_bandwidth_gbps=600.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY}))),
    # Ampere consumer — Ti variants before base models (first match wins)
    (re.compile(r"\bRTX 3090 Ti\b", re.I), ArchFacts(
        "ampere", "rtx-3090-ti", (8, 6), sm_count=84, l2_cache_mb=6.0,
        memory_bandwidth_gbps=1008.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY}))),
    (re.compile(r"\bRTX 3090\b", re.I), ArchFacts(
        "ampere", "rtx-3090", (8, 6), sm_count=82, l2_cache_mb=6.0,
        memory_bandwidth_gbps=936.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY}))),
    (re.compile(r"\bRTX 3080 Ti\b", re.I), ArchFacts(
        "ampere", "rtx-3080-ti", (8, 6), sm_count=80, l2_cache_mb=6.0,
        memory_bandwidth_gbps=912.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY}))),
    (re.compile(r"\bRTX 3080\b", re.I), ArchFacts(
        "ampere", "rtx-3080", (8, 6), sm_count=68, l2_cache_mb=5.0,
        memory_bandwidth_gbps=760.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY}))),
    (re.compile(r"\bRTX 3070 Ti\b", re.I), ArchFacts(
        "ampere", "rtx-3070-ti", (8, 6), sm_count=48, l2_cache_mb=4.0,
        memory_bandwidth_gbps=608.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY}))),
    (re.compile(r"\bRTX 3070\b", re.I), ArchFacts(
        "ampere", "rtx-3070", (8, 6), sm_count=46, l2_cache_mb=4.0,
        memory_bandwidth_gbps=448.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY}))),
    (re.compile(r"\bRTX 3060 Ti\b", re.I), ArchFacts(
        "ampere", "rtx-3060-ti", (8, 6), sm_count=38, l2_cache_mb=3.0,
        memory_bandwidth_gbps=448.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY}))),
    (re.compile(r"\bRTX 3060\b", re.I), ArchFacts(
        "ampere", "rtx-3060", (8, 6), sm_count=28, l2_cache_mb=3.0,
        memory_bandwidth_gbps=360.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY}))),
    # Ada consumer — Ti/Super variants before base models
    (re.compile(r"\bRTX 4090\b", re.I), ArchFacts(
        "ada", "rtx-4090", (8, 9), sm_count=128, l2_cache_mb=72.0,
        memory_bandwidth_gbps=1008.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, FP8}))),
    (re.compile(r"\bRTX 4080 Super\b", re.I), ArchFacts(
        "ada", "rtx-4080-super", (8, 9), sm_count=80, l2_cache_mb=64.0,
        memory_bandwidth_gbps=736.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, FP8}))),
    (re.compile(r"\bRTX 4080\b", re.I), ArchFacts(
        "ada", "rtx-4080", (8, 9), sm_count=76, l2_cache_mb=64.0,
        memory_bandwidth_gbps=716.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, FP8}))),
    (re.compile(r"\bRTX 4070 Ti Super\b", re.I), ArchFacts(
        "ada", "rtx-4070-ti-super", (8, 9), sm_count=66, l2_cache_mb=48.0,
        memory_bandwidth_gbps=672.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, FP8}))),
    (re.compile(r"\bRTX 4070 Ti\b", re.I), ArchFacts(
        "ada", "rtx-4070-ti", (8, 9), sm_count=60, l2_cache_mb=48.0,
        memory_bandwidth_gbps=504.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, FP8}))),
    (re.compile(r"\bRTX 4070 Super\b", re.I), ArchFacts(
        "ada", "rtx-4070-super", (8, 9), sm_count=56, l2_cache_mb=48.0,
        memory_bandwidth_gbps=504.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, FP8}))),
    (re.compile(r"\bRTX 4070\b", re.I), ArchFacts(
        "ada", "rtx-4070", (8, 9), sm_count=46, l2_cache_mb=36.0,
        memory_bandwidth_gbps=504.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, FP8}))),
    (re.compile(r"\bRTX 4060 Ti\b", re.I), ArchFacts(
        "ada", "rtx-4060-ti", (8, 9), sm_count=34, l2_cache_mb=32.0,
        memory_bandwidth_gbps=288.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, FP8}))),
    (re.compile(r"\bRTX 4060\b", re.I), ArchFacts(
        "ada", "rtx-4060", (8, 9), sm_count=24, l2_cache_mb=24.0,
        memory_bandwidth_gbps=272.0, shared_memory_per_sm_kb=100,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, FP8}))),
    # Blackwell consumer — Ti variants before base models
    (re.compile(r"\bRTX 5090\b", re.I), ArchFacts(
        "blackwell", "rtx-5090", (12, 0), sm_count=170, l2_cache_mb=112.0,
        memory_bandwidth_gbps=1792.0, shared_memory_per_sm_kb=128,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, TMA, FP8}))),
    (re.compile(r"\bRTX 5080\b", re.I), ArchFacts(
        "blackwell", "rtx-5080", (12, 0), sm_count=84, l2_cache_mb=64.0,
        memory_bandwidth_gbps=960.0, shared_memory_per_sm_kb=128,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, TMA, FP8}))),
    (re.compile(r"\bRTX 5070 Ti\b", re.I), ArchFacts(
        "blackwell", "rtx-5070-ti", (12, 0), sm_count=70, l2_cache_mb=48.0,
        memory_bandwidth_gbps=896.0, shared_memory_per_sm_kb=128,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, TMA, FP8}))),
    (re.compile(r"\bRTX 5070\b", re.I), ArchFacts(
        "blackwell", "rtx-5070", (12, 0), sm_count=48, l2_cache_mb=40.0,
        memory_bandwidth_gbps=672.0, shared_memory_per_sm_kb=128,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, TMA, FP8}))),
    (re.compile(r"\bRTX 5060 Ti\b", re.I), ArchFacts(
        "blackwell", "rtx-5060-ti", (12, 0), sm_count=36, l2_cache_mb=32.0,
        memory_bandwidth_gbps=448.0, shared_memory_per_sm_kb=128,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, TMA, FP8}))),
    (re.compile(r"\bRTX 5060\b", re.I), ArchFacts(
        "blackwell", "rtx-5060", (12, 0), sm_count=24, l2_cache_mb=24.0,
        memory_bandwidth_gbps=288.0, shared_memory_per_sm_kb=128,
        features=frozenset({TENSOR_CORE, ASYNC_COPY, TMA, FP8}))),
    (re.compile(r"\bV100\b", re.I), ArchFacts(
        "volta", "v100", (7, 0), sm_count=80, l2_cache_mb=6.0,
        memory_bandwidth_gbps=900.0, shared_memory_per_sm_kb=96,
        features=frozenset({TENSOR_CORE}))),
    (re.compile(r"\bT4\b", re.I), ArchFacts(
        "turing", "t4", (7, 5), sm_count=40, l2_cache_mb=4.0,
        memory_bandwidth_gbps=320.0, shared_memory_per_sm_kb=64,
        features=frozenset({TENSOR_CORE}))),
]


def facts_for_device(device_name: str | None) -> ArchFacts | None:
    """Return published facts for a device name, or None if unknown."""

    if not device_name:
        return None
    for pattern, facts in _FACTS:
        if pattern.search(device_name):
            return facts
    return None


def facts_for_capabilities(capabilities) -> ArchFacts | None:
    """Return published facts for the primary device of a capability record."""

    devices = getattr(capabilities, "devices", None) or []
    if not devices:
        return None
    return facts_for_device(devices[0].name)


def supports_feature(capabilities, feature: str) -> bool | None:
    """Whether the primary device is known to support ``feature``.

    Returns True/False when the device is in the table, or None when the device
    is unknown (callers should then *allow* the probe and rely on its own
    evidence rather than gating on incomplete data).
    """

    facts = facts_for_capabilities(capabilities)
    if facts is None:
        return None
    return feature in facts.features
