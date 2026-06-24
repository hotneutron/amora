"""L1 conflict-set associativity probe (P1).

Grows the number of cache lines mapping to the same set and watches for the
latency knee where the working set exceeds the set associativity. Per the P1
methodology, associativity is reported as a *bounded* fit because cache
indexing, replacement, and hashing can mimic one another.
"""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.runner import CudaUnavailable, run_kernel
from amora.backends.nvidia.sass import SassExpectation
from amora.probes.nvidia.baseline._sources import (
    apply_sass_gating,
    downgrade_fit,
    soften_uncertainty,
    source_descriptor,
)
from amora.schemas.evidence import EvidenceTier, FitStatus, UncertaintyCategory
from amora.schemas.results import (
    BackendInterpretation,
    LaunchDescriptor,
    NormalizedMeasurement,
    ProbeIdentity,
    ProbeResult,
    RawObservation,
    SimulatorEstimate,
    ToolContext,
)


PROBE_ID = "l1_cache.conflict_sets"
SOURCE = Path(__file__).with_name("conflict_sets.cu")

# The timed loop must hit global memory (LDG) without shared or local spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_l1_conflict",
    required_opcodes={"LDG": 1},
    forbidden_opcodes=("LDS", "STL"),
)


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def _associativity_knee(sweep: list[dict]) -> int | None:
    """Return the way-count just before the first >40% latency jump."""

    for i in range(1, len(sweep)):
        prev = float(sweep[i - 1]["cycles_per_load"])
        cur = float(sweep[i]["cycles_per_load"])
        if prev > 0 and cur > prev * 1.4:
            return int(sweep[i - 1]["ways"])
    return None


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    src_descriptor = source_descriptor(SOURCE)
    try:
        result = run_kernel(SOURCE, capabilities=capabilities, timeout=60, expectation=EXPECTATION)
    except CudaUnavailable as exc:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"L1 conflict-set probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    sweep = list(payload["sweep"])
    assoc = _associativity_knee(sweep)
    fit = FitStatus.BOUNDED if assoc else FitStatus.UNDERCONSTRAINED

    # SASS gating: reject if the timed loop is not a global-load conflict sweep.
    sass = result.sass_validation
    decision, fit, uncertainty, downgrade_reason = apply_sass_gating(
        sass, EXPECTATION, fit, UncertaintyCategory.BOUNDED_RANGE
    )
    if decision == "reject":
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"SASS validation rejected the measurement: {sass.reason}",
                tool_context=_tool_context(capabilities),
                raw_values={
                    "registered_source": src_descriptor,
                    "sass": sass.to_dict(),
                },
            )
        ]

    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    if sass is not None:
        values["sass"] = sass.to_dict()
    assumptions = [
        "ring of same-set lines grown one way at a time at a fixed power-of-two stride",
        "latency knee marks where the conflict set exceeds the effective associativity",
        "associativity is bounded: indexing/replacement/hashing can mimic the same curve",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(
                probe_id=PROBE_ID,
                binary_hash=result.binary_sha256,
                disassembly_hash=sass.disassembly_hash if sass else None,
            ),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(grid=(1, 1, 1), block=(32, 1, 1), mode="kernel"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "sweep_points": len(sweep),
                    "effective_associativity": assoc,
                    "stride_bytes": int(payload.get("stride_bytes", 0)),
                },
                units={"effective_associativity": "ways"},
                source="amora.probes.nvidia.baseline.l1_cache.conflict_sets",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="l1_effective_associativity",
                value=assoc,
                unit="ways",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="l1_effective_associativity",
                interpretation={
                    "nvidia_backend": "effective L1 associativity bounded by the conflict-set latency knee",
                },
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason
                if downgrade_reason is not None
                else (None if assoc else "no clear associativity knee detected"),
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="l1d_cache_assoc",
                value=assoc,
                unit="ways",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="conflict-set latency knee → simulator L1 associativity (bounded)",
                assumptions=assumptions,
            ),
        )
    ]
