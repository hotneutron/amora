"""L1 working-set sweep probe (P1).

Sweeps the pointer-chase working-set size and reports the latency-vs-size curve.
Capacity knees in the curve mark L1 -> L2 -> DRAM transitions. Capacity is
reported as a bounded effective range (the first knee), per the P1 methodology.
"""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.runner import CudaUnavailable, run_kernel
from amora.probes.nvidia.baseline._sources import source_descriptor
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


PROBE_ID = "l1_cache.working_set"
SOURCE = Path(__file__).with_name("working_set.cu")


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def _first_knee(sweep: list[dict]) -> tuple[int, int] | None:
    """Return (last_flat_kb, first_jump_kb) at the first >40% latency jump."""

    for i in range(1, len(sweep)):
        prev = float(sweep[i - 1]["cycles_per_load"])
        cur = float(sweep[i]["cycles_per_load"])
        if prev > 0 and cur > prev * 1.4:
            return int(sweep[i - 1]["working_set_kb"]), int(sweep[i]["working_set_kb"])
    return None


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    src_descriptor = source_descriptor(SOURCE)
    try:
        result = run_kernel(SOURCE, capabilities=capabilities, timeout=60)
    except CudaUnavailable as exc:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"L1 working-set sweep could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    sweep = list(payload["sweep"])
    knee = _first_knee(sweep)
    capacity_low = knee[0] if knee else None
    capacity_high = knee[1] if knee else None
    capacity_value = {"effective_l1_kb_low": capacity_low, "effective_l1_kb_high": capacity_high}
    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    assumptions = [
        "dependent pointer-chase latency swept across geometric working-set sizes",
        "first >40% latency jump marks the effective L1 capacity knee",
        "capacity is reported as a bounded range, not an exact scalar",
    ]
    fit = FitStatus.BOUNDED if knee else FitStatus.UNDERCONSTRAINED
    return [
        ProbeResult(
            identity=ProbeIdentity(probe_id=PROBE_ID, binary_hash=result.binary_sha256),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(grid=(1, 1, 1), block=(32, 1, 1), mode="kernel"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "sweep_points": len(sweep),
                    "effective_l1_kb_low": capacity_low,
                    "effective_l1_kb_high": capacity_high,
                },
                units={"effective_l1_kb_low": "KiB", "effective_l1_kb_high": "KiB"},
                source="amora.probes.nvidia.baseline.l1_cache.working_set",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="l1_effective_capacity",
                value=capacity_value,
                unit="KiB",
                fit_status=fit,
                uncertainty=UncertaintyCategory.BOUNDED_RANGE,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="l1_effective_capacity_knee",
                interpretation={
                    "nvidia_backend": "effective L1 capacity bounded by the first latency knee in the working-set sweep",
                },
                downgrade_reason=None if knee else "no clear capacity knee detected in the sweep",
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="l1d_cache_capacity",
                value=capacity_value,
                unit="KiB",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=UncertaintyCategory.BOUNDED_RANGE,
                mapping_contract="working-set latency knee → simulator L1 capacity range",
                assumptions=assumptions,
            ),
        )
    ]
