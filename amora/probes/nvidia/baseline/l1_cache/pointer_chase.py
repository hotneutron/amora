"""L1 cache-path dependent pointer-chase probe (P1).

Measures L1-hit load latency by walking a randomized pointer-chase ring that
fits inside the candidate L1 data cache, with a DRAM-resident ring as a control
so the hit regime can be validated (small << large cycles-per-load).
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


PROBE_ID = "l1_cache.pointer_chase"
SOURCE = Path(__file__).with_name("pointer_chase.cu")


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    src_descriptor = source_descriptor(SOURCE)
    try:
        result = run_kernel(SOURCE, capabilities=capabilities)
    except CudaUnavailable as exc:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"L1 pointer-chase probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    l1_cpl = float(payload["l1_hit_cycles_per_load"])
    dram_cpl = float(payload["dram_cycles_per_load"])
    # The small ring is only an L1-hit regime if it is clearly faster than DRAM.
    hit_regime = dram_cpl > l1_cpl * 1.5
    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    assumptions = [
        "single-thread dependent pointer chase over a randomized ring sized to fit L1",
        "a DRAM-resident ring is timed as a control; L1-hit regime requires small << large",
        "median cycles-per-load reported across N launches",
    ]
    fit = FitStatus.DIRECT if hit_regime else FitStatus.BOUNDED
    return [
        ProbeResult(
            identity=ProbeIdentity(probe_id=PROBE_ID, binary_hash=result.binary_sha256),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(grid=(1, 1, 1), block=(32, 1, 1), mode="kernel"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "l1_hit_cycles_per_load": l1_cpl,
                    "dram_cycles_per_load": dram_cpl,
                    "hit_to_dram_ratio": dram_cpl / l1_cpl if l1_cpl > 0 else None,
                },
                units={
                    "l1_hit_cycles_per_load": "cycles",
                    "dram_cycles_per_load": "cycles",
                },
                source="amora.probes.nvidia.baseline.l1_cache.pointer_chase",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="l1_hit_load_latency",
                value=l1_cpl,
                unit="cycles",
                fit_status=fit,
                uncertainty=UncertaintyCategory.STABLE_SCALAR
                if hit_regime
                else UncertaintyCategory.BOUNDED_RANGE,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="l1_path_hit_latency",
                interpretation={
                    "nvidia_backend": "dependent-load latency for an L1-resident working set in cycles",
                    "dram_control_cycles_per_load": dram_cpl,
                    "l1_hit_regime_confirmed": hit_regime,
                },
                downgrade_reason=None if hit_regime else "small ring not clearly faster than DRAM control",
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="l1_latency",
                value=l1_cpl,
                unit="cycles",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=UncertaintyCategory.STABLE_SCALAR
                if hit_regime
                else UncertaintyCategory.BOUNDED_RANGE,
                mapping_contract="dependent L1-hit chase cycles-per-load → simulator L1 hit latency",
                assumptions=assumptions,
            ),
        )
    ]
