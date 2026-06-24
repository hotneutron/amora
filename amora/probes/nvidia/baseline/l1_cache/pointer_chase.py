"""L1 cache-path dependent pointer-chase probe (P1).

Measures L1-hit load latency by walking a randomized pointer-chase ring that
fits inside the candidate L1 data cache, with a DRAM-resident ring as a control
so the hit regime can be validated (small << large cycles-per-load).
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


PROBE_ID = "l1_cache.pointer_chase"
SOURCE = Path(__file__).with_name("pointer_chase.cu")

# The timed loop must hit global memory (LDG) without shared or local spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_l1_pointer_chase",
    required_opcodes={"LDG": 1},
    forbidden_opcodes=("LDS", "STL"),
)


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    src_descriptor = source_descriptor(SOURCE)
    try:
        result = run_kernel(SOURCE, capabilities=capabilities, expectation=EXPECTATION)
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
    fit = FitStatus.DIRECT if hit_regime else FitStatus.BOUNDED
    uncertainty = (
        UncertaintyCategory.STABLE_SCALAR
        if hit_regime
        else UncertaintyCategory.BOUNDED_RANGE
    )

    # SASS gating: reject if the timed loop is not a global-load chase.
    sass = result.sass_validation
    decision, fit, uncertainty, downgrade_reason = apply_sass_gating(
        sass, EXPECTATION, fit, uncertainty
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
        "single-thread dependent pointer chase over a randomized ring sized to fit L1",
        "a DRAM-resident ring is timed as a control; L1-hit regime requires small << large",
        "median cycles-per-load reported across N launches",
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
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="l1_path_hit_latency",
                interpretation={
                    "nvidia_backend": "dependent-load latency for an L1-resident working set in cycles",
                    "dram_control_cycles_per_load": dram_cpl,
                    "l1_hit_regime_confirmed": hit_regime,
                },
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason
                if downgrade_reason is not None
                else (None if hit_regime else "small ring not clearly faster than DRAM control"),
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="l1_latency",
                value=l1_cpl,
                unit="cycles",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="dependent L1-hit chase cycles-per-load → simulator L1 hit latency",
                assumptions=assumptions,
            ),
        )
    ]
