"""CTA barrier-latency probe (P2).

Times a long run of __syncthreads() barriers with minimal work between them,
swept over block sizes. Reports cycles-per-barrier for the smallest block as the
representative scalar and the full scaling curve. SASS validation requires the
barrier instruction (BAR) and forbids spills.
"""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.runner import CudaUnavailable, run_kernel
from amora.backends.nvidia.sass import SassExpectation
from amora.probes.nvidia.baseline._sources import apply_sass_gating, source_descriptor
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


PROBE_ID = "synchronization.barrier_latency"
SOURCE = Path(__file__).with_name("barrier_latency.cu")

# The timed loop must contain CTA barriers (BAR) and no spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_baseline_barrier_latency",
    required_opcodes={"BAR": 1},
    forbidden_opcodes=("LDL", "STL"),
)


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    src_descriptor = source_descriptor(SOURCE)
    try:
        result = run_kernel(SOURCE, capabilities=capabilities, timeout=60, expectation=EXPECTATION)
    except CudaUnavailable as exc:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"barrier-latency probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    sweep = list(payload["sweep"])
    # Representative scalar: smallest block (least contention).
    first = min(sweep, key=lambda p: int(p["threads_per_block"]))
    cpb = float(first["cycles_per_barrier"])

    sass = result.sass_validation
    decision, fit, uncertainty, downgrade_reason = apply_sass_gating(
        sass, EXPECTATION, FitStatus.UNIQUELY_IDENTIFIED, UncertaintyCategory.CONDITIONAL_SCALAR
    )
    if decision == "reject":
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"SASS validation rejected the measurement: {sass.reason}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor, "sass": sass.to_dict()},
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
        "one CTA runs a long __syncthreads() loop with minimal inter-barrier work",
        "cycles-per-barrier reported for the smallest block; scaling curve retained",
        "barrier cost is occupancy-coupled; reported per the launch class measured",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(
                probe_id=PROBE_ID,
                binary_hash=result.binary_sha256,
                disassembly_hash=sass.disassembly_hash if sass else None,
            ),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(grid=(1, 1, 1), block=(int(first["threads_per_block"]), 1, 1), mode="kernel"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "cycles_per_barrier": cpb,
                    "sweep_points": len(sweep),
                },
                units={"cycles_per_barrier": "cycles"},
                source="amora.probes.nvidia.baseline.synchronization.barrier_latency",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="cta_barrier_latency",
                value=cpb,
                unit="cycles",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="cta_barrier_latency",
                interpretation={
                    "nvidia_backend": "cycles per __syncthreads() barrier for the measured CTA shape",
                },
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="barrier_latency",
                value=cpb,
                unit="cycles",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="cycles-per-barrier for a named CTA shape → simulator barrier latency (conditional)",
                assumptions=assumptions,
            ),
        )
    ]
