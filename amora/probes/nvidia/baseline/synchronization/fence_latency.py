"""Memory fence latency probe (P2).

One CTA times a long loop of __threadfence() calls bracketed by clock64(), plus
an empty-loop baseline with the same structure (minus the fence) so the wrapper
can subtract loop overhead. Reports net cycles-per-fence as the representative
scalar. SASS validation requires the MEMBAR opcode and forbids register spills.
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


PROBE_ID = "synchronization.fence_latency"
SOURCE = Path(__file__).with_name("fence_latency.cu")

# The timed loop must contain a memory fence (MEMBAR) and no register spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_baseline_fence_latency",
    required_opcodes={"MEMBAR": 1},
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
                f"memory fence latency probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    net_cycles_per_fence = float(payload["net_cycles_per_fence"])

    sass = result.sass_validation
    decision, fit, uncertainty, downgrade_reason = apply_sass_gating(
        sass, EXPECTATION, FitStatus.CONDITIONALLY_IDENTIFIED, UncertaintyCategory.CONDITIONAL_SCALAR
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
        "device-scope __threadfence() loop timed via clock64 in one CTA",
        "empty-loop baseline subtracted to remove loop/branch overhead from the per-fence cost",
        "net per-fence cost reflects fence scope as measured; median across launches",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(
                probe_id=PROBE_ID,
                binary_hash=result.binary_sha256,
                disassembly_hash=sass.disassembly_hash if sass else None,
            ),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(grid=(1, 1, 1), block=(256, 1, 1), mode="kernel"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "cycles_per_fence": float(payload["cycles_per_fence"]),
                    "cycles_per_empty": float(payload["cycles_per_empty"]),
                    "net_cycles_per_fence": net_cycles_per_fence,
                },
                units={"net_cycles_per_fence": "cycles"},
                source="amora.probes.nvidia.baseline.synchronization.fence_latency",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="memory_fence_latency",
                value=net_cycles_per_fence,
                unit="cycles",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="memory_fence_latency",
                interpretation={
                    "nvidia_backend": "net cycles per __threadfence() after subtracting empty-loop overhead",
                },
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="fence_latency",
                value=net_cycles_per_fence,
                unit="cycles",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="net per-fence cycles -> simulator memory fence latency (conditional)",
                assumptions=assumptions,
            ),
        )
    ]
