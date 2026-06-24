"""Async-copy (cp.async) tile-latency probe (P3, Phase D).

One CTA stages tiles from global to shared memory with the Ampere+ async-copy
pipeline (__pipeline_memcpy_async + commit + wait), timing the issue->wait->use
sequence per tile with clock64(). Reports cycles-per-tile as a conditional
async-copy completion latency. SASS validation requires the cp.async store
opcode (LDGSTS) and forbids spills; if the compiler emitted plain LDG/STS the
reject is an honest signal that cp.async was not used.
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


PROBE_ID = "tma_copy.async_copy_latency"
SOURCE = Path(__file__).with_name("async_copy_latency.cu")

# The timed loop must lower to cp.async (LDGSTS) with no spills. A compiler that
# fell back to plain LDG/STS yields a reject (honest "cp.async not used").
EXPECTATION = SassExpectation(
    kernel_symbol="amora_tma_async_copy_latency",
    required_opcodes={"LDGSTS": 1},
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
                f"async-copy latency probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    cycles_per_tile = float(payload["cycles_per_tile"])

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
        "one CTA stages tiles global->shared via cp.async (__pipeline_memcpy_async)",
        "cycles-per-tile brackets issue->commit->wait->use with clock64()",
        "completion latency is conditional on cp.async being emitted (LDGSTS)",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(
                probe_id=PROBE_ID,
                binary_hash=result.binary_sha256,
                disassembly_hash=sass.disassembly_hash if sass else None,
            ),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(grid=(1, 1, 1), block=(128, 1, 1), mode="kernel"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "cycles_per_tile": cycles_per_tile,
                    "tiles": int(payload["tiles"]),
                    "bytes_per_tile": int(payload["bytes_per_tile"]),
                },
                units={"cycles_per_tile": "cycles"},
                source="amora.probes.nvidia.baseline.tma_copy.async_copy_latency",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="async_copy_tile_latency",
                value=cycles_per_tile,
                unit="cycles",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="async_copy_latency",
                interpretation={
                    "nvidia_backend": "cycles per cp.async-staged tile (issue->wait->use)",
                },
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="async_copy_completion_latency",
                value=cycles_per_tile,
                unit="cycles",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="async-copy issue->wait->use cycles -> simulator async-copy completion latency (conditional)",
                assumptions=assumptions,
            ),
        )
    ]
