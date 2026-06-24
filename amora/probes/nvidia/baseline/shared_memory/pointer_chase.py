"""Shared-memory pointer-chase latency probe."""

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


PROBE_ID = "shared_memory.pointer_chase"
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
                f"shared-memory pointer-chase probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    cycles_per_load = float(payload["cycles_per_load"])
    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    assumptions = [
        "single-thread pointer chase over a 1024-entry shared-memory ring",
        "median cycles-per-load is reported across N kernel launches",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(probe_id=PROBE_ID, binary_hash=result.binary_sha256),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(grid=(1, 1, 1), block=(1024, 1, 1), mode="kernel"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "cycles_per_load": cycles_per_load,
                    "cycles_median": int(payload["cycles_median"]),
                    "chase_len": int(payload["chase_len"]),
                },
                units={"cycles_per_load": "cycles"},
                source="amora.probes.nvidia.baseline.shared_memory.pointer_chase",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="shared_memory_load_latency",
                value=cycles_per_load,
                unit="cycles",
                fit_status=FitStatus.DIRECT,
                uncertainty=UncertaintyCategory.STABLE_SCALAR,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="shared_memory_load_to_use_latency",
                interpretation={"nvidia_backend": "LDS dependent-load latency in cycles"},
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="shared_memory_load_latency_cycles",
                value=cycles_per_load,
                unit="cycles",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=FitStatus.DIRECT,
                uncertainty=UncertaintyCategory.STABLE_SCALAR,
                mapping_contract="dependent shared-memory chase cycles-per-load → simulator shared-mem latency",
                assumptions=assumptions,
            ),
        )
    ]
