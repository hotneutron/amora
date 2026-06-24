"""Arithmetic dependent-chain probe: measures FP32 FMA latency."""

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


PROBE_ID = "arithmetic_latency.dependent_chain"
SOURCE = Path(__file__).with_name("dependent_chain.cu")


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
                f"dependent-chain latency probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    cycles_per_fma = float(payload["cycles_per_fma"])
    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    assumptions = [
        "FP32 FMA dependent chain timed via clock64 inside a single warp",
        "median across N launches is reported to suppress one-shot kernel-launch jitter",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(probe_id=PROBE_ID, binary_hash=result.binary_sha256),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(grid=(1, 1, 1), block=(32, 1, 1), mode="kernel"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "cycles_per_fma": cycles_per_fma,
                    "cycles_median": int(payload["cycles_median"]),
                    "chain_length": int(payload["chain_length"]),
                },
                units={"cycles_per_fma": "cycles"},
                source="amora.probes.nvidia.baseline.arithmetic_latency.dependent_chain",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="fp32_fma_dependent_latency",
                value=cycles_per_fma,
                unit="cycles_per_op",
                fit_status=FitStatus.DIRECT,
                uncertainty=UncertaintyCategory.STABLE_SCALAR,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="fp32_fma_dependent_pipeline_latency",
                interpretation={"nvidia_backend": "cycles between issue and writeback for a dependent FMA"},
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="fp32_fma_pipeline_depth",
                value=cycles_per_fma,
                unit="cycles_per_op",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=FitStatus.DIRECT,
                uncertainty=UncertaintyCategory.STABLE_SCALAR,
                mapping_contract="dependent FMA cycles-per-op → simulator FP32 FMA latency depth",
                assumptions=assumptions,
            ),
        )
    ]
