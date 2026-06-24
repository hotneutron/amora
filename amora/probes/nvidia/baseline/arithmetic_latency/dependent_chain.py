"""Arithmetic dependent-chain probe: measures FP32 FMA latency."""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.runner import CudaUnavailable, run_kernel
from amora.backends.nvidia.sass import SassExpectation, gate_decision
from amora.probes.nvidia.baseline._sources import (
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


PROBE_ID = "arithmetic_latency.dependent_chain"
SOURCE = Path(__file__).with_name("dependent_chain.cu")

# The timed loop must be a dependent FFMA chain with no register spills.
# Counts are static (post-unroll) opcodes from cuobjdump, not dynamic; the
# #pragma unroll body yields ~unroll-factor FFMAs. Spills (LDL/STL) inside the
# function would corrupt the latency measurement. Result write-back (STG) is
# expected outside the hot loop and is not forbidden.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_baseline_fp32_dependent_chain",
    required_opcodes={"FFMA": 8},
    forbidden_opcodes=("LDL", "STL"),
    require_dependency=True,
    dependency_opcode="FFMA",
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
                f"dependent-chain latency probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    cycles_per_fma = float(payload["cycles_per_fma"])

    # SASS gating: reject if the timed loop is not a dependent FFMA chain.
    sass = result.sass_validation
    decision = gate_decision(sass, EXPECTATION) if sass is not None else "pass"
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
    fit = FitStatus.DIRECT
    uncertainty = UncertaintyCategory.STABLE_SCALAR
    downgrade_reason = None
    if decision == "downgrade":
        fit = downgrade_fit(fit)
        uncertainty = soften_uncertainty(uncertainty)
        downgrade_reason = f"SASS validation downgrade: {sass.reason}"

    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    if sass is not None:
        values["sass"] = sass.to_dict()
    assumptions = [
        "FP32 FMA dependent chain timed via clock64 inside a single warp",
        "median across N launches is reported to suppress one-shot kernel-launch jitter",
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
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="fp32_fma_dependent_pipeline_latency",
                interpretation={"nvidia_backend": "cycles between issue and writeback for a dependent FMA"},
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="fp32_fma_pipeline_depth",
                value=cycles_per_fma,
                unit="cycles_per_op",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="dependent FMA cycles-per-op → simulator FP32 FMA latency depth",
                assumptions=assumptions,
            ),
        )
    ]
