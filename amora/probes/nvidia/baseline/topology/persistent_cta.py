"""Topology persistent-CTA probe: launches a CUDA driver and parses concurrency."""

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


PROBE_ID = "topology.persistent_cta"
SOURCE = Path(__file__).with_name("persistent_cta.cu")

# The persistent-CTA kernel must run without register spills (LDL/STL).
EXPECTATION = SassExpectation(
    kernel_symbol="amora_baseline_persistent_cta",
    forbidden_opcodes=("LDL", "STL"),
)


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    src_descriptor = source_descriptor(SOURCE)
    try:
        result = run_kernel(SOURCE, capabilities=capabilities, args=("--blocks", "1024", "--threads", "32"), expectation=EXPECTATION)
    except CudaUnavailable as exc:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"persistent CTA probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    peak = int(payload["peak_resident_blocks_per_sm"])
    sms = int(payload["sm_count_observed"])
    advertised_sms = int(payload["multi_processor_count"])

    # SASS gating: reject if the persistent-CTA kernel spilled registers.
    sass = result.sass_validation
    decision, fit, uncertainty, downgrade_reason = apply_sass_gating(
        sass, EXPECTATION, FitStatus.UNIQUELY_IDENTIFIED, UncertaintyCategory.STABLE_SCALAR
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
        "concurrency derived from sweep-line over per-SM (start,end) cycle pairs",
        "kernel uses %smid plus a busy-spin to keep blocks resident long enough to overlap",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(
                probe_id=PROBE_ID,
                binary_hash=result.binary_sha256,
                disassembly_hash=sass.disassembly_hash if sass else None,
            ),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(
                grid=(int(payload["blocks_launched"]), 1, 1),
                block=(int(payload["threads_per_block"]), 1, 1),
                mode="kernel",
                extras={"busy_cycles": int(payload["busy_cycles"])},
            ),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "peak_resident_blocks_per_sm": peak,
                    "mean_resident_blocks_per_sm": float(payload["mean_resident_blocks_per_sm"]),
                    "sm_count_observed": sms,
                    "multi_processor_count": advertised_sms,
                    "elapsed_ms": float(payload["elapsed_ms"]),
                },
                units={"elapsed_ms": "ms"},
                source="amora.probes.nvidia.baseline.topology.persistent_cta",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="resident_blocks_per_sm",
                value=peak,
                unit="blocks",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="cuda_resident_blocks_per_sm",
                interpretation={
                    "nvidia_backend": "peak resident CTAs per SM under the configured launch shape",
                    "device_advertised_sm_count": advertised_sms,
                    "sm_count_observed": sms,
                },
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="max_resident_ctas_per_sm",
                value=peak,
                unit="ctas",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="observed peak block residency under busy-spin → simulator max_resident_ctas_per_sm",
                assumptions=assumptions,
            ),
        )
    ]
