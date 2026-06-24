"""Interconnect injection-rate probe (P3, Phase D).

A multi-SM grid streams a cache-exceeding global buffer, sweeping the number of
resident blocks (grid = mp_count * {1,2,4,8}) to vary the offered load on the
memory interconnect. Reports the peak aggregate achieved GB/s as a bounded
injection-saturation bandwidth. SASS validation confirms the kernel is a
global-load stream. NCU DRAM read bytes are collected best-effort.
"""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.runner import CudaUnavailable, run_kernel
from amora.backends.nvidia.sass import SassExpectation
from amora.probes.nvidia.baseline._sources import apply_sass_gating, collect_ncu_metrics, source_descriptor
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


PROBE_ID = "interconnect.injection_rate"
SOURCE = Path(__file__).with_name("injection_rate.cu")

# The streaming kernel must hit global memory (LDG) without spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_icn_injection_rate",
    required_opcodes={"LDG": 1},
    forbidden_opcodes=("LDL", "STL"),
)


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    src_descriptor = source_descriptor(SOURCE)
    try:
        result = run_kernel(SOURCE, capabilities=capabilities, timeout=120, expectation=EXPECTATION)
    except CudaUnavailable as exc:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"injection-rate probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    sweep = list(payload["sweep"])
    gbps = [float(p["gbps"]) for p in sweep if float(p["gbps"]) > 0]
    saturation_gbps = max(gbps) if gbps else 0.0

    sass = result.sass_validation
    decision, fit, uncertainty, downgrade_reason = apply_sass_gating(
        sass, EXPECTATION, FitStatus.BOUNDED, UncertaintyCategory.BOUNDED_RANGE
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

    # NCU DRAM read bytes (best effort) corroborate that the sweep is DRAM-bound.
    ncu_record = collect_ncu_metrics(
        capabilities,
        SOURCE,
        ["dram_bytes_read"],
        kernel_name="amora_icn_injection_rate",
        role="primary",
        aggregate="sum",
    )

    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    if sass is not None:
        values["sass"] = sass.to_dict()
    if ncu_record is not None:
        values["ncu"] = ncu_record
    assumptions = [
        "multi-SM grid-stride stream over a working set far larger than cache",
        "offered load swept via blocks-per-SM = {1,2,4,8}; best-of-N CUDA-event timing",
        "peak aggregate GB/s across offered loads is the injection-saturation bandwidth",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(
                probe_id=PROBE_ID,
                binary_hash=result.binary_sha256,
                disassembly_hash=sass.disassembly_hash if sass else None,
            ),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(mode="kernel"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "saturation_gbps": saturation_gbps,
                    "sweep_points": len(sweep),
                },
                units={"saturation_gbps": "GB/s"},
                source="amora.probes.nvidia.baseline.interconnect.injection_rate",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="injection_saturation_gbps",
                value=saturation_gbps,
                unit="GB/s",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="interconnect_injection",
                interpretation={
                    "nvidia_backend": "peak aggregate injection bandwidth vs offered load (blocks per SM)",
                },
                metric_resolver=ncu_record or {},
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="interconnect_injection_bandwidth",
                value=saturation_gbps,
                unit="GB/s",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="peak aggregate injection bandwidth vs offered load -> simulator interconnect injection bandwidth (bounded)",
                assumptions=assumptions,
            ),
        )
    ]
