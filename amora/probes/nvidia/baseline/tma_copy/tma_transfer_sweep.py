"""Async-copy (cp.async) transfer-size sweep probe (P3, Phase D).

One CTA bulk-stages global memory into shared memory via the Ampere+ async-copy
pipeline, sweeping the shared-tile size (1KB..32KB) and measuring achieved GB/s
per size with CUDA events. Reports the peak achieved throughput as a bounded
async-copy bandwidth. SASS validation requires the cp.async store (LDGSTS) and
forbids spills.
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


PROBE_ID = "tma_copy.tma_transfer_sweep"
SOURCE = Path(__file__).with_name("tma_transfer_sweep.cu")

# The timed loop must lower to cp.async (LDGSTS) with no spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_tma_transfer_sweep",
    required_opcodes={"LDGSTS": 1},
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
                f"async-copy transfer-sweep probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    sweep = list(payload["sweep"])
    gbps = [float(p["gbps"]) for p in sweep if float(p["gbps"]) > 0]
    peak_gbps = max(gbps) if gbps else 0.0

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

    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    if sass is not None:
        values["sass"] = sass.to_dict()
    assumptions = [
        "one CTA bulk-stages global->shared via cp.async over a tile-size sweep",
        "best-of-N CUDA-event timing per size; throughput is bounded by clock variation",
        "peak GB/s across the sweep is reported as achieved async-copy bandwidth",
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
                    "peak_gbps": peak_gbps,
                    "sweep_points": len(sweep),
                },
                units={"peak_gbps": "GB/s"},
                source="amora.probes.nvidia.baseline.tma_copy.tma_transfer_sweep",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="async_copy_throughput",
                value=peak_gbps,
                unit="GB/s",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="async_copy_throughput",
                interpretation={
                    "nvidia_backend": "peak achieved cp.async global->shared bandwidth across tile sizes",
                },
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="tma_transfer_throughput",
                value=peak_gbps,
                unit="GB/s",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="peak async-copy bandwidth across tile sizes -> simulator async-copy transfer throughput (bounded)",
                assumptions=assumptions,
            ),
        )
    ]
