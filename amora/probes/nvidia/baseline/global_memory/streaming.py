"""DRAM/HBM streaming-bandwidth probe (P2).

Measures sustained read, write, and copy bandwidth over a cache-exceeding
working set timed with CUDA events. Reports achieved GB/s per traffic class as a
bounded simulator-facing bandwidth (clocks may vary). SASS validation confirms
the copy kernel uses global loads with no spills.
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


PROBE_ID = "global_memory.streaming"
SOURCE = Path(__file__).with_name("streaming.cu")

# The copy kernel must be global-load/store bound with no spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_stream_copy",
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
                f"streaming-bandwidth probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    read_gbps = float(payload["read_gbps"])
    write_gbps = float(payload["write_gbps"])
    copy_gbps = float(payload["copy_gbps"])
    peak = max(read_gbps, write_gbps, copy_gbps)

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
    derived = {
        "read_gbps": read_gbps,
        "write_gbps": write_gbps,
        "copy_gbps": copy_gbps,
        "peak_gbps": peak,
    }
    assumptions = [
        "grid-stride read/write/copy over a working set far larger than cache",
        "best-of-N CUDA-event timing; bandwidth is bounded by clock variation",
        "copy moves 2x bytes (read+write); reported as achieved sustained GB/s",
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
                    "read_gbps": read_gbps,
                    "write_gbps": write_gbps,
                    "copy_gbps": copy_gbps,
                    "peak_gbps": peak,
                },
                units={
                    "read_gbps": "GB/s",
                    "write_gbps": "GB/s",
                    "copy_gbps": "GB/s",
                    "peak_gbps": "GB/s",
                },
                source="amora.probes.nvidia.baseline.global_memory.streaming",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="dram_sustained_bandwidth",
                value=derived,
                unit="GB/s",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="dram_streaming_bandwidth",
                interpretation={
                    "nvidia_backend": "sustained DRAM/HBM bandwidth per traffic class from streaming kernels",
                },
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="dram_bandwidth",
                value=derived,
                unit="GB/s",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="achieved sustained bandwidth per traffic class → simulator DRAM bandwidth (bounded)",
                assumptions=assumptions,
            ),
        )
    ]
