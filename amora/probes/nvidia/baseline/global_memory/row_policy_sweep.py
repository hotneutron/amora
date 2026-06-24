"""DRAM row-locality (row-buffer policy) sweep probe (P2, Phase B).

Streams a large DRAM buffer with several element strides to vary DRAM row-buffer
locality and times each with CUDA events. The bounded ratio of best to worst
bandwidth quantifies how sensitive the device is to row locality (a proxy for the
DRAM row-buffer policy). SASS validation confirms the kernel is a global-load
stream.
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


PROBE_ID = "global_memory.row_policy_sweep"
SOURCE = Path(__file__).with_name("row_policy_sweep.cu")

# The timed loop must hit global memory (LDG) without shared or local spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_gmem_row_policy",
    required_opcodes={"LDG": 1},
    forbidden_opcodes=("LDS", "STL"),
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
                f"row-policy-sweep probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    sweep = list(payload["sweep"])
    gbps = [float(p["gbps"]) for p in sweep if float(p["gbps"]) > 0]
    best_gbps = max(gbps) if gbps else 0.0
    worst_gbps = min(gbps) if gbps else 0.0
    sensitivity = (best_gbps / worst_gbps) if worst_gbps > 0 else None

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
        "grid-stride read with several element strides to vary DRAM row-buffer locality",
        "best-of-N CUDA-event timing per stride; bandwidth bounded by clock variation",
        "row_locality_sensitivity = best_gbps / worst_gbps across strides (bounded)",
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
                    "best_gbps": best_gbps,
                    "worst_gbps": worst_gbps,
                    "row_locality_sensitivity": sensitivity,
                },
                units={"best_gbps": "GB/s", "worst_gbps": "GB/s"},
                source="amora.probes.nvidia.baseline.global_memory.row_policy_sweep",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="row_locality_sensitivity",
                value=sensitivity,
                unit="ratio",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="dram_row_locality",
                interpretation={
                    "nvidia_backend": "DRAM row-locality sensitivity from a stride bandwidth sweep",
                    "row_locality_sensitivity": sensitivity,
                },
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="dram_row_policy_class",
                value=sensitivity,
                unit="ratio",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="stride bandwidth spread -> simulator DRAM row-buffer policy class (bounded)",
                assumptions=assumptions,
            ),
        )
    ]
