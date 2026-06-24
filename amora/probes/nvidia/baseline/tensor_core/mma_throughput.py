"""Tensor-core MMA throughput probe (P2).

Many warps each run independent wmma::mma_sync (FP16 m16n16k16) accumulators to
saturate the tensor pipe, bracketed by clock64(). Reports MMA-ops per cycle per
warp; the simulator initiation interval is its reciprocal. SASS validation
requires the HMMA opcode and forbids register spills.
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


PROBE_ID = "tensor_core.mma_throughput"
SOURCE = Path(__file__).with_name("mma_throughput.cu")

# The timed loop must contain independent HMMA ops and no register spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_tc_mma_throughput",
    required_opcodes={"HMMA": 1},
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
                f"tensor-core MMA throughput probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    mma_per_cycle = float(payload["mma_per_cycle_per_warp"])

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
    initiation_interval = round(1.0 / mma_per_cycle, 4) if mma_per_cycle > 0 else None
    assumptions = [
        "independent wmma::mma_sync accumulators (FP16 m16n16k16) expose ILP to saturate the tensor pipe",
        "median across launches; throughput reported per warp",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(
                probe_id=PROBE_ID,
                binary_hash=result.binary_sha256,
                disassembly_hash=sass.disassembly_hash if sass else None,
            ),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(grid=(1, 1, 1), block=(32 * int(payload["warps"]), 1, 1), mode="kernel"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "mma_per_cycle_per_warp": mma_per_cycle,
                    "cycles_median": int(payload["cycles_median"]),
                },
                units={"mma_per_cycle_per_warp": "mma/cycle"},
                source="amora.probes.nvidia.baseline.tensor_core.mma_throughput",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="tensor_mma_throughput",
                value=mma_per_cycle,
                unit="mma/cycle",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="tensor_core_mma_throughput",
                interpretation={
                    "nvidia_backend": "independent FP16 16x16x16 MMA throughput in MMA-ops per cycle per warp",
                    "mma_shape": payload["mma_shape"],
                },
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="tensor_core_initiation_interval",
                value=initiation_interval,
                unit="cycles_per_op",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="independent MMA throughput -> simulator tensor-core initiation interval",
                assumptions=assumptions,
            ),
        )
    ]
