"""Tensor-core MMA dependent-latency probe (P2).

One warp runs a dependent chain of wmma::mma_sync (FP16 m16n16k16) ops where the
accumulator feeds the next iteration, bracketed by clock64(). Reports
cycles-per-MMA as the representative scalar. SASS validation requires the HMMA
opcode and forbids register spills.
"""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.runner import CudaUnavailable, run_kernel
from amora.backends.nvidia.sass import SassExpectation
from amora.probes.nvidia.baseline._sources import (
    apply_sass_gating,
    collect_gcom_counter_comparison,
    collect_stall_attribution,
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


PROBE_ID = "tensor_core.mma_latency"
SOURCE = Path(__file__).with_name("mma_latency.cu")

# The timed loop must be a dependent HMMA chain with no register spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_tc_mma_latency",
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
                f"tensor-core MMA latency probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    cycles_per_mma = float(payload["cycles_per_mma"])

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

    stall_record = collect_stall_attribution(
        capabilities, SOURCE, kernel_name="amora_tc_mma_latency"
    )
    ncu_record = collect_gcom_counter_comparison(
        capabilities, SOURCE, kernel_name="amora_tc_mma_latency",
        extra_logicals=("tensor_pipe_active",),
    )

    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    if sass is not None:
        values["sass"] = sass.to_dict()
    if stall_record is not None:
        values["stall_attribution"] = stall_record
    if ncu_record is not None:
        values["gcom_counter_comparison"] = ncu_record
    assumptions = [
        "dependent wmma::mma_sync chain (FP16 m16n16k16) timed via clock64 in one warp",
        "median across launches",
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
                    "cycles_per_mma": cycles_per_mma,
                    "cycles_median": int(payload["cycles_median"]),
                },
                units={"cycles_per_mma": "cycles_per_op"},
                source="amora.probes.nvidia.baseline.tensor_core.mma_latency",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="tensor_mma_latency",
                value=cycles_per_mma,
                unit="cycles_per_op",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="tensor_core_mma_latency",
                interpretation={
                    "nvidia_backend": "dependent FP16 16x16x16 MMA latency in cycles",
                    "mma_shape": payload["mma_shape"],
                },
                metric_resolver=ncu_record or {},
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="tensor_core_mma_latency",
                value=cycles_per_mma,
                unit="cycles_per_op",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="dependent MMA cycles-per-op -> simulator tensor-core pipeline latency",
                assumptions=assumptions,
            ),
        )
    ]
