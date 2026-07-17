"""Arithmetic independent-chains probe: measures FP32 FMA throughput per SM."""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.runner import CudaUnavailable, run_kernel
from amora.backends.nvidia.sass import SassExpectation
from amora.probes.nvidia.baseline._sources import (
    apply_sass_gating,
    collect_gcom_counter_comparison,
    collect_stall_attribution,
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


PROBE_ID = "arithmetic_throughput.independent_chains"
SOURCE = Path(__file__).with_name("independent_chains.cu")

# The timed loop must be independent FFMA chains with no register spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_baseline_fp32_independent_chains",
    required_opcodes={"FFMA": 8},
    forbidden_opcodes=("LDL", "STL"),
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
                f"independent-chains throughput probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    cycles_per_fma = float(payload["cycles_per_fma_per_thread"])
    fma_per_cycle_per_sm = float(payload["approx_fma_per_cycle_per_sm"])

    # SASS gating: reject if the timed loop is not FFMA throughput bound.
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

    stall_record = collect_stall_attribution(
        capabilities, SOURCE, kernel_name="amora_baseline_fp32_independent_chains"
    )
    ncu_record = collect_gcom_counter_comparison(
        capabilities, SOURCE, kernel_name="amora_baseline_fp32_independent_chains"
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
        "4 independent FMA chains per thread to expose ILP",
        "throughput is per-thread cycles-per-op; per-SM is approximate (assumes resident across all SMs)",
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
                grid=(int(payload["blocks"]), 1, 1),
                block=(int(payload["threads"]), 1, 1),
                mode="kernel",
            ),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "cycles_per_fma_per_thread": cycles_per_fma,
                    "approx_fma_per_cycle_per_sm": fma_per_cycle_per_sm,
                    "cycles_median": int(payload["cycles_median"]),
                },
                units={
                    "cycles_per_fma_per_thread": "cycles",
                    "approx_fma_per_cycle_per_sm": "fma/cycle/sm",
                },
                source="amora.probes.nvidia.baseline.arithmetic_throughput.independent_chains",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="fp32_fma_throughput",
                value=cycles_per_fma,
                unit="cycles_per_op",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="fp32_fma_independent_pipeline_throughput",
                interpretation={"nvidia_backend": "effective FMA cycles-per-op once ILP saturates the FP32 pipe"},
                metric_resolver=ncu_record or {},
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="fp32_fma_throughput",
                value=cycles_per_fma,
                unit="cycles_per_op",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="independent FMA cycles-per-op → simulator FP32 FMA throughput",
                assumptions=assumptions,
            ),
        )
    ]
