"""Differential register / operand-delivery latency probe (P1).

Measures the per-op latency of a tight same-register RAW chain vs a rotating
multi-register chain of equal length. The difference isolates the differential
operand-delivery / scoreboard cost, kept separate from the absolute arithmetic
latency. Reported as conditional per the P1 methodology (operand-collector
parameters are entangled with scoreboard and bank behavior).
"""

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


PROBE_ID = "register_file.register_latency"
SOURCE = Path(__file__).with_name("register_latency.cu")


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
                f"register-latency probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    same = float(payload["same_reg_cycles_per_op"])
    rot = float(payload["rotating_reg_cycles_per_op"])
    differential = float(payload["differential_cycles_per_op"])
    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    assumptions = [
        "same-register (RAW distance 1) vs rotating-register (relaxed RAW) chains of equal length",
        "differential cycles-per-op isolates operand-delivery cost from absolute arithmetic latency",
        "operand-collector parameters stay conditional: scoreboard/bank effects are entangled",
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
                    "same_reg_cycles_per_op": same,
                    "rotating_reg_cycles_per_op": rot,
                    "differential_cycles_per_op": differential,
                },
                units={
                    "same_reg_cycles_per_op": "cycles",
                    "rotating_reg_cycles_per_op": "cycles",
                    "differential_cycles_per_op": "cycles",
                },
                source="amora.probes.nvidia.baseline.register_file.register_latency",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="operand_delivery_differential_latency",
                value=differential,
                unit="cycles",
                fit_status=FitStatus.CONDITIONALLY_IDENTIFIED,
                uncertainty=UncertaintyCategory.CONDITIONAL_SCALAR,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="register_operand_delivery_latency",
                interpretation={
                    "nvidia_backend": "extra per-op cost of tight RAW dependence attributable to operand delivery",
                    "same_reg_cycles_per_op": same,
                    "rotating_reg_cycles_per_op": rot,
                },
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="max_latency_regular_register_file_latency",
                value=differential,
                unit="cycles",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=FitStatus.CONDITIONALLY_IDENTIFIED,
                uncertainty=UncertaintyCategory.CONDITIONAL_SCALAR,
                mapping_contract="RAW-distance differential cycles → simulator operand-delivery latency (conditional)",
                assumptions=assumptions,
            ),
        )
    ]
