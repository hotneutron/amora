"""Register-bank / operand-delivery sweep probe (P1).

Sweeps the number of independent live accumulators (operand width / register
pressure) and reports cycles-per-op per width. Periodic throughput dips suggest
register-bank or operand-collector conflicts. Per the P1 methodology, and
because this is a CUDA approximation of the ideal SASS-controlled register
sweep, the result is reported as an underconstrained candidate curve rather than
an exact bank count.
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


PROBE_ID = "register_file.register_bank_sweep"
SOURCE = Path(__file__).with_name("register_bank_sweep.cu")

# The templated reg-width kernel must be FFMA-bound with no spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_reg_width",
    required_opcodes={"FFMA": 8},
    forbidden_opcodes=("LDL", "STL"),
)


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def _candidate_period(sweep: list[dict]) -> int | None:
    """Width at which cycles-per-op stops improving (operand-delivery plateau)."""

    best_width = None
    best_cpo = None
    for point in sweep:
        cpo = float(point["cycles_per_op"])
        if best_cpo is None or cpo < best_cpo * 0.98:
            best_cpo = cpo
            best_width = int(point["width"])
    return best_width


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    src_descriptor = source_descriptor(SOURCE)
    try:
        result = run_kernel(SOURCE, capabilities=capabilities, timeout=60, expectation=EXPECTATION)
    except CudaUnavailable as exc:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"register-bank sweep could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    sweep = list(payload["sweep"])
    plateau_width = _candidate_period(sweep)

    # SASS gating: the templated reg-width kernel must be FFMA-bound with no spills.
    sass = result.sass_validation
    decision, fit, uncertainty, downgrade_reason = apply_sass_gating(
        sass, EXPECTATION, FitStatus.UNDERCONSTRAINED, UncertaintyCategory.MULTI_FIT
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
        "operand-width sweep of independent FMA accumulators (register pressure proxy)",
        "CUDA approximation of the SASS-controlled register sweep; bank count is not uniquely identified",
        "plateau width marks where added ILP stops improving cycles-per-op",
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
                    "ilp_plateau_width": plateau_width,
                    "sweep_points": len(sweep),
                },
                units={"ilp_plateau_width": "accumulators"},
                source="amora.probes.nvidia.baseline.register_file.register_bank_sweep",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="operand_delivery_plateau",
                value=plateau_width,
                unit="accumulators",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="register_bank_operand_delivery",
                interpretation={
                    "nvidia_backend": "operand-delivery throughput plateau across register-pressure widths",
                },
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason
                if downgrade_reason is not None
                else "CUDA proxy cannot isolate register-bank count without SASS register control",
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="gpgpu_num_reg_banks",
                value=plateau_width,
                unit="accumulators",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="operand-width plateau → simulator register-bank pressure (candidate, multi-fit)",
                assumptions=assumptions,
            ),
        )
    ]
