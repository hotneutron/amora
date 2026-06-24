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

# The templated reg-width kernel must be FFMA-bound with no spills. We also
# count distinct FFMA register operands in SASS to confirm the compiler kept the
# accumulators in separate registers (the prerequisite for a bank claim).
EXPECTATION = SassExpectation(
    kernel_symbol="amora_reg_width",
    required_opcodes={"FFMA": 8},
    forbidden_opcodes=("LDL", "STL"),
    count_registers_opcode="FFMA",
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

    # SASS-controlled graduation: if the disassembly confirms the kernel used
    # several distinct FFMA registers (i.e. the compiler did not coalesce the
    # swept accumulators), the operand-width plateau is backed by real register
    # control and graduates from underconstrained to bounded.
    reg_count = sass.register_count if sass else None
    sass_register_controlled = reg_count is not None and reg_count >= 4
    if sass_register_controlled and fit == FitStatus.UNDERCONSTRAINED:
        fit = FitStatus.BOUNDED
        uncertainty = UncertaintyCategory.BOUNDED_RANGE

    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    if sass is not None:
        values["sass"] = sass.to_dict()
    assumptions = [
        "operand-width sweep of independent FMA accumulators (register pressure proxy)",
        "SASS confirms distinct FFMA register operands so the sweep is register-controlled",
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
                    "sass_distinct_ffma_registers": reg_count,
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
                else (
                    None
                    if sass_register_controlled
                    else "register sweep not SASS-confirmed; bank count not isolated"
                ),
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
