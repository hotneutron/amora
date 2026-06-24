"""Register-file & operand-delivery cross-probe analyzer (P1).

Separates register-bank evidence (the operand-width plateau) from operand-
delivery latency (the RAW-distance differential), keeping bank count as a
candidate and the differential as conditional, per the P1 methodology.
"""

from __future__ import annotations

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.probes.nvidia.baseline.register_file import register_bank_sweep, register_latency
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


PROBE_ID = "register_file.analyze"


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    bank = register_bank_sweep.run(capabilities)[0]
    latency = register_latency.run(capabilities)[0]
    inputs = {"register_bank_sweep": bank, "register_latency": latency}

    unsupported = [name for name, r in inputs.items()
                   if r.raw_observation.evidence_tier == EvidenceTier.UNSUPPORTED]
    if unsupported:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"register-file analyzer cannot run: missing inputs from {', '.join(unsupported)}",
                tool_context=_tool_context(capabilities),
                raw_values={name: r.raw_observation.evidence_tier.value
                            for name, r in inputs.items()},
            )
        ]

    plateau = bank.normalized_measurement.value
    differential = latency.normalized_measurement.value
    derived = {
        "operand_delivery_plateau_accumulators": plateau,
        "operand_delivery_differential_cycles": differential,
    }
    values = {
        "register_bank_sweep": {
            "binary_sha256": bank.identity.binary_hash,
            "ilp_plateau_width": plateau,
        },
        "register_latency": {
            "binary_sha256": latency.identity.binary_hash,
            "differential_cycles_per_op": differential,
        },
        "derived": derived,
    }
    assumptions = [
        "keeps register-bank evidence (plateau) separate from operand-delivery latency (differential)",
        "bank count remains a candidate; differential latency remains conditional",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(probe_id=PROBE_ID),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(mode="analysis"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.COUPLED_INFERENCE,
                values=values,
                metrics=derived,
                source="amora.probes.nvidia.baseline.register_file.analyze",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="register_file_summary",
                value=derived,
                fit_status=FitStatus.UNDERCONSTRAINED,
                uncertainty=UncertaintyCategory.MULTI_FIT,
                assumptions=assumptions,
                coupled_with=["register_file.register_bank_sweep", "register_file.register_latency"],
            ),
            backend_interpretation=BackendInterpretation(
                concept="register_file_summary",
                interpretation={
                    "nvidia_backend": "merged register-bank pressure and operand-delivery differential latency",
                },
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="register_file_summary",
                value=derived,
                evidence_tier=EvidenceTier.COUPLED_INFERENCE,
                fit_status=FitStatus.UNDERCONSTRAINED,
                uncertainty=UncertaintyCategory.MULTI_FIT,
                mapping_contract="cross-probe register-file summary for simulator operand-delivery model (candidate)",
                assumptions=assumptions,
                coupled_with=["register_file.register_bank_sweep", "register_file.register_latency"],
            ),
        )
    ]
