"""Shared-memory cross-probe analyzer: merges pointer-chase + bank-stride results."""

from __future__ import annotations

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.probes.nvidia.baseline.shared_memory import bank_stride, pointer_chase
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


PROBE_ID = "shared_memory.analyze"


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    pchase_results = pointer_chase.run(capabilities)
    bank_results = bank_stride.run(capabilities)
    pchase = pchase_results[0]
    bank = bank_results[0]

    pchase_unsupported = pchase.raw_observation.evidence_tier == EvidenceTier.UNSUPPORTED
    bank_unsupported = bank.raw_observation.evidence_tier == EvidenceTier.UNSUPPORTED
    if pchase_unsupported or bank_unsupported:
        missing = []
        if pchase_unsupported:
            missing.append("pointer_chase")
        if bank_unsupported:
            missing.append("bank_stride")
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"shared-memory analyzer cannot run: missing inputs from {', '.join(missing)}",
                tool_context=_tool_context(capabilities),
                raw_values={
                    "pointer_chase_evidence": pchase.raw_observation.evidence_tier.value,
                    "bank_stride_evidence": bank.raw_observation.evidence_tier.value,
                },
            )
        ]

    pchase_metrics = pchase.raw_observation.metrics
    bank_metrics = bank.raw_observation.metrics
    cycles_per_load = float(pchase_metrics["cycles_per_load"])
    no_conflict = bank_metrics.get("no_conflict_cycles_per_access")
    full_conflict = bank_metrics.get("full_conflict_cycles_per_access")
    inferred_banks = bank_metrics.get("inferred_bank_count")
    serialization = (
        float(full_conflict) / float(no_conflict)
        if no_conflict and full_conflict and no_conflict > 0
        else None
    )

    values = {
        "pointer_chase": {
            "binary_sha256": pchase.identity.binary_hash,
            "cycles_per_load": cycles_per_load,
        },
        "bank_stride": {
            "binary_sha256": bank.identity.binary_hash,
            "no_conflict_cycles_per_access": no_conflict,
            "full_conflict_cycles_per_access": full_conflict,
            "inferred_bank_count": inferred_banks,
        },
        "derived": {
            "shared_load_latency_cycles": cycles_per_load,
            "bank_serialization_factor": serialization,
            "bank_count": inferred_banks,
        },
    }
    assumptions = [
        "consumes the cycles-per-load median from pointer_chase and the stride sweep from bank_stride",
        "bank_serialization_factor is full_conflict / no_conflict cycles-per-access",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(probe_id=PROBE_ID),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(mode="analysis"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.COUPLED_INFERENCE,
                values=values,
                metrics={
                    "shared_load_latency_cycles": cycles_per_load,
                    "bank_count": inferred_banks,
                    "bank_serialization_factor": serialization,
                },
                source="amora.probes.nvidia.baseline.shared_memory.analyze",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="shared_memory_summary",
                value=values["derived"],
                fit_status=FitStatus.UNIQUELY_IDENTIFIED,
                uncertainty=UncertaintyCategory.STABLE_SCALAR,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="shared_memory_summary",
                interpretation={
                    "nvidia_backend": "merged shared-memory characterization derived from pointer-chase and bank-stride probes",
                },
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="shared_memory_summary",
                value=values["derived"],
                evidence_tier=EvidenceTier.COUPLED_INFERENCE,
                fit_status=FitStatus.UNIQUELY_IDENTIFIED,
                uncertainty=UncertaintyCategory.STABLE_SCALAR,
                mapping_contract="cross-probe summary suitable for simulator shared-memory model parameters",
                assumptions=assumptions,
            ),
        )
    ]
