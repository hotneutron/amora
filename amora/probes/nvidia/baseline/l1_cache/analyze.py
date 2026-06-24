"""L1 cache-path cross-probe analyzer (P1).

Merges the L1 pointer-chase hit latency, working-set capacity knee, and
conflict-set associativity into one simulator-facing cache record. Inherits the
weakest fit status of its inputs and stays bounded/behavioral when any input is
underconstrained, per the P1 methodology.
"""

from __future__ import annotations

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.probes.nvidia.baseline.l1_cache import conflict_sets, pointer_chase, working_set
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


PROBE_ID = "l1_cache.analyze"


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    pchase = pointer_chase.run(capabilities)[0]
    wset = working_set.run(capabilities)[0]
    conflict = conflict_sets.run(capabilities)[0]
    inputs = {"pointer_chase": pchase, "working_set": wset, "conflict_sets": conflict}

    unsupported = [name for name, r in inputs.items()
                   if r.raw_observation.evidence_tier == EvidenceTier.UNSUPPORTED]
    if unsupported:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"L1 analyzer cannot run: missing inputs from {', '.join(unsupported)}",
                tool_context=_tool_context(capabilities),
                raw_values={name: r.raw_observation.evidence_tier.value
                            for name, r in inputs.items()},
            )
        ]

    hit_latency = pchase.raw_observation.metrics.get("l1_hit_cycles_per_load")
    capacity = wset.normalized_measurement.value
    assoc = conflict.normalized_measurement.value

    # The merged fit is the weakest of the contributing fits.
    fit_order = [
        FitStatus.UNSUPPORTED,
        FitStatus.UNDERCONSTRAINED,
        FitStatus.BEHAVIORAL_ONLY,
        FitStatus.BOUNDED,
        FitStatus.CONDITIONALLY_IDENTIFIED,
        FitStatus.UNIQUELY_IDENTIFIED,
        FitStatus.DIRECT,
    ]
    merged_fit = min(
        (r.normalized_measurement.fit_status for r in inputs.values()),
        key=fit_order.index,
    )

    derived = {
        "l1_hit_latency_cycles": hit_latency,
        "l1_effective_capacity_kb": capacity,
        "l1_effective_associativity_ways": assoc,
    }
    values = {
        "pointer_chase": {
            "binary_sha256": pchase.identity.binary_hash,
            "l1_hit_cycles_per_load": hit_latency,
        },
        "working_set": {
            "binary_sha256": wset.identity.binary_hash,
            "effective_capacity": capacity,
        },
        "conflict_sets": {
            "binary_sha256": conflict.identity.binary_hash,
            "effective_associativity": assoc,
        },
        "derived": derived,
    }
    assumptions = [
        "merges L1 hit latency, capacity knee, and associativity knee",
        "merged fit status is the weakest of the contributing probe fits",
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
                source="amora.probes.nvidia.baseline.l1_cache.analyze",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="l1_cache_summary",
                value=derived,
                fit_status=merged_fit,
                uncertainty=UncertaintyCategory.BOUNDED_RANGE,
                assumptions=assumptions,
                coupled_with=["l1_cache.pointer_chase", "l1_cache.working_set", "l1_cache.conflict_sets"],
            ),
            backend_interpretation=BackendInterpretation(
                concept="l1_cache_summary",
                interpretation={
                    "nvidia_backend": "merged L1 path characterization from latency, capacity, and conflict probes",
                },
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="l1d_cache_summary",
                value=derived,
                evidence_tier=EvidenceTier.COUPLED_INFERENCE,
                fit_status=merged_fit,
                uncertainty=UncertaintyCategory.BOUNDED_RANGE,
                mapping_contract="cross-probe L1 summary for simulator L1 cache model parameters",
                assumptions=assumptions,
                coupled_with=["l1_cache.pointer_chase", "l1_cache.working_set", "l1_cache.conflict_sets"],
            ),
        )
    ]
