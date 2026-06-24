"""Global-memory cross-probe analyzer (P2, Phase B).

Merges the streaming peak bandwidth, partition-camping class, and DRAM
row-locality sensitivity into one simulator-facing global-memory record.
Inherits the weakest fit status of its inputs (weakest-fit merge) and stays a
bounded coupled inference, per the P2 methodology.
"""

from __future__ import annotations

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.probes.nvidia.baseline.global_memory import (
    partition_sweep,
    row_policy_sweep,
    streaming,
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


PROBE_ID = "global_memory.analyze"

_COUPLED = [
    "global_memory.streaming",
    "global_memory.partition_sweep",
    "global_memory.row_policy_sweep",
]

# Ordered weakest -> strongest, used to inherit the weakest contributing fit.
_FIT_ORDER = [
    FitStatus.UNSUPPORTED,
    FitStatus.UNDERCONSTRAINED,
    FitStatus.BEHAVIORAL_ONLY,
    FitStatus.BOUNDED,
    FitStatus.CONDITIONALLY_IDENTIFIED,
    FitStatus.UNIQUELY_IDENTIFIED,
    FitStatus.DIRECT,
]


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    stream = streaming.run(capabilities)[0]
    partition = partition_sweep.run(capabilities)[0]
    row_policy = row_policy_sweep.run(capabilities)[0]
    inputs = {
        "streaming": stream,
        "partition_sweep": partition,
        "row_policy_sweep": row_policy,
    }

    unsupported = [
        name
        for name, r in inputs.items()
        if r.raw_observation.evidence_tier == EvidenceTier.UNSUPPORTED
    ]
    if unsupported:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"global-memory analyzer cannot run: missing inputs from {', '.join(unsupported)}",
                tool_context=_tool_context(capabilities),
                raw_values={
                    name: r.raw_observation.evidence_tier.value
                    for name, r in inputs.items()
                },
            )
        ]

    peak_gbps = stream.raw_observation.metrics.get("peak_gbps")
    partition_class = partition.raw_observation.metrics.get("partition_camping_class")
    row_sensitivity = row_policy.raw_observation.metrics.get("row_locality_sensitivity")

    merged_fit = min(
        (r.normalized_measurement.fit_status for r in inputs.values()),
        key=_FIT_ORDER.index,
    )

    derived = {
        "peak_gbps": peak_gbps,
        "partition_class": partition_class,
        "row_locality_sensitivity": row_sensitivity,
    }
    values = {
        "streaming": {
            "binary_sha256": stream.identity.binary_hash,
            "peak_gbps": peak_gbps,
        },
        "partition_sweep": {
            "binary_sha256": partition.identity.binary_hash,
            "partition_class": partition_class,
        },
        "row_policy_sweep": {
            "binary_sha256": row_policy.identity.binary_hash,
            "row_locality_sensitivity": row_sensitivity,
        },
        "derived": derived,
    }
    assumptions = [
        "merges streaming peak bandwidth, partition-camping class, and row-locality sensitivity",
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
                source="amora.probes.nvidia.baseline.global_memory.analyze",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="global_memory_summary",
                value=derived,
                fit_status=merged_fit,
                uncertainty=UncertaintyCategory.BOUNDED_RANGE,
                assumptions=assumptions,
                coupled_with=_COUPLED,
            ),
            backend_interpretation=BackendInterpretation(
                concept="global_memory_summary",
                interpretation={
                    "nvidia_backend": "merged global-memory characterization from streaming, partition, and row-locality probes",
                },
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="global_memory_summary",
                value=derived,
                evidence_tier=EvidenceTier.COUPLED_INFERENCE,
                fit_status=merged_fit,
                uncertainty=UncertaintyCategory.BOUNDED_RANGE,
                mapping_contract="cross-probe global-memory summary for simulator DRAM bandwidth + partition + row-policy parameters",
                assumptions=assumptions,
                coupled_with=_COUPLED,
            ),
        )
    ]
