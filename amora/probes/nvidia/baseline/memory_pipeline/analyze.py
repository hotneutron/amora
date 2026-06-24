"""Memory-pipeline cross-probe analyzer (P2, Phase B).

Merges the lane-pattern coalescing scalar and the outstanding-request saturation
knee into one simulator-facing memory-pipeline record. Inherits the weakest fit
status of its inputs (weakest-fit merge) and stays a coupled inference, per the
P2 methodology.
"""

from __future__ import annotations

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.probes.nvidia.baseline.memory_pipeline import lane_patterns, outstanding_requests
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


PROBE_ID = "memory_pipeline.analyze"

_COUPLED = ["memory_pipeline.lane_patterns", "memory_pipeline.outstanding_requests"]

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
    lanes = lane_patterns.run(capabilities)[0]
    outstanding = outstanding_requests.run(capabilities)[0]
    inputs = {"lane_patterns": lanes, "outstanding_requests": outstanding}

    unsupported = [
        name
        for name, r in inputs.items()
        if r.raw_observation.evidence_tier == EvidenceTier.UNSUPPORTED
    ]
    if unsupported:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"memory-pipeline analyzer cannot run: missing inputs from {', '.join(unsupported)}",
                tool_context=_tool_context(capabilities),
                raw_values={
                    name: r.raw_observation.evidence_tier.value
                    for name, r in inputs.items()
                },
            )
        ]

    sectors_per_request = lanes.raw_observation.metrics.get("sectors_per_request")
    effective_outstanding = outstanding.raw_observation.metrics.get(
        "effective_outstanding_requests"
    )

    merged_fit = min(
        (r.normalized_measurement.fit_status for r in inputs.values()),
        key=_FIT_ORDER.index,
    )

    derived = {
        "coalescing_sectors_per_request": sectors_per_request,
        "effective_outstanding_requests": effective_outstanding,
    }
    values = {
        "lane_patterns": {
            "binary_sha256": lanes.identity.binary_hash,
            "sectors_per_request": sectors_per_request,
        },
        "outstanding_requests": {
            "binary_sha256": outstanding.identity.binary_hash,
            "effective_outstanding_requests": effective_outstanding,
        },
        "derived": derived,
    }
    assumptions = [
        "merges lane-pattern coalescing (sectors/request) with the outstanding-request knee",
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
                source="amora.probes.nvidia.baseline.memory_pipeline.analyze",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="memory_pipeline_summary",
                value=derived,
                fit_status=merged_fit,
                uncertainty=UncertaintyCategory.BOUNDED_RANGE,
                assumptions=assumptions,
                coupled_with=_COUPLED,
            ),
            backend_interpretation=BackendInterpretation(
                concept="memory_pipeline_summary",
                interpretation={
                    "nvidia_backend": "merged memory-pipeline characterization from coalescing and outstanding-request probes",
                },
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="memory_pipeline_summary",
                value=derived,
                evidence_tier=EvidenceTier.COUPLED_INFERENCE,
                fit_status=merged_fit,
                uncertainty=UncertaintyCategory.BOUNDED_RANGE,
                mapping_contract="cross-probe memory-pipeline summary for simulator coalescing + load/store-queue parameters",
                assumptions=assumptions,
                coupled_with=_COUPLED,
            ),
        )
    ]
