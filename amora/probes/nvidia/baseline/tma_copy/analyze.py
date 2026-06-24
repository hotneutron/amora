"""TMA / async-copy cross-probe analyzer (P3, Phase D).

Merges the async-copy tile latency and the peak async-copy throughput into one
simulator-facing async-copy record. Inherits the weakest fit status of its
inputs (weakest-fit merge) and stays a bounded coupled inference.
"""

from __future__ import annotations

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.probes.nvidia.baseline.tma_copy import async_copy_latency, tma_transfer_sweep
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


PROBE_ID = "tma_copy.analyze"

_COUPLED = [
    "tma_copy.async_copy_latency",
    "tma_copy.tma_transfer_sweep",
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
    latency = async_copy_latency.run(capabilities)[0]
    throughput = tma_transfer_sweep.run(capabilities)[0]
    inputs = {
        "async_copy_latency": latency,
        "tma_transfer_sweep": throughput,
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
                f"async-copy analyzer cannot run: missing inputs from {', '.join(unsupported)}",
                tool_context=_tool_context(capabilities),
                raw_values={
                    name: r.raw_observation.evidence_tier.value
                    for name, r in inputs.items()
                },
            )
        ]

    async_copy_tile_latency = latency.raw_observation.metrics.get("cycles_per_tile")
    async_copy_peak_gbps = throughput.raw_observation.metrics.get("peak_gbps")

    merged_fit = min(
        (r.normalized_measurement.fit_status for r in inputs.values()),
        key=_FIT_ORDER.index,
    )

    derived = {
        "async_copy_tile_latency": async_copy_tile_latency,
        "async_copy_peak_gbps": async_copy_peak_gbps,
    }
    values = {
        "async_copy_latency": {
            "binary_sha256": latency.identity.binary_hash,
            "async_copy_tile_latency": async_copy_tile_latency,
        },
        "tma_transfer_sweep": {
            "binary_sha256": throughput.identity.binary_hash,
            "async_copy_peak_gbps": async_copy_peak_gbps,
        },
        "derived": derived,
    }
    assumptions = [
        "merges async-copy tile latency and peak async-copy throughput",
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
                source="amora.probes.nvidia.baseline.tma_copy.analyze",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="async_copy_summary",
                value=derived,
                fit_status=merged_fit,
                uncertainty=UncertaintyCategory.BOUNDED_RANGE,
                assumptions=assumptions,
                coupled_with=_COUPLED,
            ),
            backend_interpretation=BackendInterpretation(
                concept="async_copy_summary",
                interpretation={
                    "nvidia_backend": "merged async-copy characterization from latency and throughput probes",
                },
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="async_copy_summary",
                value=derived,
                evidence_tier=EvidenceTier.COUPLED_INFERENCE,
                fit_status=merged_fit,
                uncertainty=UncertaintyCategory.BOUNDED_RANGE,
                mapping_contract="cross-probe async-copy summary for simulator async-copy latency + throughput parameters",
                assumptions=assumptions,
                coupled_with=_COUPLED,
            ),
        )
    ]
