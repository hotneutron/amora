"""Interconnect cross-probe analyzer (P3, Phase D).

Merges the injection-saturation bandwidth and the address-mapping behavioral
class into one simulator-facing interconnect record. Inherits the weakest fit
status of its inputs (weakest-fit merge) and stays a behavioral coupled
inference.
"""

from __future__ import annotations

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.probes.nvidia.baseline.interconnect import address_mapping, injection_rate
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


PROBE_ID = "interconnect.analyze"

_COUPLED = [
    "interconnect.injection_rate",
    "interconnect.address_mapping",
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
    injection = injection_rate.run(capabilities)[0]
    mapping = address_mapping.run(capabilities)[0]
    inputs = {
        "injection_rate": injection,
        "address_mapping": mapping,
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
                f"interconnect analyzer cannot run: missing inputs from {', '.join(unsupported)}",
                tool_context=_tool_context(capabilities),
                raw_values={
                    name: r.raw_observation.evidence_tier.value
                    for name, r in inputs.items()
                },
            )
        ]

    injection_saturation_gbps = injection.raw_observation.metrics.get("saturation_gbps")
    address_mapping_class = mapping.raw_observation.metrics.get("address_mapping_class")

    merged_fit = min(
        (r.normalized_measurement.fit_status for r in inputs.values()),
        key=_FIT_ORDER.index,
    )

    derived = {
        "injection_saturation_gbps": injection_saturation_gbps,
        "address_mapping_class": address_mapping_class,
    }
    values = {
        "injection_rate": {
            "binary_sha256": injection.identity.binary_hash,
            "injection_saturation_gbps": injection_saturation_gbps,
        },
        "address_mapping": {
            "binary_sha256": mapping.identity.binary_hash,
            "address_mapping_class": address_mapping_class,
        },
        "derived": derived,
    }
    assumptions = [
        "merges injection-saturation bandwidth and address-mapping behavioral class",
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
                source="amora.probes.nvidia.baseline.interconnect.analyze",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="interconnect_summary",
                value=derived,
                fit_status=merged_fit,
                uncertainty=UncertaintyCategory.BEHAVIORAL_CLASS,
                assumptions=assumptions,
                coupled_with=_COUPLED,
            ),
            backend_interpretation=BackendInterpretation(
                concept="interconnect_summary",
                interpretation={
                    "nvidia_backend": "merged interconnect characterization from injection-rate and address-mapping probes",
                },
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="interconnect_summary",
                value=derived,
                evidence_tier=EvidenceTier.COUPLED_INFERENCE,
                fit_status=merged_fit,
                uncertainty=UncertaintyCategory.BEHAVIORAL_CLASS,
                mapping_contract="cross-probe interconnect summary for simulator injection bandwidth + address-mapping parameters",
                assumptions=assumptions,
                coupled_with=_COUPLED,
            ),
        )
    ]
