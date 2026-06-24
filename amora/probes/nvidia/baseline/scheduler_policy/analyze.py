"""Scheduler & issue cross-probe analyzer (P1).

Combines the ready-warp issue-scaling saturation point with the mixed-issue
overlap class into one scheduler behavioral record. Scheduler behavior is
reported as a behavioral class with the issue-capacity estimate attached as a
conditional value, per the P1 methodology.
"""

from __future__ import annotations

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.probes.nvidia.baseline.scheduler_policy import mixed_issue, ready_warps
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


PROBE_ID = "scheduler_policy.analyze"


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    ready = ready_warps.run(capabilities)[0]
    mixed = mixed_issue.run(capabilities)[0]
    inputs = {"ready_warps": ready, "mixed_issue": mixed}

    unsupported = [name for name, r in inputs.items()
                   if r.raw_observation.evidence_tier == EvidenceTier.UNSUPPORTED]
    if unsupported:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"scheduler analyzer cannot run: missing inputs from {', '.join(unsupported)}",
                tool_context=_tool_context(capabilities),
                raw_values={name: r.raw_observation.evidence_tier.value
                            for name, r in inputs.items()},
            )
        ]

    saturation_warps = ready.normalized_measurement.value
    overlap_class = mixed.normalized_measurement.value
    derived = {
        "issue_saturation_warps": saturation_warps,
        "mixed_issue_class": overlap_class,
        "peak_ops_per_cycle": ready.raw_observation.metrics.get("peak_ops_per_cycle"),
    }
    values = {
        "ready_warps": {
            "binary_sha256": ready.identity.binary_hash,
            "saturation_warps": saturation_warps,
        },
        "mixed_issue": {
            "binary_sha256": mixed.identity.binary_hash,
            "overlap_class": overlap_class,
            "overlap_ratio": mixed.raw_observation.metrics.get("overlap_ratio"),
        },
        "derived": derived,
    }
    assumptions = [
        "combines ready-warp issue saturation with mixed-issue overlap class",
        "scheduler policy reported as a behavioral class with a conditional issue-capacity value",
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
                source="amora.probes.nvidia.baseline.scheduler_policy.analyze",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="scheduler_summary",
                value=derived,
                fit_status=FitStatus.BEHAVIORAL_ONLY,
                uncertainty=UncertaintyCategory.BEHAVIORAL_CLASS,
                assumptions=assumptions,
                coupled_with=["scheduler_policy.ready_warps", "scheduler_policy.mixed_issue"],
            ),
            backend_interpretation=BackendInterpretation(
                concept="scheduler_summary",
                interpretation={
                    "nvidia_backend": "scheduler issue scaling and pipe-overlap behavior on one SM",
                },
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="gpgpu_scheduler_summary",
                value=derived,
                evidence_tier=EvidenceTier.COUPLED_INFERENCE,
                fit_status=FitStatus.BEHAVIORAL_ONLY,
                uncertainty=UncertaintyCategory.BEHAVIORAL_CLASS,
                mapping_contract="cross-probe scheduler summary for simulator scheduler model (behavioral)",
                assumptions=assumptions,
                coupled_with=["scheduler_policy.ready_warps", "scheduler_policy.mixed_issue"],
            ),
        )
    ]
