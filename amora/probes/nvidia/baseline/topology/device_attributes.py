"""CUDA-visible NVIDIA topology metadata probe."""

from __future__ import annotations

from typing import Any

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.archinfo import facts_for_capabilities
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


PROBE_ID = "topology.device_attributes"


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(
        device={
            "devices": [device.to_dict() for device in capabilities.devices],
            "gpu_available": capabilities.gpu_available,
        },
        tools={
            name: status.to_dict()
            for name, status in capabilities.tools.items()
        },
        environment={
            "backend": capabilities.backend,
            "cuda_available": capabilities.cuda_available,
            "unsupported_reasons": capabilities.unsupported_reasons,
        },
    )


def _unsupported(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    reason = "; ".join(capabilities.unsupported_reasons) or "CUDA metadata unavailable"
    return [
        ProbeResult.unsupported(
            PROBE_ID,
            reason,
            tool_context=_tool_context(capabilities),
        )
    ]


def _device_values(capabilities: NvidiaCapabilities) -> dict[str, Any]:
    first = capabilities.devices[0]
    return {
        "device_index": first.index,
        "device_name": first.name,
        "uuid": first.uuid,
        "driver_version": first.driver_version,
    }


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    if not capabilities.gpu_available:
        return _unsupported(capabilities)

    values = _device_values(capabilities)
    facts = facts_for_capabilities(capabilities)
    interpretation: dict[str, Any] = {
        "nvidia_backend": "device identity is available; resource limits need CUDA API helper in the next cutline"
    }
    if facts is not None:
        # Trust-and-verify anchor: attach the curated published facts so probes
        # and reports can cross-check runtime metadata against known specs.
        values = {**values, "published_facts": facts.to_dict()}
        interpretation["published_facts"] = facts.to_dict()
    results = [
        ProbeResult(
            identity=ProbeIdentity(probe_id=PROBE_ID),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(mode="metadata"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.DIRECT_METADATA,
                values=values,
                source="nvidia-smi",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="cuda_device_identity",
                value=values,
                unit=None,
                fit_status=FitStatus.DIRECT,
                uncertainty=UncertaintyCategory.STABLE_SCALAR,
                assumptions=[
                    "nvidia-smi identity metadata is treated as direct runtime metadata",
                    "published_facts are curated trust-and-verify anchors, not runtime measurements",
                ],
            ),
            backend_interpretation=BackendInterpretation(
                concept="runtime_visible_device_identity",
                interpretation=interpretation,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="device_identity",
                value=values,
                evidence_tier=EvidenceTier.DIRECT_METADATA,
                fit_status=FitStatus.DIRECT,
                uncertainty=UncertaintyCategory.STABLE_SCALAR,
                mapping_contract="identity metadata is recorded for traceability and is not a simulator structural parameter",
            ),
        )
    ]
    return results
