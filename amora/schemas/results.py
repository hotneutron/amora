"""Dataclasses used by probe implementations and report renderers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from amora.schemas.evidence import EvidenceTier, FitStatus, UncertaintyCategory


JsonDict = dict[str, Any]


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, tuple):
        return [_clean(v) for v in value]
    if isinstance(value, set):
        return sorted(_clean(v) for v in value)
    if hasattr(value, "value"):
        return value.value
    return value


@dataclass(frozen=True)
class ProbeIdentity:
    """Stable identity for a probe result."""

    probe_id: str
    backend: str = "nvidia_cuda"
    family: str = "baseline"
    source_hash: str | None = None
    binary_hash: str | None = None
    disassembly_hash: str | None = None


@dataclass(frozen=True)
class ToolContext:
    """Tool and device versions relevant to a probe run."""

    device: Mapping[str, Any] = field(default_factory=dict)
    tools: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LaunchDescriptor:
    """Kernel launch shape and execution mode."""

    grid: tuple[int, int, int] | None = None
    block: tuple[int, int, int] | None = None
    dynamic_shared_memory_bytes: int | None = None
    mode: str = "metadata"
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawObservation:
    """Raw evidence before normalization."""

    evidence_tier: EvidenceTier
    values: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    source: str | None = None
    unsupported_reason: str | None = None


@dataclass(frozen=True)
class NormalizedMeasurement:
    """Hardware-neutral measurement derived from raw evidence."""

    name: str
    value: Any = None
    unit: str | None = None
    fit_status: FitStatus = FitStatus.UNSUPPORTED
    uncertainty: UncertaintyCategory = UncertaintyCategory.INDETERMINATE
    variance: Mapping[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    coupled_with: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BackendInterpretation:
    """NVIDIA-specific interpretation of a normalized measurement."""

    concept: str
    interpretation: Mapping[str, Any] = field(default_factory=dict)
    metric_resolver: Mapping[str, Any] = field(default_factory=dict)
    sass_validation: Mapping[str, Any] = field(default_factory=dict)
    downgrade_reason: str | None = None


@dataclass(frozen=True)
class SimulatorEstimate:
    """Simulator-facing estimate derived through a mapping contract."""

    parameter: str
    value: Any = None
    unit: str | None = None
    evidence_tier: EvidenceTier = EvidenceTier.UNSUPPORTED
    fit_status: FitStatus = FitStatus.UNSUPPORTED
    uncertainty: UncertaintyCategory = UncertaintyCategory.INDETERMINATE
    mapping_contract: str = ""
    assumptions: list[str] = field(default_factory=list)
    coupled_with: list[str] = field(default_factory=list)
    unsupported_reason: str | None = None


@dataclass(frozen=True)
class ProbeResult:
    """Complete layered result for a probe."""

    identity: ProbeIdentity
    tool_context: ToolContext = field(default_factory=ToolContext)
    launch: LaunchDescriptor = field(default_factory=LaunchDescriptor)
    raw_observation: RawObservation = field(
        default_factory=lambda: RawObservation(EvidenceTier.UNSUPPORTED)
    )
    normalized_measurement: NormalizedMeasurement = field(
        default_factory=lambda: NormalizedMeasurement(name="unsupported")
    )
    backend_interpretation: BackendInterpretation = field(
        default_factory=lambda: BackendInterpretation(concept="unsupported")
    )
    simulator_estimate: SimulatorEstimate = field(
        default_factory=lambda: SimulatorEstimate(
            parameter="unsupported",
            mapping_contract="unsupported",
            unsupported_reason="unsupported",
        )
    )

    def to_dict(self) -> JsonDict:
        return _clean(asdict(self))

    @classmethod
    def unsupported(
        cls,
        probe_id: str,
        reason: str,
        *,
        backend: str = "nvidia_cuda",
        family: str = "baseline",
        tool_context: ToolContext | None = None,
    ) -> "ProbeResult":
        identity = ProbeIdentity(probe_id=probe_id, backend=backend, family=family)
        return cls(
            identity=identity,
            tool_context=tool_context or ToolContext(),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.UNSUPPORTED,
                unsupported_reason=reason,
            ),
            normalized_measurement=NormalizedMeasurement(
                name=probe_id,
                fit_status=FitStatus.UNSUPPORTED,
                uncertainty=UncertaintyCategory.INDETERMINATE,
                assumptions=[reason],
            ),
            backend_interpretation=BackendInterpretation(
                concept=probe_id,
                downgrade_reason=reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter=probe_id,
                evidence_tier=EvidenceTier.UNSUPPORTED,
                fit_status=FitStatus.UNSUPPORTED,
                uncertainty=UncertaintyCategory.INDETERMINATE,
                mapping_contract="unsupported",
                unsupported_reason=reason,
            ),
        )
