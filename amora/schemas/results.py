"""Dataclasses used by probe implementations and report renderers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceTier(str, Enum):
    """How directly a measurement supports a parameter estimate."""

    DIRECT_METADATA = "direct_metadata"
    DIRECT_COUNTER = "direct_counter"
    TIMING_DIRECT = "timing_direct"
    INSTRUMENTED_STREAM = "instrumented_stream"
    COUPLED_INFERENCE = "coupled_inference"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ParameterEstimate:
    """A simulator-parameter estimate derived from one or more observations."""

    name: str
    value: Any
    evidence: EvidenceTier
    confidence: float
    unit: str | None = None
    risk: str = "medium"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "evidence": self.evidence.value,
            "confidence": self.confidence,
            "unit": self.unit,
            "risk": self.risk,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ProbeArtifact:
    """A source, binary, disassembly, or raw-output artifact."""

    kind: str
    path: str
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "path": self.path, "sha256": self.sha256}


@dataclass
class ProbeResult:
    """Normalized output from one AMORA probe."""

    name: str
    tier: str
    status: str
    measurements: dict[str, Any] = field(default_factory=dict)
    estimates: list[ParameterEstimate] = field(default_factory=list)
    artifacts: list[ProbeArtifact] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "status": self.status,
            "measurements": self.measurements,
            "estimates": [estimate.to_dict() for estimate in self.estimates],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "warnings": self.warnings,
        }


@dataclass
class HardwareProfile:
    """A target profile assembled from multiple probe results."""

    target: dict[str, Any]
    raw_results: list[ProbeResult] = field(default_factory=list)

    def repo_parameter_estimates(self) -> dict[str, Any]:
        estimates: dict[str, Any] = {}
        for result in self.raw_results:
            for estimate in result.estimates:
                estimates[estimate.name] = estimate.value
        return estimates

    def confidence(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for result in self.raw_results:
            for estimate in result.estimates:
                values[estimate.name] = estimate.confidence
        return values

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "repo_parameter_estimates": self.repo_parameter_estimates(),
            "confidence": self.confidence(),
            "raw_results": [result.to_dict() for result in self.raw_results],
        }
