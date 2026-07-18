"""Data contracts for static and generated benchmark case sets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, tuple):
        return [_clean(item) for item in value]
    return value


@dataclass(frozen=True)
class BenchmarkCase:
    """One generated or statically declared benchmark case."""

    case_key: str
    benchmark_id: str
    benchmark_revision: int
    definition_kind: str
    kernel_id: str
    kernel_revision: int
    shape: Mapping[str, int]
    shape_key: str
    shape_class: str
    axis_tags: tuple[str, ...] = ()
    regime_tags: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    execution_contract: Mapping[str, Any] = field(default_factory=dict)
    case_generation: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass(frozen=True)
class CaseSetManifest:
    """Immutable materialization metadata and its ordered benchmark cases."""

    benchmark_id: str
    benchmark_revision: int
    definition_kind: str
    target: Mapping[str, str]
    generator: Mapping[str, Any]
    requested_case_count: int
    materialized_case_count: int
    cases: tuple[BenchmarkCase, ...]
    case_set_digest: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "benchmark_revision": self.benchmark_revision,
            "definition_kind": self.definition_kind,
            "target": _clean(dict(self.target)),
            "generator": _clean(dict(self.generator)),
            "case_count_requested": self.requested_case_count,
            "case_count_materialized": self.materialized_case_count,
            "case_set_digest": self.case_set_digest,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class DetailedCaseResult:
    """One backend's detailed evidence for a classified benchmark case."""

    case_key: str
    kernel_id: str
    size_rank: str
    backend: str
    status: str
    measurement: Mapping[str, Any] = field(default_factory=dict)
    logical_metrics: Mapping[str, Any] = field(default_factory=dict)
    resolved_metrics: Mapping[str, str] = field(default_factory=dict)
    raw_metrics: Mapping[str, float] = field(default_factory=dict)
    stall_histogram: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))
