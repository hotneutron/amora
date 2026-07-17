"""Hardware basic-stat classification and deterministic size-rank assignment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from amora.benchmarking.schema import BenchmarkCase

NCU_BASIC_RECIPE = "ncu_basic_v1"
INSTRUCTION_LOGICAL = "inst_executed"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ClassificationResult:
    """One case's basic hardware classification evidence."""

    case_key: str
    status: str
    total_instructions: float | None
    kernel_name: str | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    resolved_metrics: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_key": self.case_key,
            "status": self.status,
            "total_instructions": self.total_instructions,
            "kernel_name": self.kernel_name,
            "metrics": dict(self.metrics),
            "resolved_metrics": dict(self.resolved_metrics),
            "provenance": dict(self.provenance),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ClassificationManifest:
    """Immutable classification overlay for one materialized case set."""

    case_set_digest: str
    target: Mapping[str, str]
    recipe: str
    instruction_logical: str
    results: tuple[ClassificationResult, ...]
    case_count_expected: int
    case_count_attempted: int
    case_coverage_complete: bool
    rank_assignments: Mapping[str, Mapping[str, Any]]
    rank_boundaries: Mapping[str, float | None]
    classification_digest: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_set_digest": self.case_set_digest,
            "target": dict(self.target),
            "recipe": self.recipe,
            "instruction_logical": self.instruction_logical,
            "classification_digest": self.classification_digest,
            "case_count_expected": self.case_count_expected,
            "case_count_attempted": self.case_count_attempted,
            "case_coverage_complete": self.case_coverage_complete,
            "rank_boundaries": dict(self.rank_boundaries),
            "results": [result.to_dict() for result in self.results],
            "rank_assignments": {
                case_key: dict(value)
                for case_key, value in sorted(self.rank_assignments.items())
            },
        }


def assign_size_ranks(
    results: Iterable[ClassificationResult],
) -> tuple[dict[str, dict[str, Any]], dict[str, float | None]]:
    """Assign deterministic equal-count small/medium/large instruction ranks."""

    valid = sorted(
        (
            result
            for result in results
            if result.status == "classified"
            and isinstance(result.total_instructions, (int, float))
        ),
        key=lambda result: (float(result.total_instructions), result.case_key),
    )
    assignments: dict[str, dict[str, Any]] = {}
    if not valid:
        return assignments, {"small_max": None, "medium_max": None, "large_max": None}

    base, remainder = divmod(len(valid), 3)
    rank_counts = [base + (index < remainder) for index in range(3)]
    ranks = ("small", "medium", "large")
    index = 0
    boundaries: dict[str, float | None] = {}
    for rank, count in zip(ranks, rank_counts):
        rank_results = valid[index:index + count]
        for ordinal, result in enumerate(rank_results, start=index):
            assignments[result.case_key] = {
                "size_rank": rank,
                "size_rank_ordinal": ordinal,
                "total_instructions": float(result.total_instructions),
            }
        boundaries[f"{rank}_max"] = (
            float(rank_results[-1].total_instructions) if rank_results else None
        )
        index += count
    return assignments, boundaries


def build_classification_manifest(
    *,
    case_set_digest: str,
    target: Mapping[str, str],
    results: Iterable[ClassificationResult],
    expected_case_keys: Iterable[str] | None = None,
    recipe: str = NCU_BASIC_RECIPE,
) -> ClassificationManifest:
    """Build a deterministic classification overlay with rank assignments."""

    ordered = tuple(sorted(results, key=lambda result: result.case_key))
    keys = [result.case_key for result in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("classification results contain duplicate case keys")
    expected = set(expected_case_keys or keys)
    if not set(keys) <= expected:
        raise ValueError("classification results include a case outside the manifest")
    complete = set(keys) == expected
    if complete:
        assignments, boundaries = assign_size_ranks(ordered)
    else:
        assignments = {}
        boundaries = {"small_max": None, "medium_max": None, "large_max": None}
    payload = {
        "schema_version": 1,
        "case_set_digest": case_set_digest,
        "target": dict(target),
        "recipe": recipe,
        "instruction_logical": INSTRUCTION_LOGICAL,
        "case_count_expected": len(expected),
        "case_count_attempted": len(ordered),
        "case_coverage_complete": complete,
        "rank_boundaries": boundaries,
        "results": [result.to_dict() for result in ordered],
        "rank_assignments": assignments,
    }
    digest = sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return ClassificationManifest(
        case_set_digest=case_set_digest,
        target=target,
        recipe=recipe,
        instruction_logical=INSTRUCTION_LOGICAL,
        results=ordered,
        case_count_expected=len(expected),
        case_count_attempted=len(ordered),
        case_coverage_complete=complete,
        rank_assignments=assignments,
        rank_boundaries=boundaries,
        classification_digest=digest,
    )


def classify_cases(
    cases: Iterable[BenchmarkCase],
    *,
    case_set_digest: str,
    target: Mapping[str, str],
    classify_case: Callable[[BenchmarkCase], ClassificationResult],
    expected_case_keys: Iterable[str] | None = None,
    recipe: str = NCU_BASIC_RECIPE,
) -> ClassificationManifest:
    """Classify every selected case and build its immutable rank overlay."""

    results = [classify_case(case) for case in cases]
    return build_classification_manifest(
        case_set_digest=case_set_digest,
        target=target,
        results=results,
        expected_case_keys=expected_case_keys,
        recipe=recipe,
    )


def write_classification_manifest(
    manifest: ClassificationManifest,
    path: str | Path,
) -> Path:
    """Write one classification overlay as canonical JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
