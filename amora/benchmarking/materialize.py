"""Deterministically materialize benchmark definitions into case-set manifests."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from amora.benchmarking.schema import BenchmarkCase, CaseSetManifest


def canonical_json(value: Any) -> str:
    """Encode structured data deterministically for hashing and manifests."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_manifest(path: str | Path) -> CaseSetManifest:
    """Load a materialized case-set manifest from canonical JSON."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = tuple(
        BenchmarkCase(
            case_key=row["case_key"],
            benchmark_id=row["benchmark_id"],
            benchmark_revision=int(row["benchmark_revision"]),
            definition_kind=row["definition_kind"],
            kernel_id=row["kernel_id"],
            kernel_revision=int(row["kernel_revision"]),
            shape={key: int(value) for key, value in row["shape"].items()},
            shape_key=row["shape_key"],
            shape_class=row["shape_class"],
            axis_tags=tuple(row.get("axis_tags") or ()),
            regime_tags=tuple(row.get("regime_tags") or ()),
            tags=tuple(row.get("tags") or ()),
            execution_contract=row.get("execution_contract") or {},
            case_generation=row.get("case_generation") or {},
        )
        for row in data.get("cases") or ()
    )
    manifest = CaseSetManifest(
        benchmark_id=data["benchmark_id"],
        benchmark_revision=int(data["benchmark_revision"]),
        definition_kind=data["definition_kind"],
        target=data.get("target") or {},
        generator=data.get("generator") or {},
        requested_case_count=int(data["case_count_requested"]),
        materialized_case_count=int(data["case_count_materialized"]),
        cases=cases,
        case_set_digest=data["case_set_digest"],
        schema_version=int(data.get("schema_version", 1)),
    )
    if len(cases) != manifest.materialized_case_count:
        raise ValueError(
            f"{path}: declared {manifest.materialized_case_count} cases, found {len(cases)}"
        )
    return manifest


def shape_key(shape: dict[str, int]) -> str:
    """Return a stable, human-readable key with lexicographic dimensions."""

    return "_".join(f"{name}{shape[name]}" for name in sorted(shape))


def case_key(
    *,
    benchmark_id: str,
    benchmark_revision: int,
    target: dict[str, str],
    kernel_id: str,
    kernel_revision: int,
    shape: dict[str, int],
) -> str:
    """Return the stable identity for one benchmark case."""

    target_id = target.get("arch_profile") or target.get("hardware_sku") or "generic"
    return (
        f"{benchmark_id}:r{benchmark_revision}:{target_id}:{kernel_id}:"
        f"r{kernel_revision}:{shape_key(shape)}"
    )


def _case_payload(case: BenchmarkCase) -> dict[str, Any]:
    payload = case.to_dict()
    payload.pop("case_generation", None)
    return payload


def _manifest_digest(
    *,
    benchmark_id: str,
    benchmark_revision: int,
    definition_kind: str,
    target: dict[str, str],
    generator: dict[str, Any],
    requested_case_count: int,
    cases: list[BenchmarkCase],
) -> str:
    payload = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "benchmark_revision": benchmark_revision,
        "definition_kind": definition_kind,
        "target": target,
        "generator": generator,
        "case_count_requested": requested_case_count,
        "cases": [_case_payload(case) for case in cases],
    }
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def materialize_benchmark(
    benchmark_id: str,
    *,
    target: dict[str, str],
    case_count: int,
    seed: int,
) -> CaseSetManifest:
    """Materialize exactly ``case_count`` ordered cases from a definition."""

    from amora.benchmarking.registry import get_benchmark

    definition = get_benchmark(benchmark_id)
    cases, generator = definition.materialize(
        target=target,
        case_count=case_count,
        seed=seed,
    )
    ordered = sorted(cases, key=lambda case: case.case_key)
    keys = [case.case_key for case in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{benchmark_id}: duplicate materialized case keys")
    if len(ordered) != case_count:
        raise ValueError(
            f"{benchmark_id}: requested {case_count} cases, materialized {len(ordered)}"
        )
    digest = _manifest_digest(
        benchmark_id=definition.benchmark_id,
        benchmark_revision=definition.benchmark_revision,
        definition_kind=definition.definition_kind,
        target=target,
        generator=generator,
        requested_case_count=case_count,
        cases=ordered,
    )
    return CaseSetManifest(
        benchmark_id=definition.benchmark_id,
        benchmark_revision=definition.benchmark_revision,
        definition_kind=definition.definition_kind,
        target=target,
        generator=generator,
        requested_case_count=case_count,
        materialized_case_count=len(ordered),
        cases=tuple(ordered),
        case_set_digest=digest,
    )


def write_manifest(manifest: CaseSetManifest, path: str | Path) -> Path:
    """Write a materialized manifest with canonical formatting."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
