"""AMORA-owned deterministic materializer for canonical PPP benchmarks."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from amora.benchmarking.materialize import case_key, canonical_json, shape_key
from amora.benchmarking.schema import BenchmarkCase
from benchmark_generators.ppp_canonical.cases import candidate_items
from benchmark_generators.ppp_canonical.kernels import CANONICAL_KERNELS, KernelSpec
from benchmark_generators.ppp_canonical.presets import PRESETS


GENERATOR_ROOT = Path(__file__).resolve().parent


def _git_revision(path: Path) -> tuple[str | None, bool | None]:
    try:
        root = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
        if root.returncode != 0:
            return None, None
        repository = Path(root.stdout.strip())
        relative_path = path.relative_to(repository)
        revision = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if revision.returncode != 0:
            return None, None
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain",
                "--",
                str(relative_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None, None
    return revision.stdout.strip() or None, bool(dirty.stdout.strip())


def _source_sha256(path: Path) -> str:
    digest = sha256()
    for source in sorted(path.rglob("*.py")):
        digest.update(source.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _allocation(case_count: int, capacities: list[int]) -> list[int]:
    if case_count <= 0:
        raise ValueError("case_count must be positive")
    if case_count > sum(capacities):
        raise ValueError(
            f"requested {case_count} cases, only {sum(capacities)} unique shapes available"
        )
    allocation = [0] * len(capacities)
    remaining = case_count
    active = list(range(len(capacities)))
    while remaining and active:
        share, remainder = divmod(remaining, len(active))
        next_active: list[int] = []
        for ordinal, index in enumerate(active):
            take = min(capacities[index] - allocation[index], share + (ordinal < remainder))
            allocation[index] += take
            remaining -= take
            if allocation[index] < capacities[index]:
                next_active.append(index)
        if remaining == 0:
            break
        if next_active == active:
            raise ValueError("unable to allocate requested benchmark cases")
        active = next_active
    if remaining:
        raise ValueError("unable to allocate requested benchmark cases")
    return allocation


def _candidate_offset(*, candidate_count: int, seed: int, kernel_id: str) -> int:
    if candidate_count == 0:
        return 0
    payload = f"{seed}:{kernel_id}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big") % candidate_count


@dataclass(frozen=True)
class PPPCanonicalDefinition:
    benchmark_id: str = "ppp_canonical"
    benchmark_revision: int = 1
    definition_kind: str = "generator"

    @property
    def presets(self) -> dict[str, dict[str, int]]:
        return PRESETS

    def describe(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_revision": self.benchmark_revision,
            "definition_kind": self.definition_kind,
            "description": "AMORA-owned canonical PPP kernel-and-shape generator",
            "kernels": [spec.kernel_id for spec in CANONICAL_KERNELS],
            "presets": sorted(PRESETS),
        }

    def materialize(
        self,
        *,
        target: dict[str, str],
        case_count: int,
        seed: int,
    ) -> tuple[list[BenchmarkCase], dict[str, Any]]:
        candidate_sets = [candidate_items(spec) for spec in CANONICAL_KERNELS]
        allocations = _allocation(case_count, [len(candidates) for candidates in candidate_sets])
        git_commit, git_dirty = _git_revision(GENERATOR_ROOT)
        source_sha = _source_sha256(GENERATOR_ROOT)
        generator = {
            "module": "benchmark_generators.ppp_canonical",
            "revision": self.benchmark_revision,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "source_sha256": source_sha,
            "seed": seed,
            "allocation": "balanced_per_kernel",
            "case_count_requested": case_count,
            "curated_classes": [],
            "generated_class": "sweep",
        }
        generator["generator_digest"] = sha256(
            canonical_json(generator).encode("utf-8")
        ).hexdigest()

        cases: list[BenchmarkCase] = []
        for spec, candidates, allocated in zip(CANONICAL_KERNELS, candidate_sets, allocations):
            if allocated > len(candidates):
                raise ValueError(
                    f"{spec.kernel_id}: requested {allocated} shapes, "
                    f"only {len(candidates)} candidates available"
                )
            offset = _candidate_offset(
                candidate_count=len(candidates),
                seed=seed,
                kernel_id=spec.kernel_id,
            )
            selected = [
                candidates[(offset + index) % len(candidates)]
                for index in range(allocated)
            ]
            cases.extend(
                self._case_from_item(
                    spec=spec,
                    item=item,
                    target=target,
                    generator=generator,
                )
                for item in selected
            )
        return cases, generator

    def _case_from_item(
        self,
        *,
        spec: KernelSpec,
        item: dict[str, Any],
        target: dict[str, str],
        generator: dict[str, Any],
    ) -> BenchmarkCase:
        shape = dict(item["shape"])
        return BenchmarkCase(
            case_key=case_key(
                benchmark_id=self.benchmark_id,
                benchmark_revision=self.benchmark_revision,
                target=target,
                kernel_id=spec.kernel_id,
                kernel_revision=spec.kernel_revision,
                shape=shape,
            ),
            benchmark_id=self.benchmark_id,
            benchmark_revision=self.benchmark_revision,
            definition_kind=self.definition_kind,
            kernel_id=spec.kernel_id,
            kernel_revision=spec.kernel_revision,
            shape=shape,
            shape_key=shape_key(shape),
            shape_class=item["shape_class"],
            axis_tags=tuple(item["axis_tags"]),
            regime_tags=tuple(item["regime_tags"]),
            tags=tuple(item["tags"]),
            execution_contract={
                "measurement_semantics": spec.measurement_semantics,
                "kernel_name_hw": spec.kernel_name_hw,
                "primary_resource": spec.primary_resource,
                "replay_contract_revision": 1,
            },
            case_generation={
                "generator_digest": generator["generator_digest"],
                "generator_git_commit": generator["git_commit"],
                "generator_source_sha256": generator["source_sha256"],
                "seed": generator["seed"],
            },
        )


PPP_CANONICAL = PPPCanonicalDefinition()
