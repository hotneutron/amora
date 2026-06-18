"""Source registration for P0 shared-memory probes."""

from __future__ import annotations

from amora.core.artifacts import read_package_text, sha256_text
from amora.schemas.results import ProbeArtifact, ProbeResult


SHARED_MEMORY_SOURCES = ("pointer_chase.cu", "bank_stride.cu")


def shared_memory_source_probe() -> ProbeResult:
    artifacts: list[ProbeArtifact] = []
    for source in SHARED_MEMORY_SOURCES:
        text = read_package_text(__package__, source)
        artifacts.append(
            ProbeArtifact(
                kind="cuda_source",
                path=f"amora/probes/nvidia/p0/shared_memory/{source}",
                sha256=sha256_text(text),
            )
        )

    return ProbeResult(
        name="shared_memory/source_generation",
        tier="P0",
        status="planned",
        artifacts=artifacts,
    )
