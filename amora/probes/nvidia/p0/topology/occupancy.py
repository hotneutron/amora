"""Occupancy sweep planning and resource-limit fitting."""

from __future__ import annotations

from dataclasses import dataclass

from amora.schemas.results import EvidenceTier, ParameterEstimate, ProbeResult


@dataclass(frozen=True)
class OccupancyPoint:
    threads_per_block: int
    registers_per_thread: int
    dynamic_shared_memory_bytes: int

    @property
    def warps_per_block(self) -> int:
        return (self.threads_per_block + 31) // 32


def generate_occupancy_points(
    block_sizes: tuple[int, ...] = (32, 64, 128, 256, 512, 1024),
    register_counts: tuple[int, ...] = (16, 32, 64, 96, 128),
    shared_memory_sizes: tuple[int, ...] = (0, 1024, 8192, 32768, 65536),
) -> list[OccupancyPoint]:
    return [
        OccupancyPoint(block_size, reg_count, smem_size)
        for block_size in block_sizes
        for reg_count in register_counts
        for smem_size in shared_memory_sizes
    ]


def fit_residency_limit(
    point: OccupancyPoint,
    attributes: dict[str, int],
) -> dict[str, int | str]:
    """Fit the tightest runtime-visible occupancy limit for one launch shape."""

    max_threads = attributes["max_threads_per_multiprocessor"]
    max_blocks = attributes["max_blocks_per_multiprocessor"]
    max_registers = attributes["regs_per_multiprocessor"]
    max_shared = attributes["shared_memory_per_multiprocessor"]

    by_threads = max_threads // point.threads_per_block
    regs_per_block = point.threads_per_block * point.registers_per_thread
    by_registers = max_registers // regs_per_block if regs_per_block else max_blocks
    by_shared = (
        max_shared // point.dynamic_shared_memory_bytes
        if point.dynamic_shared_memory_bytes
        else max_blocks
    )
    candidates = {
        "threads": by_threads,
        "blocks": max_blocks,
        "registers": by_registers,
        "shared_memory": by_shared,
    }
    limiting_resource = min(candidates, key=candidates.get)
    return {
        "resident_blocks": max(0, candidates[limiting_resource]),
        "limiting_resource": limiting_resource,
    }


def occupancy_plan_probe(attributes: dict[str, int] | None = None) -> ProbeResult:
    """Create the P0 occupancy sweep plan and optional fitted summary."""

    points = generate_occupancy_points()
    measurements: dict[str, object] = {
        "sweep_points": [point.__dict__ | {"warps_per_block": point.warps_per_block} for point in points]
    }
    estimates: list[ParameterEstimate] = []

    if attributes is not None:
        fitted = [
            fit_residency_limit(point, attributes)
            | {"threads_per_block": point.threads_per_block}
            for point in points
        ]
        measurements["fitted_points"] = fitted
        estimates = [
            ParameterEstimate(
                name="shader_core_config::max_cta_per_core",
                value=attributes["max_blocks_per_multiprocessor"],
                evidence=EvidenceTier.DIRECT_METADATA,
                confidence=0.9,
                risk="low",
                notes=("Cross-check with persistent_cta.cu before final calibration.",),
            )
        ]

    return ProbeResult(
        name="topology/occupancy.py",
        tier="P0",
        status="planned" if attributes is None else "ok",
        measurements=measurements,
        estimates=estimates,
    )
