"""Occupancy planning sweep (CPU-only, no kernel launch)."""

from __future__ import annotations

from dataclasses import dataclass

from amora.backends.nvidia.cuda import NvidiaCapabilities
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


PROBE_ID = "topology.occupancy"

DEFAULT_BLOCK_SIZES: tuple[int, ...] = (32, 64, 128, 256, 512, 1024)
DEFAULT_REGISTERS_PER_THREAD: tuple[int, ...] = (16, 32, 64, 96, 128)
DEFAULT_SHARED_MEMORY_BYTES: tuple[int, ...] = (0, 1024, 8192, 32768)


@dataclass(frozen=True)
class OccupancyPoint:
    threads_per_block: int
    registers_per_thread: int
    dynamic_shared_memory_bytes: int

    @property
    def warps_per_block(self) -> int:
        return (self.threads_per_block + 31) // 32

    def to_dict(self) -> dict[str, int]:
        return {
            "threads_per_block": self.threads_per_block,
            "warps_per_block": self.warps_per_block,
            "registers_per_thread": self.registers_per_thread,
            "dynamic_shared_memory_bytes": self.dynamic_shared_memory_bytes,
        }


def generate_occupancy_points(
    block_sizes: tuple[int, ...] = DEFAULT_BLOCK_SIZES,
    register_counts: tuple[int, ...] = DEFAULT_REGISTERS_PER_THREAD,
    shared_memory_sizes: tuple[int, ...] = DEFAULT_SHARED_MEMORY_BYTES,
) -> list[OccupancyPoint]:
    return [
        OccupancyPoint(block, regs, smem)
        for block in block_sizes
        for regs in register_counts
        for smem in shared_memory_sizes
    ]


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    points = generate_occupancy_points()
    sweep = [point.to_dict() for point in points]
    values = {
        "block_sizes": list(DEFAULT_BLOCK_SIZES),
        "registers_per_thread": list(DEFAULT_REGISTERS_PER_THREAD),
        "dynamic_shared_memory_bytes": list(DEFAULT_SHARED_MEMORY_BYTES),
        "sweep_points": sweep,
        "point_count": len(sweep),
    }
    assumptions = [
        "occupancy sweep is a planning artifact; resident-block fitting requires CUDA Occupancy API",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(probe_id=PROBE_ID),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(mode="planning"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.DIRECT_METADATA,
                values=values,
                source="amora.probes.nvidia.baseline.topology.occupancy",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="occupancy_sweep_plan",
                value=values,
                fit_status=FitStatus.DIRECT,
                uncertainty=UncertaintyCategory.STABLE_SCALAR,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="cuda_launch_shape_grid",
                interpretation={
                    "nvidia_backend": "block/register/shared-memory cross-product for downstream occupancy fits",
                },
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="launch_shape_sweep",
                value=values,
                evidence_tier=EvidenceTier.DIRECT_METADATA,
                fit_status=FitStatus.DIRECT,
                uncertainty=UncertaintyCategory.STABLE_SCALAR,
                mapping_contract="planning artifact; not a structural simulator parameter",
                assumptions=assumptions,
            ),
        )
    ]
