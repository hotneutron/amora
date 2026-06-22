"""Internal NVIDIA baseline probe registry and runner."""

from __future__ import annotations

from collections.abc import Callable

from amora.backends.nvidia.cuda import NvidiaCapabilities, discover_capabilities
from amora.probes.nvidia.baseline.arithmetic_latency.dependent_chain import (
    run as run_dependent_chain,
)
from amora.probes.nvidia.baseline.arithmetic_throughput.independent_chains import (
    run as run_independent_chains,
)
from amora.probes.nvidia.baseline.shared_memory.analyze import run as run_shared_analyze
from amora.probes.nvidia.baseline.shared_memory.bank_stride import run as run_bank_stride
from amora.probes.nvidia.baseline.shared_memory.pointer_chase import run as run_pointer_chase
from amora.probes.nvidia.baseline.topology.device_attributes import run as run_device_attributes
from amora.probes.nvidia.baseline.topology.occupancy import run as run_occupancy
from amora.probes.nvidia.baseline.topology.persistent_cta import run as run_persistent_cta
from amora.schemas.results import ProbeResult, ToolContext


ProbeRunner = Callable[[NvidiaCapabilities], list[ProbeResult]]


PROBES: dict[str, ProbeRunner] = {
    "topology.device_attributes": run_device_attributes,
    "topology.occupancy": run_occupancy,
    "topology.persistent_cta": run_persistent_cta,
    "arithmetic_latency.dependent_chain": run_dependent_chain,
    "arithmetic_throughput.independent_chains": run_independent_chains,
    "shared_memory.pointer_chase": run_pointer_chase,
    "shared_memory.bank_stride": run_bank_stride,
    "shared_memory.analyze": run_shared_analyze,
}

IMPLEMENTED_PROBES: frozenset[str] = frozenset({"topology.device_attributes"})

PLANNED_PROBES: tuple[str, ...] = (
    "topology.device_attributes",
    "topology.occupancy",
    "topology.persistent_cta",
    "arithmetic_latency.dependent_chain",
    "arithmetic_throughput.independent_chains",
    "shared_memory.pointer_chase",
    "shared_memory.bank_stride",
    "shared_memory.analyze",
)


def list_probes() -> list[dict[str, object]]:
    return [
        {
            "probe_id": probe_id,
            "implemented": probe_id in IMPLEMENTED_PROBES,
            "runner_available": probe_id in PROBES,
            "status": "implemented" if probe_id in IMPLEMENTED_PROBES else "planned_unsupported",
        }
        for probe_id in PLANNED_PROBES
    ]


def run_probe(probe_id: str, capabilities: NvidiaCapabilities | None = None) -> list[ProbeResult]:
    caps = capabilities or discover_capabilities()
    if probe_id in PROBES:
        return PROBES[probe_id](caps)
    return [
        ProbeResult.unsupported(
            probe_id,
            "probe is planned but not implemented in the baseline cutline",
            tool_context=ToolContext(tools=caps.to_dict()),
        )
    ]


def run_all(capabilities: NvidiaCapabilities | None = None) -> list[ProbeResult]:
    caps = capabilities or discover_capabilities()
    results: list[ProbeResult] = []
    for probe_id in PLANNED_PROBES:
        results.extend(run_probe(probe_id, caps))
    return results
