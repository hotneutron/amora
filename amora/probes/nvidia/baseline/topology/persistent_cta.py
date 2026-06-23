"""Persistent CTA residency probe (planned). Registers CUDA source hash."""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.probes.nvidia.baseline._sources import source_descriptor
from amora.schemas.results import ProbeResult, ToolContext


PROBE_ID = "topology.persistent_cta"
SOURCE = Path(__file__).with_name("persistent_cta.cu")


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    return [
        ProbeResult.unsupported(
            PROBE_ID,
            "persistent CTA CUDA probe is not implemented in the baseline cutline",
            tool_context=ToolContext(tools=capabilities.to_dict()),
            raw_values={"registered_source": source_descriptor(SOURCE)},
        )
    ]
