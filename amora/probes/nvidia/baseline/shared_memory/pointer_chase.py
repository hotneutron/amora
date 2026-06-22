"""Shared-memory pointer-chase latency probe placeholder."""

from __future__ import annotations

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.schemas.results import ProbeResult, ToolContext


PROBE_ID = "shared_memory.pointer_chase"


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    return [
        ProbeResult.unsupported(
            PROBE_ID,
            "shared-memory pointer-chase CUDA probe is not implemented in the baseline cutline",
            tool_context=ToolContext(tools=capabilities.to_dict()),
        )
    ]
