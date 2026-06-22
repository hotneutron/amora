"""Shared-memory baseline analysis placeholder."""

from __future__ import annotations

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.schemas.results import ProbeResult, ToolContext


PROBE_ID = "shared_memory.analyze"


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    return [
        ProbeResult.unsupported(
            PROBE_ID,
            "shared-memory analyzer requires pointer-chase and bank-stride inputs",
            tool_context=ToolContext(tools=capabilities.to_dict()),
        )
    ]
