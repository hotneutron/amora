"""Shared-memory bank-stride probe placeholder."""

from __future__ import annotations

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.schemas.results import ProbeResult, ToolContext


PROBE_ID = "shared_memory.bank_stride"


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    return [
        ProbeResult.unsupported(
            PROBE_ID,
            "shared-memory bank-stride CUDA probe is not implemented in the baseline cutline",
            tool_context=ToolContext(tools=capabilities.to_dict()),
        )
    ]
