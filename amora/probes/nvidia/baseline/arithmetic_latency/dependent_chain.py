"""Dependent arithmetic latency probe placeholder."""

from __future__ import annotations

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.schemas.results import ProbeResult, ToolContext


PROBE_ID = "arithmetic_latency.dependent_chain"


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    return [
        ProbeResult.unsupported(
            PROBE_ID,
            "dependent-chain CUDA timing probe is not implemented in the baseline cutline",
            tool_context=ToolContext(tools=capabilities.to_dict()),
        )
    ]
