"""Independent arithmetic throughput probe placeholder."""

from __future__ import annotations

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.schemas.results import ProbeResult, ToolContext


PROBE_ID = "arithmetic_throughput.independent_chains"


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    return [
        ProbeResult.unsupported(
            PROBE_ID,
            "independent-chain CUDA throughput probe is not implemented in the baseline cutline",
            tool_context=ToolContext(tools=capabilities.to_dict()),
        )
    ]
