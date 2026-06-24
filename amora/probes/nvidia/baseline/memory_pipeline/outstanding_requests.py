"""Memory-pipeline outstanding-requests probe (P2).

Measures effective memory-level parallelism: each thread issues a swept number
of INDEPENDENT global loads before consuming them, and achieved bytes/cycle is
reported per setting. The saturation knee (smallest in-flight count reaching
>=95% of peak throughput) is the effective number of outstanding requests the
load/store pipeline sustains. SASS validation requires global loads (LDG) and
forbids shared/local spills.
"""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.runner import CudaUnavailable, run_kernel
from amora.backends.nvidia.sass import SassExpectation
from amora.probes.nvidia.baseline._sources import apply_sass_gating, collect_stall_attribution, source_descriptor
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


PROBE_ID = "memory_pipeline.outstanding_requests"
SOURCE = Path(__file__).with_name("outstanding_requests.cu")

# The timed loop must hit global memory (LDG) without shared or local spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_mem_outstanding",
    required_opcodes={"LDG": 1},
    forbidden_opcodes=("LDS", "STL"),
)


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def _saturation_knee(sweep: list[dict]) -> int | None:
    """Smallest in_flight reaching >=95% of peak bytes_per_cycle, or None."""

    if not sweep:
        return None
    peak = max(float(p["bytes_per_cycle"]) for p in sweep)
    if peak <= 0:
        return None
    ordered = sorted(sweep, key=lambda p: int(p["in_flight"]))
    for p in ordered:
        if float(p["bytes_per_cycle"]) >= 0.95 * peak:
            return int(p["in_flight"])
    return None


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    src_descriptor = source_descriptor(SOURCE)
    try:
        result = run_kernel(SOURCE, capabilities=capabilities, timeout=60, expectation=EXPECTATION)
    except CudaUnavailable as exc:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"memory-pipeline outstanding-requests probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    sweep = list(payload["sweep"])
    knee = _saturation_knee(sweep)
    fit = FitStatus.BOUNDED if knee is not None else FitStatus.UNDERCONSTRAINED

    # SASS gating: reject if the timed loop is not a global-load stream.
    sass = result.sass_validation
    decision, fit, uncertainty, downgrade_reason = apply_sass_gating(
        sass, EXPECTATION, fit, UncertaintyCategory.BOUNDED_RANGE
    )
    if decision == "reject":
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"SASS validation rejected the measurement: {sass.reason}",
                tool_context=_tool_context(capabilities),
                raw_values={
                    "registered_source": src_descriptor,
                    "sass": sass.to_dict(),
                },
            )
        ]

    stall_record = collect_stall_attribution(capabilities, SOURCE, kernel_name="amora_mem_outstanding")

    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    if sass is not None:
        values["sass"] = sass.to_dict()
    if stall_record is not None:
        values["stall_attribution"] = stall_record
    assumptions = [
        "each thread issues a swept number of independent global loads before consuming them",
        "a single wave of blocks is launched so throughput is bound by outstanding-request capacity",
        "saturation knee = smallest in-flight count reaching >=95% of peak bytes/cycle",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(
                probe_id=PROBE_ID,
                binary_hash=result.binary_sha256,
                disassembly_hash=sass.disassembly_hash if sass else None,
            ),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(grid=None, block=(256, 1, 1), mode="kernel"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "effective_outstanding_requests": knee,
                    "sweep_points": len(sweep),
                },
                units={"effective_outstanding_requests": "loads"},
                source="amora.probes.nvidia.baseline.memory_pipeline.outstanding_requests",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="effective_outstanding_requests",
                value=knee,
                unit="loads",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="memory_level_parallelism",
                interpretation={
                    "nvidia_backend": "in-flight independent loads at which memory throughput saturates",
                    "saturation_knee_loads": knee,
                    **({"dominant_stall": stall_record["dominant_stall"]} if stall_record else {}),
                },
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason
                if downgrade_reason is not None
                else (None if knee is not None else "no clear throughput saturation knee detected"),
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="ldst_queue_capacity",
                value=knee,
                unit="loads",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="outstanding-load saturation knee -> simulator load/store queue capacity (bounded)",
                assumptions=assumptions,
            ),
        )
    ]
