"""Scheduler ready-warp issue-scaling probe (P1).

Sweeps the number of ready warps on one SM and measures aggregate ops/cycle.
The saturation knee classifies effective issue capacity. Per the P1 methodology
the scheduler *policy* string is behavioral; only the issue-scaling curve and a
conditional saturation point are reported.
"""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.runner import CudaUnavailable, run_kernel
from amora.backends.nvidia.sass import SassExpectation
from amora.probes.nvidia.baseline._sources import (
    apply_sass_gating,
    downgrade_fit,
    soften_uncertainty,
    source_descriptor,
)
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


PROBE_ID = "scheduler_policy.ready_warps"
SOURCE = Path(__file__).with_name("ready_warps.cu")

# The timed loop must be dependent-FMA warps with no register spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_sched_ready_warps",
    required_opcodes={"FFMA": 8},
    forbidden_opcodes=("LDL", "STL"),
)


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def _saturation_warps(sweep: list[dict]) -> int | None:
    """Smallest warp count reaching >=95% of the peak ops/cycle."""

    if not sweep:
        return None
    peak = max(float(p["ops_per_cycle"]) for p in sweep)
    if peak <= 0:
        return None
    for point in sweep:
        if float(point["ops_per_cycle"]) >= 0.95 * peak:
            return int(point["warps"])
    return None


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    src_descriptor = source_descriptor(SOURCE)
    try:
        result = run_kernel(SOURCE, capabilities=capabilities, timeout=60, expectation=EXPECTATION)
    except CudaUnavailable as exc:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"scheduler ready-warps probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    sweep = list(payload["sweep"])
    sat_warps = _saturation_warps(sweep)
    peak_ipc = max((float(p["ops_per_cycle"]) for p in sweep), default=None)
    fit = FitStatus.CONDITIONALLY_IDENTIFIED if sat_warps else FitStatus.BEHAVIORAL_ONLY
    uncertainty = (
        UncertaintyCategory.CONDITIONAL_SCALAR
        if sat_warps
        else UncertaintyCategory.BEHAVIORAL_CLASS
    )

    # SASS gating: reject if the timed loop is not dependent-FMA bound.
    sass = result.sass_validation
    decision, fit, uncertainty, downgrade_reason = apply_sass_gating(
        sass, EXPECTATION, fit, uncertainty
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

    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    if sass is not None:
        values["sass"] = sass.to_dict()
    assumptions = [
        "one CTA on one SM runs N independent dependent-FMA warps (no memory)",
        "saturation warp count = smallest warp count reaching 95% of peak ops/cycle",
        "scheduler policy name is behavioral; only issue scaling is reported",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(
                probe_id=PROBE_ID,
                binary_hash=result.binary_sha256,
                disassembly_hash=sass.disassembly_hash if sass else None,
            ),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(grid=(1, 1, 1), block=(1024, 1, 1), mode="kernel"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "saturation_warps": sat_warps,
                    "peak_ops_per_cycle": peak_ipc,
                    "sweep_points": len(sweep),
                },
                units={"peak_ops_per_cycle": "ops/cycle"},
                source="amora.probes.nvidia.baseline.scheduler_policy.ready_warps",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="issue_saturation_warps",
                value=sat_warps,
                unit="warps",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="scheduler_issue_scaling",
                interpretation={
                    "nvidia_backend": "ready-warp count at which issue throughput saturates on one SM",
                    "peak_ops_per_cycle": peak_ipc,
                },
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="gpgpu_num_sched_per_core",
                value=sat_warps,
                unit="warps",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="issue-scaling saturation knee → simulator scheduler issue capacity (conditional)",
                assumptions=assumptions,
            ),
        )
    ]
