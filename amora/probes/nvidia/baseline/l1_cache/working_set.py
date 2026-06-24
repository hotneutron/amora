"""L1 working-set sweep probe (P1).

Sweeps the pointer-chase working-set size and reports the latency-vs-size curve.
Capacity knees in the curve mark L1 -> L2 -> DRAM transitions. Capacity is
reported as a bounded effective range (the first knee), per the P1 methodology.
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


PROBE_ID = "l1_cache.working_set"
SOURCE = Path(__file__).with_name("working_set.cu")

# The timed loop must hit global memory (LDG) without shared or local spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_l1_working_set",
    required_opcodes={"LDG": 1},
    forbidden_opcodes=("LDS", "STL"),
)


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def _first_knee(sweep: list[dict]) -> tuple[int, int] | None:
    """Return (last_flat_kb, first_jump_kb) at the first >40% latency jump."""

    for i in range(1, len(sweep)):
        prev = float(sweep[i - 1]["cycles_per_load"])
        cur = float(sweep[i]["cycles_per_load"])
        if prev > 0 and cur > prev * 1.4:
            return int(sweep[i - 1]["working_set_kb"]), int(sweep[i]["working_set_kb"])
    return None


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    src_descriptor = source_descriptor(SOURCE)
    try:
        result = run_kernel(SOURCE, capabilities=capabilities, timeout=60, expectation=EXPECTATION)
    except CudaUnavailable as exc:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"L1 working-set sweep could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    sweep = list(payload["sweep"])
    knee = _first_knee(sweep)
    capacity_low = knee[0] if knee else None
    capacity_high = knee[1] if knee else None
    capacity_value = {"effective_l1_kb_low": capacity_low, "effective_l1_kb_high": capacity_high}
    fit = FitStatus.BOUNDED if knee else FitStatus.UNDERCONSTRAINED

    # SASS gating: reject if the timed loop is not a global-load sweep.
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

    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    if sass is not None:
        values["sass"] = sass.to_dict()
    assumptions = [
        "dependent pointer-chase latency swept across geometric working-set sizes",
        "first >40% latency jump marks the effective L1 capacity knee",
        "capacity is reported as a bounded range, not an exact scalar",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(
                probe_id=PROBE_ID,
                binary_hash=result.binary_sha256,
                disassembly_hash=sass.disassembly_hash if sass else None,
            ),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(grid=(1, 1, 1), block=(32, 1, 1), mode="kernel"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "sweep_points": len(sweep),
                    "effective_l1_kb_low": capacity_low,
                    "effective_l1_kb_high": capacity_high,
                },
                units={"effective_l1_kb_low": "KiB", "effective_l1_kb_high": "KiB"},
                source="amora.probes.nvidia.baseline.l1_cache.working_set",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="l1_effective_capacity",
                value=capacity_value,
                unit="KiB",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="l1_effective_capacity_knee",
                interpretation={
                    "nvidia_backend": "effective L1 capacity bounded by the first latency knee in the working-set sweep",
                },
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason
                if downgrade_reason is not None
                else (None if knee else "no clear capacity knee detected in the sweep"),
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="l1d_cache_capacity",
                value=capacity_value,
                unit="KiB",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="working-set latency knee → simulator L1 capacity range",
                assumptions=assumptions,
            ),
        )
    ]
