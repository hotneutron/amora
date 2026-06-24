"""L2 cache-path dependent pointer-chase probe (P2).

Measures L2-hit load latency by walking a randomized pointer-chase ring that
exceeds the L1 data cache but fits inside the L2, with a DRAM-resident ring as a
control so the hit regime can be validated (l2 << dram cycles-per-load). L2 is
harder to isolate than L1, so the default fit is BOUNDED rather than DIRECT.
"""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.runner import CudaUnavailable, run_kernel
from amora.backends.nvidia.sass import SassExpectation
from amora.probes.nvidia.baseline._sources import apply_sass_gating, collect_ncu_metrics, source_descriptor
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


PROBE_ID = "l2_cache.pointer_chase"
SOURCE = Path(__file__).with_name("pointer_chase.cu")

# The timed loop must hit global memory (LDG) without shared or local spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_l2_pointer_chase",
    required_opcodes={"LDG": 1},
    forbidden_opcodes=("LDS", "STL"),
)


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    src_descriptor = source_descriptor(SOURCE)
    try:
        result = run_kernel(SOURCE, capabilities=capabilities, timeout=60, expectation=EXPECTATION)
    except CudaUnavailable as exc:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"L2 pointer-chase probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    l2_cpl = float(payload["l2_hit_cycles_per_load"])
    dram_cpl = float(payload["dram_cycles_per_load"])
    # The L2-fit ring is only an L2-hit regime if it is clearly faster than DRAM.
    hit_regime = dram_cpl > l2_cpl * 1.3
    fit = FitStatus.BOUNDED if hit_regime else FitStatus.UNDERCONSTRAINED

    # SASS gating: reject if the timed loop is not a global-load chase.
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

    ncu_record = collect_ncu_metrics(
        capabilities,
        SOURCE,
        ["l2_sector_hits"],
        kernel_name="amora_l2_pointer_chase",
        role="validation",
        aggregate="max",
    )

    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    if sass is not None:
        values["sass"] = sass.to_dict()
    if ncu_record is not None:
        values["ncu"] = ncu_record
    assumptions = [
        "single-thread dependent pointer chase over a randomized ring sized to exceed L1 but fit L2",
        "a DRAM-resident ring is timed as a control; L2-hit regime requires l2 << dram",
        "median cycles-per-load reported across N launches",
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
                    "l2_hit_cycles_per_load": l2_cpl,
                    "dram_cycles_per_load": dram_cpl,
                    "hit_to_dram_ratio": dram_cpl / l2_cpl if l2_cpl > 0 else None,
                },
                units={
                    "l2_hit_cycles_per_load": "cycles",
                    "dram_cycles_per_load": "cycles",
                },
                source="amora.probes.nvidia.baseline.l2_cache.pointer_chase",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="l2_hit_load_latency",
                value=l2_cpl,
                unit="cycles",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="l2_hit_latency",
                interpretation={
                    "nvidia_backend": "dependent-load latency for an L2-resident working set in cycles",
                    "dram_control_cycles_per_load": dram_cpl,
                    "l2_hit_regime_confirmed": hit_regime,
                },
                metric_resolver=ncu_record or {},
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason
                if downgrade_reason is not None
                else (None if hit_regime else "L2 ring not clearly faster than DRAM control"),
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="l2_latency",
                value=l2_cpl,
                unit="cycles",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="dependent L2-resident chase cycles-per-load -> simulator L2 hit latency (bounded)",
                assumptions=assumptions,
            ),
        )
    ]
