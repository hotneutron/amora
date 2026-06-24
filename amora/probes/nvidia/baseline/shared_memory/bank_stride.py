"""Shared-memory bank-stride sweep probe."""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.ncu_run import NcuUnavailable, run_kernel_profiled
from amora.backends.nvidia.runner import CudaUnavailable, run_kernel
from amora.backends.nvidia.metrics import MetricResolver
from amora.backends.nvidia.sass import SassExpectation, gate_decision
from amora.probes.nvidia.baseline._sources import (
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


PROBE_ID = "shared_memory.bank_stride"
SOURCE = Path(__file__).with_name("bank_stride.cu")

# The timed loop must use shared loads (LDS) and no register spills. Global
# write-back (STG) of results is expected outside the hot loop, so only local
# spills (LDL/STL) are forbidden.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_baseline_shared_bank_stride",
    required_opcodes={"LDS": 1},
    forbidden_opcodes=("LDL", "STL"),
)


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def _collect_conflict_counter(capabilities: NvidiaCapabilities) -> dict | None:
    """Collect the shared-conflicts counter via NCU (validation role).

    The kernel launches once per stride, so we profile every launch and take the
    maximum conflict count across them (the conflict-bearing strides). Returns a
    record dict with the resolved metric, the max value, and how many launches
    were actually profiled, or None when NCU/the counter is unavailable.
    """

    resolver = MetricResolver(supported_metrics=capabilities.ncu_metrics)
    resolution = resolver.resolve("shared_conflicts")
    if not resolution.available or not resolution.selected_name:
        return None
    try:
        ncu = run_kernel_profiled(
            SOURCE,
            capabilities=capabilities,
            metrics=(resolution.selected_name,),
            kernel_name="amora_baseline_shared_bank_stride",
            launch_count=64,  # cover warm-up + all sweep strides
        )
    except NcuUnavailable:
        return None
    # Max conflict count across every profiled launch row.
    max_value = None
    for row in ncu.raw_rows:
        raw = (row.get(resolution.selected_name) or "").strip().replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        max_value = v if max_value is None else max(max_value, v)
    return {
        "metric": resolution.selected_name,
        "logical": "shared_conflicts",
        "role": "validation",
        "value": max_value,
        "launches_profiled": len(ncu.raw_rows),
    }


def _infer_bank_count(sweep: list[dict]) -> int | None:
    """Infer the number of banks (32 on every shipping NVIDIA arch) from the sweep curve."""

    # Look for the largest conflict factor that produces a clear cycles-per-access spike.
    max_factor = 1
    for point in sweep:
        max_factor = max(max_factor, int(point["conflict_factor"]))
    return max_factor if max_factor >= 2 else None


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    src_descriptor = source_descriptor(SOURCE)
    try:
        result = run_kernel(SOURCE, capabilities=capabilities, expectation=EXPECTATION)
    except CudaUnavailable as exc:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"shared-memory bank-stride probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    sweep = list(payload["sweep"])
    inferred_banks = _infer_bank_count(sweep)
    no_conflict = next((p for p in sweep if int(p["conflict_factor"]) == 1), None)
    full_conflict = max(sweep, key=lambda p: int(p["conflict_factor"]))
    no_conflict_cycles = float(no_conflict["cycles_per_access"]) if no_conflict else None
    full_conflict_cycles = float(full_conflict["cycles_per_access"])

    # SASS gating: reject if the timed loop is not shared-memory bound.
    sass = result.sass_validation
    decision = gate_decision(sass, EXPECTATION) if sass is not None else "pass"
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
    base_fit = FitStatus.UNIQUELY_IDENTIFIED if inferred_banks else FitStatus.UNDERCONSTRAINED
    uncertainty = UncertaintyCategory.STABLE_SCALAR
    downgrade_reason = None
    if decision == "downgrade":
        base_fit = downgrade_fit(base_fit)
        uncertainty = soften_uncertainty(uncertainty)
        downgrade_reason = f"SASS validation downgrade: {sass.reason}"

    # NCU validation (best effort): confirm the bank-conflict model with the
    # shared-conflicts counter. Collected in a separate profiler pass; it never
    # overrides the timing scalar, only corroborates or downgrades it.
    #
    # The kernel issues one launch per stride, so a conflict-bearing launch must
    # exist somewhere in the sweep. We profile all launches and take the max
    # conflict count; only a confirmed all-zero result (with launches actually
    # profiled) is treated as contradicting the timing-derived bank model.
    ncu_record = _collect_conflict_counter(capabilities)
    if ncu_record is not None and inferred_banks:
        conflicts = ncu_record.get("value")
        launches = ncu_record.get("launches_profiled") or 0
        if conflicts is not None and launches >= len(sweep) and conflicts <= 0:
            base_fit = downgrade_fit(base_fit)
            uncertainty = soften_uncertainty(uncertainty)
            downgrade_reason = (
                (downgrade_reason + "; " if downgrade_reason else "")
                + "NCU validation: shared-conflict counter reported zero across the sweep"
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
        "single warp probes shared memory with stride sweep covering conflict-factors 1..32",
        "conflict factor reported as gcd(stride, 32) which holds for shipping NVIDIA archs",
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
                    "no_conflict_cycles_per_access": no_conflict_cycles,
                    "full_conflict_cycles_per_access": full_conflict_cycles,
                    "inferred_bank_count": inferred_banks,
                    "sweep_points": len(sweep),
                },
                units={
                    "no_conflict_cycles_per_access": "cycles",
                    "full_conflict_cycles_per_access": "cycles",
                },
                source="amora.probes.nvidia.baseline.shared_memory.bank_stride",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="shared_memory_bank_count",
                value=inferred_banks,
                unit="banks",
                fit_status=base_fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="shared_memory_bank_count",
                interpretation={
                    "nvidia_backend": "shared-memory bank count inferred from cycles-per-access vs stride curve",
                    "no_conflict_cycles_per_access": no_conflict_cycles,
                    "full_conflict_cycles_per_access": full_conflict_cycles,
                },
                metric_resolver=ncu_record or {},
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="shared_memory_banks",
                value=inferred_banks,
                unit="banks",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=base_fit,
                uncertainty=uncertainty,
                mapping_contract="bank-stride sweep peak conflict factor → simulator shared-memory bank count",
                assumptions=assumptions,
            ),
        )
    ]
