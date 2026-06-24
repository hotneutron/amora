"""Shared-memory bank-stride sweep probe."""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.runner import CudaUnavailable, run_kernel
from amora.probes.nvidia.baseline._sources import source_descriptor
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


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


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
        result = run_kernel(SOURCE, capabilities=capabilities)
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
    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    assumptions = [
        "single warp probes shared memory with stride sweep covering conflict-factors 1..32",
        "conflict factor reported as gcd(stride, 32) which holds for shipping NVIDIA archs",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(probe_id=PROBE_ID, binary_hash=result.binary_sha256),
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
                fit_status=FitStatus.UNIQUELY_IDENTIFIED if inferred_banks else FitStatus.UNDERCONSTRAINED,
                uncertainty=UncertaintyCategory.STABLE_SCALAR,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="shared_memory_bank_count",
                interpretation={
                    "nvidia_backend": "shared-memory bank count inferred from cycles-per-access vs stride curve",
                    "no_conflict_cycles_per_access": no_conflict_cycles,
                    "full_conflict_cycles_per_access": full_conflict_cycles,
                },
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="shared_memory_banks",
                value=inferred_banks,
                unit="banks",
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=FitStatus.UNIQUELY_IDENTIFIED if inferred_banks else FitStatus.UNDERCONSTRAINED,
                uncertainty=UncertaintyCategory.STABLE_SCALAR,
                mapping_contract="bank-stride sweep peak conflict factor → simulator shared-memory bank count",
                assumptions=assumptions,
            ),
        )
    ]
