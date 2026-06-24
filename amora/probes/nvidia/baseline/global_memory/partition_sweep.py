"""DRAM partition-camping sweep probe (P2, Phase B).

Streams a large DRAM buffer from several base byte-offsets and times each with
CUDA events. The variation across offsets indicates whether the device is
sensitive to memory-partition (channel) camping. The measurement is a behavioral
class: "balanced" when the max/min bandwidth ratio is small, otherwise
"camping_sensitive". NCU DRAM read bytes are collected best-effort as
corroboration. SASS validation confirms the kernel is a global-load stream.
"""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.metrics import MetricResolver
from amora.backends.nvidia.ncu_run import NcuUnavailable, run_kernel_profiled
from amora.backends.nvidia.runner import CudaUnavailable, run_kernel
from amora.backends.nvidia.sass import SassExpectation
from amora.probes.nvidia.baseline._sources import apply_sass_gating, source_descriptor
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


PROBE_ID = "global_memory.partition_sweep"
SOURCE = Path(__file__).with_name("partition_sweep.cu")

_CAMPING_RATIO = 1.15

# The timed loop must hit global memory (LDG) without shared or local spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_gmem_partition",
    required_opcodes={"LDG": 1},
    forbidden_opcodes=("LDS", "STL"),
)


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def _collect_counter(capabilities: NvidiaCapabilities, logical: str) -> dict | None:
    """Collect a DRAM byte counter via NCU (corroboration role)."""

    resolver = MetricResolver(supported_metrics=capabilities.ncu_metrics)
    resolution = resolver.resolve(logical)
    if not resolution.available or not resolution.selected_name:
        return None
    try:
        ncu = run_kernel_profiled(
            SOURCE,
            capabilities=capabilities,
            metrics=(resolution.selected_name,),
            kernel_name="amora_gmem_partition",
            launch_count=16,  # cover warm-up + all offset launches
        )
    except NcuUnavailable:
        return None
    max_value = None
    if resolution.selected_name in ncu.metrics:
        max_value = float(ncu.metrics[resolution.selected_name])
    for row in ncu.raw_rows:
        raw = (row.get(resolution.selected_name) or "").strip().replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        max_value = v if max_value is None else max(max_value, v)
    return {
        "metric": resolution.selected_name,
        "logical": logical,
        "role": "corroboration",
        "value": max_value,
        "launches_profiled": len(ncu.raw_rows),
    }


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    src_descriptor = source_descriptor(SOURCE)
    try:
        result = run_kernel(SOURCE, capabilities=capabilities, timeout=120, expectation=EXPECTATION)
    except CudaUnavailable as exc:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"partition-sweep probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    sweep = list(payload["sweep"])
    gbps = [float(p["gbps"]) for p in sweep if float(p["gbps"]) > 0]
    max_gbps = max(gbps) if gbps else 0.0
    min_gbps = min(gbps) if gbps else 0.0
    ratio = (max_gbps / min_gbps) if min_gbps > 0 else None
    camping_class = (
        "balanced" if (ratio is not None and ratio < _CAMPING_RATIO) else "camping_sensitive"
    )

    sass = result.sass_validation
    decision, fit, uncertainty, downgrade_reason = apply_sass_gating(
        sass, EXPECTATION, FitStatus.BEHAVIORAL_ONLY, UncertaintyCategory.BEHAVIORAL_CLASS
    )
    if decision == "reject":
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"SASS validation rejected the measurement: {sass.reason}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor, "sass": sass.to_dict()},
            )
        ]

    # NCU DRAM read bytes (best effort) corroborate that the sweep is DRAM-bound.
    ncu_record = _collect_counter(capabilities, "dram_bytes_read")

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
        "grid-stride read from several base offsets relative to the partition interleave",
        "best-of-N CUDA-event timing per offset; bandwidth varies with clock/partition balance",
        f"max/min bandwidth ratio < {_CAMPING_RATIO} classifies as balanced else camping_sensitive",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(
                probe_id=PROBE_ID,
                binary_hash=result.binary_sha256,
                disassembly_hash=sass.disassembly_hash if sass else None,
            ),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(mode="kernel"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "max_gbps": max_gbps,
                    "min_gbps": min_gbps,
                    "bandwidth_ratio": ratio,
                    "partition_camping_class": camping_class,
                },
                units={"max_gbps": "GB/s", "min_gbps": "GB/s"},
                source="amora.probes.nvidia.baseline.global_memory.partition_sweep",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="partition_camping_class",
                value=camping_class,
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="memory_partition_behavior",
                interpretation={
                    "nvidia_backend": "DRAM partition-camping sensitivity from base-offset bandwidth sweep",
                    "bandwidth_ratio": ratio,
                    "partition_camping_class": camping_class,
                },
                metric_resolver=ncu_record or {},
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="memory_partition_class",
                value=camping_class,
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="base-offset bandwidth variation -> simulator memory-partition camping class",
                assumptions=assumptions,
            ),
        )
    ]
