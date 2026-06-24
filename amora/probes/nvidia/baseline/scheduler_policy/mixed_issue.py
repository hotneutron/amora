"""Scheduler mixed-issue / pipeline-overlap probe (P1).

Compares a mixed FP32+INT independent stream against single-pipe FP32 and INT
baselines. The overlap ratio (mixed vs additive) classifies dual-issue-like
behavior. Per the P1 methodology this is a behavioral classification, not a
named scheduler policy.
"""

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


PROBE_ID = "scheduler_policy.mixed_issue"
SOURCE = Path(__file__).with_name("mixed_issue.cu")


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def _classify(fp32: float, i32: float, mixed: float) -> tuple[str, float]:
    """Return (overlap_class, overlap_ratio).

    overlap_ratio = mixed / max(fp32, int). ~1.0 means no overlap (single issue),
    approaching (fp32+int)/max means full dual-issue overlap.
    """

    base_max = max(fp32, i32)
    additive = fp32 + i32
    if base_max <= 0:
        return "indeterminate", 0.0
    ratio = mixed / base_max
    additive_ratio = additive / base_max  # theoretical full-overlap ceiling
    midpoint = 1.0 + (additive_ratio - 1.0) * 0.5
    if ratio >= midpoint:
        return "dual_issue_like", ratio
    if ratio <= 1.1:
        return "single_issue_like", ratio
    return "partial_overlap", ratio


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    src_descriptor = source_descriptor(SOURCE)
    try:
        result = run_kernel(SOURCE, capabilities=capabilities, timeout=60)
    except CudaUnavailable as exc:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"scheduler mixed-issue probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    fp32 = float(payload["fp32_ops_per_cycle"])
    i32 = float(payload["int_ops_per_cycle"])
    mixed = float(payload["mixed_ops_per_cycle"])
    overlap_class, ratio = _classify(fp32, i32, mixed)
    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    assumptions = [
        "independent FP32 (FMA) and INT (MAD) streams run alone and interleaved",
        "overlap_ratio = mixed / max(fp32, int); higher means more pipe overlap",
        "mixed-issue capability is a behavioral class, not a named policy",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(probe_id=PROBE_ID, binary_hash=result.binary_sha256),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(grid=(1, 1, 1), block=(256, 1, 1), mode="kernel"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "fp32_ops_per_cycle": fp32,
                    "int_ops_per_cycle": i32,
                    "mixed_ops_per_cycle": mixed,
                    "overlap_ratio": ratio,
                },
                units={
                    "fp32_ops_per_cycle": "ops/cycle",
                    "int_ops_per_cycle": "ops/cycle",
                    "mixed_ops_per_cycle": "ops/cycle",
                },
                source="amora.probes.nvidia.baseline.scheduler_policy.mixed_issue",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="mixed_issue_overlap",
                value=overlap_class,
                fit_status=FitStatus.BEHAVIORAL_ONLY,
                uncertainty=UncertaintyCategory.BEHAVIORAL_CLASS,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="mixed_pipeline_issue_overlap",
                interpretation={
                    "nvidia_backend": "FP32/INT pipe overlap classified from mixed vs single-pipe throughput",
                    "overlap_ratio": ratio,
                },
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="gpgpu_dual_issue_diff_exec_units",
                value=overlap_class,
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=FitStatus.BEHAVIORAL_ONLY,
                uncertainty=UncertaintyCategory.BEHAVIORAL_CLASS,
                mapping_contract="mixed/single-pipe overlap ratio → simulator dual-issue behavioral class",
                assumptions=assumptions,
            ),
        )
    ]
