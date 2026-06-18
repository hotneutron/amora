"""Analysis helpers for P0 shared-memory probes."""

from __future__ import annotations

from amora.core.statistics import detect_periodic_peaks, summarize_samples
from amora.schemas.results import EvidenceTier, ParameterEstimate, ProbeResult


def infer_shared_memory_bank_period(stride_cycles: list[float]) -> int | None:
    """Infer bank-period spacing from a stride-latency curve."""

    return detect_periodic_peaks(stride_cycles, min_ratio=1.2)


def shared_memory_analysis_probe(stride_cycles: list[float]) -> ProbeResult:
    """Analyze shared-memory stride timing results."""

    period = infer_shared_memory_bank_period(stride_cycles)
    summary = summarize_samples(stride_cycles)
    estimates: list[ParameterEstimate] = []
    warnings: list[str] = []

    if period is None:
        warnings.append("No stable bank-conflict periodicity was detected.")
    else:
        estimates.append(
            ParameterEstimate(
                name="shader_core_config::gpgpu_shmem_num_banks",
                value=period,
                evidence=EvidenceTier.TIMING_DIRECT,
                confidence=0.65,
                risk="medium",
                notes=("Period is inferred from stride-latency peaks.",),
            )
        )

    return ProbeResult(
        name="shared_memory/analyze.py",
        tier="P0",
        status="ok" if period is not None else "indeterminate",
        measurements={"stride_cycles": stride_cycles, "summary": summary},
        estimates=estimates,
        warnings=warnings,
    )
