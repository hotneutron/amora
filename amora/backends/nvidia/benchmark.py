"""NVIDIA NCU basic-stat classification for AMORA benchmark cases."""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.metrics import MetricResolver
from amora.backends.nvidia.ncu_run import NcuUnavailable, run_kernel_profiled
from amora.benchmarking.classification import (
    ClassificationResult,
    INSTRUCTION_LOGICAL,
    NCU_BASIC_RECIPE,
)
from amora.benchmarking.schema import BenchmarkCase
from benchmark_generators.ppp_canonical.replay import contract_for_case


NCU_BASIC_LOGICALS = (
    INSTRUCTION_LOGICAL,
    "elapsed_cycles",
    "duration_ns",
)


def _ncu_provenance(capabilities: NvidiaCapabilities) -> dict[str, object]:
    tool = capabilities.tools.get("ncu")
    return {"ncu_tool": tool.to_dict() if tool is not None else None}


def classify_case_basic(
    case: BenchmarkCase,
    *,
    capabilities: NvidiaCapabilities,
    arch: str = "sm_90",
    build_root: Path,
    timeout: int = 600,
) -> ClassificationResult:
    """Collect basic NCU stats for one AMORA-owned benchmark replay."""

    resolver = MetricResolver(capabilities.ncu_metrics)
    resolved = {}
    for logical in NCU_BASIC_LOGICALS:
        resolution = resolver.resolve(logical)
        if resolution.available and resolution.selected_name:
            resolved[logical] = resolution.selected_name
    if INSTRUCTION_LOGICAL not in resolved:
        reason = "NCU instruction metric is unavailable"
        if capabilities.ncu_metrics_error:
            reason = f"{reason}: {capabilities.ncu_metrics_error}"
        return ClassificationResult(
            case_key=case.case_key,
            status="unclassified",
            total_instructions=None,
            reason=reason,
            resolved_metrics=resolved,
            provenance=_ncu_provenance(capabilities),
        )

    try:
        contract = contract_for_case(case)
    except (KeyError, ValueError) as exc:
        return ClassificationResult(
            case_key=case.case_key,
            status="missing_artifact",
            total_instructions=None,
            reason=str(exc),
            resolved_metrics=resolved,
            provenance=_ncu_provenance(capabilities),
        )
    if not contract.source.exists():
        return ClassificationResult(
            case_key=case.case_key,
            status="missing_artifact",
            total_instructions=None,
            reason=f"missing replay source: {contract.source}",
            resolved_metrics=resolved,
            provenance=_ncu_provenance(capabilities),
        )

    try:
        result = run_kernel_profiled(
            contract.source,
            capabilities=capabilities,
            metrics=tuple(resolved.values()),
            args=contract.args,
            launch_skip=contract.launch_skip,
            launch_count=contract.launch_count,
            timeout=timeout,
            arch=arch,
            build_root=build_root,
            link_flags=contract.link_flags,
        )
    except NcuUnavailable as exc:
        return ClassificationResult(
            case_key=case.case_key,
            status="failed",
            total_instructions=None,
            reason=str(exc),
            resolved_metrics=resolved,
            provenance=_ncu_provenance(capabilities),
        )

    metrics = {
        logical: result.metrics[metric]
        for logical, metric in resolved.items()
        if metric in result.metrics
    }
    instructions = metrics.get(INSTRUCTION_LOGICAL)
    kernel_name = None
    if result.raw_rows:
        kernel_name = result.raw_rows[-1].get("Kernel Name") or None
    if instructions is None:
        return ClassificationResult(
            case_key=case.case_key,
            status="unclassified",
            total_instructions=None,
            kernel_name=kernel_name,
            metrics=metrics,
            resolved_metrics=resolved,
            provenance={**result.provenance(), **_ncu_provenance(capabilities)},
            reason="NCU output omitted the resolved instruction metric",
        )
    return ClassificationResult(
        case_key=case.case_key,
        status="classified",
        total_instructions=float(instructions),
        kernel_name=kernel_name,
        metrics=metrics,
        resolved_metrics=resolved,
        provenance={**result.provenance(), **_ncu_provenance(capabilities)},
    )
