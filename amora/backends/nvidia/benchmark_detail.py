"""Detailed NCU evidence collection for classified benchmark cases."""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.benchmark import NCU_BASIC_LOGICALS, _ncu_provenance
from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.metrics import MetricResolver
from amora.backends.nvidia.ncu_run import NcuUnavailable, run_kernel_profiled
from amora.backends.nvidia.stall_metrics import (
    resolve_stall_metric_family,
    select_stall_launch_row,
)
from amora.benchmarking.schema import BenchmarkCase, DetailedCaseResult
from benchmark_generators.ppp_canonical.replay import contract_for_case


def collect_case_detail(
    case: BenchmarkCase,
    *,
    capabilities: NvidiaCapabilities,
    arch: str,
    build_root: Path,
    timeout: int,
    size_rank: str,
) -> DetailedCaseResult:
    """Collect detailed NCU counters and stall reasons for one replay contract."""

    resolver = MetricResolver(capabilities.ncu_metrics)
    resolved: dict[str, str] = {}
    for logical in NCU_BASIC_LOGICALS:
        resolution = resolver.resolve(logical)
        if resolution.available and resolution.selected_name:
            resolved[logical] = resolution.selected_name
    stall_selection = resolve_stall_metric_family(capabilities.ncu_metrics)
    resolved.update(stall_selection.logical_to_metric)
    if "inst_executed" not in resolved:
        return DetailedCaseResult(
            case_key=case.case_key,
            kernel_id=case.kernel_id,
            size_rank=size_rank,
            backend="nvidia_cuda",
            status="unavailable",
            resolved_metrics=resolved,
            provenance=_ncu_provenance(capabilities),
            reason="NCU instruction metric is unavailable",
        )

    try:
        contract = contract_for_case(case)
    except (KeyError, ValueError) as exc:
        return DetailedCaseResult(
            case_key=case.case_key,
            kernel_id=case.kernel_id,
            size_rank=size_rank,
            backend="nvidia_cuda",
            status="missing_artifact",
            resolved_metrics=resolved,
            provenance=_ncu_provenance(capabilities),
            reason=str(exc),
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
        return DetailedCaseResult(
            case_key=case.case_key,
            kernel_id=case.kernel_id,
            size_rank=size_rank,
            backend="nvidia_cuda",
            status="failed",
            resolved_metrics=resolved,
            provenance=_ncu_provenance(capabilities),
            reason=str(exc),
        )

    logical_metrics = {
        logical: result.metrics[metric]
        for logical, metric in resolved.items()
        if metric in result.metrics
    }
    selected_index, stalls, total_stall = select_stall_launch_row(
        result.raw_rows,
        stall_selection.reason_to_metric,
    )
    logical_metrics.update(
        {f"stall_{reason}": value for reason, value in stalls.items()}
    )
    kernel_name = result.raw_rows[-1].get("Kernel Name") if result.raw_rows else None
    return DetailedCaseResult(
        case_key=case.case_key,
        kernel_id=case.kernel_id,
        size_rank=size_rank,
        backend="nvidia_cuda",
        status="measured",
        measurement={
            "total_instructions": logical_metrics.get("inst_executed"),
            "elapsed_cycles": logical_metrics.get("elapsed_cycles"),
            "duration_ns": logical_metrics.get("duration_ns"),
            "kernel_name": kernel_name,
            "semantic": case.execution_contract.get("measurement_semantics"),
        },
        logical_metrics=logical_metrics,
        resolved_metrics=resolved,
        raw_metrics=result.metrics,
        stall_histogram={
            "schema": "ncu-stall-v1",
            "metric_family": stall_selection.family,
            "unit": stall_selection.unit,
            "reasons": stalls,
            "complete": not stall_selection.missing_reasons,
            "missing_reasons": list(stall_selection.missing_reasons),
            "launch_selection": {
                "strategy": "median_total_stall",
                "selected_index": selected_index,
                "launches_profiled": len(result.raw_rows),
                "total_stall": total_stall,
            },
        },
        provenance={**result.provenance(), **_ncu_provenance(capabilities)},
    )
