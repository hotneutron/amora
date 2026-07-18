"""Detailed NCU evidence collection for classified benchmark cases."""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.benchmark import NCU_BASIC_LOGICALS, _ncu_provenance
from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.metrics import MetricResolver
from amora.backends.nvidia.ncu_run import NcuUnavailable, run_kernel_profiled
from amora.benchmarking.schema import BenchmarkCase, DetailedCaseResult
from benchmark_generators.ppp_canonical.replay import contract_for_case


_STALL_LOGICALS = (
    "stall_selected",
    "stall_not_selected",
    "stall_dispatch_stall",
    "stall_warpgroup_arrive",
    "stall_long_scoreboard",
    "stall_short_scoreboard",
    "stall_barrier",
    "stall_wait",
    "stall_mio_throttle",
    "stall_math_pipe_throttle",
    "stall_mma",
    "stall_no_instructions",
    "stall_imc_miss",
    "stall_sleeping",
    "stall_branch_resolving",
    "stall_membar",
    "stall_drain",
    "stall_lg_throttle",
    "stall_tex_throttle",
    "stall_misc",
)


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
    for logical in (*NCU_BASIC_LOGICALS, *_STALL_LOGICALS):
        resolution = resolver.resolve(logical)
        if resolution.available and resolution.selected_name:
            resolved[logical] = resolution.selected_name
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
    stalls = {
        logical.removeprefix("stall_"): value
        for logical, value in logical_metrics.items()
        if logical.startswith("stall_")
    }
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
            "reasons": stalls,
            "complete": len(stalls) == len(_STALL_LOGICALS),
            "missing_reasons": [
                logical.removeprefix("stall_")
                for logical in _STALL_LOGICALS
                if logical not in logical_metrics
            ],
        },
        provenance={**result.provenance(), **_ncu_provenance(capabilities)},
    )
