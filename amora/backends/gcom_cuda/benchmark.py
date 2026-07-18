"""GCoM trace/simulation evidence collection for classified benchmark cases."""

from __future__ import annotations

from pathlib import Path

from amora.backends.gcom_cuda import config as cfg
from amora.backends.gcom_cuda import runner, trace
from amora.backends.gcom_cuda.gcom import GcomCapabilities
from amora.benchmarking.schema import BenchmarkCase, DetailedCaseResult
from benchmark_generators.ppp_canonical.replay import contract_for_case


def simulate_case_detail(
    case: BenchmarkCase,
    *,
    capabilities: GcomCapabilities,
    sku: str,
    out_dir: Path,
    trace_timeout: int,
    sim_timeout: int,
    omp_threads: int | None,
    size_rank: str,
) -> DetailedCaseResult:
    """Trace the measured benchmark launch and collect GCoM detailed evidence."""

    if not capabilities.simulator_built or not capabilities.tracer_built:
        return DetailedCaseResult(
            case_key=case.case_key,
            kernel_id=case.kernel_id,
            size_rank=size_rank,
            backend="gcom_cuda",
            status="unavailable",
            reason="; ".join(capabilities.unsupported_reasons) or "GCoM is unavailable",
            provenance={"capabilities": capabilities.to_dict()},
        )
    try:
        contract = contract_for_case(case)
    except (KeyError, ValueError) as exc:
        return DetailedCaseResult(
            case_key=case.case_key,
            kernel_id=case.kernel_id,
            size_rank=size_rank,
            backend="gcom_cuda",
            status="missing_artifact",
            reason=str(exc),
        )
    try:
        trace_dir = trace.trace_probe(
            case.case_key,
            contract.source,
            out_dir,
            defines=contract.trace_defines,
            argv=contract.args,
            link_flags=contract.link_flags,
            timeout=trace_timeout,
        )
        profile = cfg.get_sku_profile(sku)
        sim = runner.simulate(
            profile,
            trace_dir,
            log_path=out_dir / "gcom_sim.log",
            timeout=sim_timeout,
            omp_threads=omp_threads,
        )
    except (trace.TraceError, runner.SimulateError) as exc:
        return DetailedCaseResult(
            case_key=case.case_key,
            kernel_id=case.kernel_id,
            size_rank=size_rank,
            backend="gcom_cuda",
            status="failed",
            reason=str(exc),
            provenance={"replay_source": str(contract.source), "args": list(contract.args)},
        )
    if not sim.core_present():
        return DetailedCaseResult(
            case_key=case.case_key,
            kernel_id=case.kernel_id,
            size_rank=size_rank,
            backend="gcom_cuda",
            status="missing_stat",
            reason=f"simulator emitted no {runner.REQUIRED_CORE_STAT}",
            provenance={"trace_dir": str(trace_dir), "sim_log": str(out_dir / "gcom_sim.log")},
        )
    stall_hist = runner.extract_stall_reason_histogram(sim.stats)
    return DetailedCaseResult(
        case_key=case.case_key,
        kernel_id=case.kernel_id,
        size_rank=size_rank,
        backend="gcom_cuda",
        status="simulated",
        measurement={
            "gpu_sim_cycle": sim.stats.get("gpu_sim_cycle"),
            "gpu_tot_sim_insn": sim.stats.get("gpu_tot_sim_insn"),
            "semantic": case.execution_contract.get("measurement_semantics"),
        },
        logical_metrics=runner.derive_logical_metrics(sim.stats),
        raw_metrics=sim.stats,
        stall_histogram=stall_hist,
        provenance={
            "trace_dir": str(trace_dir),
            "sim_log": str(out_dir / "gcom_sim.log"),
            "replay_source": str(contract.source),
            "args": list(contract.args),
            "trace_defines": list(contract.trace_defines),
            "trace_selection": "single measured launch; warmup compiled out",
        },
    )
