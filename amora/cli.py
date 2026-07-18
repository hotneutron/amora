"""Command-line interface for AMORA."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from amora.backends.nvidia.cuda import discover_capabilities as nvidia_discover
from amora.probes.nvidia import baseline as nvidia_baseline
from amora.reports.json_report import render_report, write_report
from amora.reports.markdown_report import write_reports_from_json


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


# --- Shared handlers (injected with a backend's discover + baseline module). ---


def _cmd_list(baseline) -> int:
    _print_json({"probes": baseline.list_probes()})
    return 0


def _cmd_capabilities(discover) -> int:
    _print_json(discover().to_dict())
    return 0


def _cmd_run_nvidia(args: argparse.Namespace) -> int:
    capabilities = nvidia_discover()
    if args.all:
        results = nvidia_baseline.run_all(capabilities)
    else:
        results = nvidia_baseline.run_probe(args.probe, capabilities)
    metadata = {"backend_capabilities": capabilities.to_dict()}
    if args.output:
        write_report(Path(args.output), results, metadata=metadata)
    else:
        _print_json(render_report(results, metadata=metadata))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    written = write_reports_from_json(
        Path(args.input),
        Path(args.out_dir),
        vendor=args.vendor,
        family=args.family,
        sku=args.sku,
    )
    print(written)
    return 0


# --- benchmark handlers ---


def _cmd_list_benchmarks(_args: argparse.Namespace) -> int:
    from amora.benchmarking.registry import list_benchmarks

    _print_json({"benchmarks": list_benchmarks()})
    return 0


def _cmd_inspect_benchmark(args: argparse.Namespace) -> int:
    from amora.benchmarking.registry import get_benchmark

    _print_json(get_benchmark(args.benchmark_id).describe())
    return 0


def _cmd_materialize_benchmark(args: argparse.Namespace) -> int:
    from amora.benchmarking.materialize import materialize_benchmark, write_manifest
    from amora.benchmarking.registry import get_benchmark

    definition = get_benchmark(args.benchmark_id)
    case_count = args.cases
    seed = args.seed
    if args.preset:
        presets = getattr(definition, "presets", {})
        try:
            preset = presets[args.preset]
        except KeyError as exc:
            known = ", ".join(sorted(presets)) or "none"
            raise ValueError(
                f"unknown preset {args.preset!r} for {args.benchmark_id}; known: {known}"
            ) from exc
        if case_count is None:
            case_count = preset["case_count"]
        if seed is None:
            seed = preset["seed"]
    if case_count is None:
        raise ValueError("pass --cases or --preset")
    if seed is None:
        seed = 0
    target = {
        "vendor": args.vendor,
        "family": args.family,
        "hardware_sku": args.sku,
        "arch_profile": args.arch_profile,
    }
    manifest = materialize_benchmark(
        args.benchmark_id,
        target=target,
        case_count=case_count,
        seed=seed,
    )
    default_out = (
        Path("out")
        / "benchmarks"
        / manifest.benchmark_id
        / f"r{manifest.benchmark_revision}"
        / manifest.case_set_digest
        / "manifest.json"
    )
    destination = write_manifest(manifest, args.output or default_out)
    _print_json(
        {
            "manifest": str(destination),
            "benchmark_id": manifest.benchmark_id,
            "benchmark_revision": manifest.benchmark_revision,
            "case_count_materialized": manifest.materialized_case_count,
            "case_set_digest": manifest.case_set_digest,
        }
    )
    return 0


def _cmd_classify_benchmark(args: argparse.Namespace) -> int:
    from amora.backends.nvidia.benchmark import classify_case_basic
    from amora.backends.nvidia.cuda import discover_capabilities
    from amora.benchmarking.classification import classify_cases, write_classification_manifest
    from amora.benchmarking.materialize import load_manifest

    manifest = load_manifest(args.manifest)
    if args.recipe != "ncu_basic_v1":
        raise ValueError(f"unsupported classification recipe: {args.recipe}")
    cases = list(manifest.cases)
    if args.limit is not None:
        cases = cases[:args.limit]
    capabilities = discover_capabilities()
    build_root = (
        Path("out")
        / "benchmarks"
        / manifest.benchmark_id
        / f"r{manifest.benchmark_revision}"
        / manifest.case_set_digest
        / "build-cache"
    )
    classification = classify_cases(
        cases,
        case_set_digest=manifest.case_set_digest,
        target=manifest.target,
        classify_case=lambda case: classify_case_basic(
            case,
            capabilities=capabilities,
            arch=args.arch,
            build_root=build_root,
            timeout=args.timeout,
        ),
        expected_case_keys=[case.case_key for case in manifest.cases],
        recipe=args.recipe,
    )
    default_out = (
        Path("out")
        / "benchmarks"
        / manifest.benchmark_id
        / f"r{manifest.benchmark_revision}"
        / manifest.case_set_digest
        / "classification"
        / "nvidia_cuda"
        / manifest.target.get("family", "generic")
        / manifest.target.get("hardware_sku", "generic")
        / classification.classification_digest
        / "classification.json"
    )
    destination = write_classification_manifest(classification, args.output or default_out)
    counts = {}
    for result in classification.results:
        counts[result.status] = counts.get(result.status, 0) + 1
    ranks = {}
    for assignment in classification.rank_assignments.values():
        rank = assignment["size_rank"]
        ranks[rank] = ranks.get(rank, 0) + 1
    _print_json(
        {
            "classification": str(destination),
            "case_set_digest": classification.case_set_digest,
            "classification_digest": classification.classification_digest,
            "result_counts": counts,
            "rank_counts": ranks,
            "rank_boundaries": classification.rank_boundaries,
        }
    )
    return 0


def _cmd_detail_benchmark(args: argparse.Namespace) -> int:
    from amora.backends.gcom_cuda.benchmark import simulate_case_detail
    from amora.backends.gcom_cuda.gcom import discover_capabilities as gcom_discover
    from amora.backends.nvidia.benchmark_detail import collect_case_detail
    from amora.backends.nvidia.cuda import discover_capabilities as nvidia_discover
    from amora.benchmarking.detailed import (
        build_detailed_comparison,
        build_detail_run_manifest,
        load_classification_manifest,
        load_review_marker,
        review_gate_for_rank,
        select_ranked_cases,
        write_case_results,
        write_detail_run_manifest,
        write_detailed_comparison,
    )
    from amora.benchmarking.materialize import load_manifest

    manifest = load_manifest(args.manifest)
    classification = load_classification_manifest(args.classification)
    if args.benchmark_id != manifest.benchmark_id:
        raise ValueError(
            f"benchmark ID {args.benchmark_id!r} does not match manifest "
            f"{manifest.benchmark_id!r}"
        )
    review_marker = load_review_marker(args.review_marker) if args.review_marker else None
    review_gate = review_gate_for_rank(
        manifest,
        classification,
        size_rank=args.size_rank,
        review_marker=review_marker,
    )
    selected = select_ranked_cases(
        manifest,
        classification,
        size_rank=args.size_rank,
    )
    if args.limit is not None:
        selected = selected[:args.limit]
    rank_case_count = sum(
        assignment.get("size_rank") == args.size_rank
        for assignment in classification.rank_assignments.values()
    )
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run ID must contain only letters, digits, '.', '_', or '-'")
    case_root = (
        Path("out")
        / "benchmarks"
        / manifest.benchmark_id
        / f"r{manifest.benchmark_revision}"
        / manifest.case_set_digest
    )
    run_dir = (
        case_root
        / "runs"
        / args.size_rank
        / classification.classification_digest
        / run_id
    )
    if run_dir.exists():
        raise FileExistsError(f"detail run directory already exists: {run_dir}")
    started_at = datetime.now(timezone.utc).isoformat()
    nvidia_caps = nvidia_discover()
    gcom_caps = gcom_discover(args.sku)
    hardware_results = [
        collect_case_detail(
            case,
            capabilities=nvidia_caps,
            arch=args.arch,
            build_root=case_root / "build-cache",
            timeout=args.ncu_timeout,
            size_rank=args.size_rank,
        )
        for case in selected
    ]
    simulation_results = [
        simulate_case_detail(
            case,
            capabilities=gcom_caps,
            sku=args.sku,
            out_dir=run_dir / "gcom" / case.case_key,
            trace_timeout=args.trace_timeout,
            sim_timeout=args.sim_timeout,
            omp_threads=args.omp_threads,
            size_rank=args.size_rank,
        )
        for case in selected
    ]
    hardware_path = write_case_results(hardware_results, run_dir / "hardware.jsonl")
    simulation_path = write_case_results(simulation_results, run_dir / "gcom.jsonl")
    comparison = build_detailed_comparison(
        manifest=manifest,
        classification=classification,
        size_rank=args.size_rank,
        hardware_results=hardware_results,
        simulation_results=simulation_results,
        rank_case_count=rank_case_count,
        selected_cases=selected,
        run_id=run_id,
        review_gate=review_gate,
    )
    raw_written = write_detailed_comparison(comparison, run_dir)
    completed_at = datetime.now(timezone.utc).isoformat()
    run_manifest = build_detail_run_manifest(
        manifest=manifest,
        classification=classification,
        comparison=comparison,
        hardware_results=hardware_results,
        simulation_results=simulation_results,
        hardware_path=hardware_path,
        simulation_path=simulation_path,
        comparison_path=raw_written["json"],
        review_gate=review_gate,
        started_at=started_at,
        completed_at=completed_at,
        run_options={
            "arch": args.arch,
            "gcom_sku": args.sku,
            "limit": args.limit,
            "ncu_timeout": args.ncu_timeout,
            "trace_timeout": args.trace_timeout,
            "sim_timeout": args.sim_timeout,
            "omp_threads": args.omp_threads,
        },
    )
    run_manifest_path = write_detail_run_manifest(run_manifest, run_dir / "run.json")
    default_report_out = (
        Path("reports")
        / "benchmarks"
        / manifest.benchmark_id
        / f"r{manifest.benchmark_revision}"
        / manifest.case_set_digest
        / "comparisons"
        / f"nvidia_cuda-{manifest.target.get('hardware_sku', 'generic')}__gcom_cuda-{args.sku}"
        / args.size_rank
        / comparison["comparison_digest"]
    )
    written = write_detailed_comparison(comparison, args.output or default_report_out)
    _print_json(
        {
            "run_id": run_id,
            "run_manifest": str(run_manifest_path),
            "hardware_results": str(hardware_path),
            "gcom_results": str(simulation_path),
            "raw_comparison_json": str(raw_written["json"]),
            "raw_comparison_markdown": str(raw_written["markdown"]),
            "comparison_json": str(written["json"]),
            "comparison_markdown": str(written["markdown"]),
            "comparison_digest": comparison["comparison_digest"],
            "coverage": comparison["coverage"],
        }
    )
    return 0


def _cmd_review_benchmark(args: argparse.Namespace) -> int:
    from amora.benchmarking.detailed import (
        build_review_marker,
        load_classification_manifest,
        load_detailed_comparison,
        write_review_marker,
    )
    from amora.benchmarking.materialize import load_manifest

    manifest = load_manifest(args.manifest)
    classification = load_classification_manifest(args.classification)
    if args.benchmark_id != manifest.benchmark_id:
        raise ValueError(
            f"benchmark ID {args.benchmark_id!r} does not match manifest "
            f"{manifest.benchmark_id!r}"
        )
    reviewed_at = args.reviewed_at or datetime.now(timezone.utc).isoformat()
    marker = build_review_marker(
        manifest=manifest,
        classification=classification,
        comparisons=[load_detailed_comparison(path) for path in args.comparison],
        known_failures=args.known_failure,
        semantic_decisions=args.semantic_decision,
        reviewer=args.reviewer,
        reviewed_at=reviewed_at,
    )
    default_out = (
        Path("out")
        / "benchmarks"
        / manifest.benchmark_id
        / f"r{manifest.benchmark_revision}"
        / manifest.case_set_digest
        / "reviews"
        / marker.reviewed_rank
        / classification.classification_digest
        / marker.review_marker_digest
        / "review.json"
    )
    destination = write_review_marker(marker, args.output or default_out)
    _print_json(
        {
            "review_marker": str(destination),
            "review_marker_digest": marker.review_marker_digest,
            "reviewed_rank": marker.reviewed_rank,
            "accepted_run_ids": list(marker.accepted_run_ids),
            "accepted_comparison_digests": list(marker.accepted_comparison_digests),
        }
    )
    return 0


# --- gcom_cuda handlers ---


def _cmd_run_gcom(args: argparse.Namespace) -> int:
    from amora.backends.gcom_cuda.gcom import discover_capabilities as gcom_discover
    from amora.backends.gcom_cuda.version import collect_version_metadata
    from amora.backends.gcom_cuda import config as gcfg
    from amora.probes.gcom_cuda import baseline as gcom_baseline

    capabilities = gcom_discover(args.sku)
    profile = gcfg.get_sku_profile(args.sku)
    hw_baseline = None
    if args.hw_baseline:
        from amora.backends.gcom_cuda.compare import load_backend_report

        hw_baseline = load_backend_report(args.hw_baseline)
    ctx = gcom_baseline.RunContext(sku=args.sku, hw_baseline=hw_baseline,
                                   sim_timeout=args.sim_timeout,
                                   trace_timeout=args.trace_timeout,
                                   max_workers=args.max_workers,
                                   omp_threads=args.omp_threads)
    if args.all:
        results = gcom_baseline.run_all(capabilities, ctx)
    else:
        results = gcom_baseline.run_probe(args.probe, capabilities, ctx)
    metadata = {
        "backend_capabilities": capabilities.to_dict(),
        "version": collect_version_metadata(
            profile, devices=[d.to_dict() for d in capabilities.devices]
        ),
    }
    if args.output:
        write_report(Path(args.output), results, metadata=metadata)
    else:
        _print_json(render_report(results, metadata=metadata))
    return 0


def _cmd_compare_gcom(args: argparse.Namespace) -> int:
    from amora.backends.gcom_cuda.compare import build_comparison, write_outputs

    comparison = build_comparison(args.real, args.sim)
    written = write_outputs(comparison, args.out_dir, family=args.family, sku=args.sku)
    print(json.dumps(written, indent=2))
    return 0


def _add_backend_subparser(subparsers, name: str, baseline, discover, run_func,
                           planned: tuple[str, ...]) -> argparse._SubParsersAction:
    backend = subparsers.add_parser(name)
    sub = backend.add_subparsers(dest="command")

    list_parser = sub.add_parser("list")
    list_parser.set_defaults(func=lambda _a: _cmd_list(baseline))

    cap_parser = sub.add_parser("inspect-capabilities")
    cap_parser.set_defaults(func=lambda _a: _cmd_capabilities(discover))

    run_parser = sub.add_parser("run")
    target = run_parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--probe", choices=planned)
    target.add_argument("--all", action="store_true")
    run_parser.add_argument("--output")
    return backend, sub, run_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amora")
    subparsers = parser.add_subparsers(dest="backend")

    # --- benchmarks ---
    benchmarks_parser = subparsers.add_parser("benchmarks")
    benchmarks_sub = benchmarks_parser.add_subparsers(dest="benchmark_command")
    benchmarks_list = benchmarks_sub.add_parser("list")
    benchmarks_list.set_defaults(func=_cmd_list_benchmarks)
    benchmarks_inspect = benchmarks_sub.add_parser("inspect")
    benchmarks_inspect.add_argument("benchmark_id")
    benchmarks_inspect.set_defaults(func=_cmd_inspect_benchmark)
    benchmarks_materialize = benchmarks_sub.add_parser("materialize")
    benchmarks_materialize.add_argument("benchmark_id")
    benchmarks_materialize.add_argument("--cases", type=int, default=None)
    benchmarks_materialize.add_argument("--preset", default=None)
    benchmarks_materialize.add_argument("--seed", type=int, default=None)
    benchmarks_materialize.add_argument("--vendor", default="nvidia")
    benchmarks_materialize.add_argument("--family", default="hopper")
    benchmarks_materialize.add_argument("--sku", default="h100-80g")
    benchmarks_materialize.add_argument("--arch-profile", default="sm_90_h100")
    benchmarks_materialize.add_argument("--output", type=Path, default=None)
    benchmarks_materialize.set_defaults(func=_cmd_materialize_benchmark)
    benchmarks_classify = benchmarks_sub.add_parser("classify")
    benchmarks_classify.add_argument("benchmark_id")
    benchmarks_classify.add_argument("--manifest", type=Path, required=True)
    benchmarks_classify.add_argument("--recipe", default="ncu_basic_v1")
    benchmarks_classify.add_argument("--arch", default="sm_90")
    benchmarks_classify.add_argument("--timeout", type=int, default=600)
    benchmarks_classify.add_argument("--limit", type=int, default=None)
    benchmarks_classify.add_argument("--output", type=Path, default=None)
    benchmarks_classify.set_defaults(func=_cmd_classify_benchmark)
    benchmarks_detail = benchmarks_sub.add_parser("detail")
    benchmarks_detail.add_argument("benchmark_id")
    benchmarks_detail.add_argument("--manifest", type=Path, required=True)
    benchmarks_detail.add_argument("--classification", type=Path, required=True)
    benchmarks_detail.add_argument("--size-rank", choices=("small", "medium", "large"), required=True)
    benchmarks_detail.add_argument("--review-marker", type=Path, default=None)
    benchmarks_detail.add_argument("--run-id", default=None)
    benchmarks_detail.add_argument("--limit", type=int, default=None)
    benchmarks_detail.add_argument("--sku", default="gcom_h100")
    benchmarks_detail.add_argument("--arch", default="sm_90")
    benchmarks_detail.add_argument("--ncu-timeout", type=int, default=600)
    benchmarks_detail.add_argument("--trace-timeout", type=int, default=1800)
    benchmarks_detail.add_argument("--sim-timeout", type=int, default=1200)
    benchmarks_detail.add_argument("--omp-threads", type=int, default=None)
    benchmarks_detail.add_argument("--output", type=Path, default=None)
    benchmarks_detail.set_defaults(func=_cmd_detail_benchmark)
    benchmarks_review = benchmarks_sub.add_parser("review")
    benchmarks_review.add_argument("benchmark_id")
    benchmarks_review.add_argument("--manifest", type=Path, required=True)
    benchmarks_review.add_argument("--classification", type=Path, required=True)
    benchmarks_review.add_argument("--comparison", type=Path, action="append", required=True)
    benchmarks_review.add_argument("--known-failure", action="append", default=[])
    benchmarks_review.add_argument("--semantic-decision", action="append", default=[])
    benchmarks_review.add_argument("--reviewer", default=None)
    benchmarks_review.add_argument("--reviewed-at", default=None)
    benchmarks_review.add_argument("--output", type=Path, default=None)
    benchmarks_review.set_defaults(func=_cmd_review_benchmark)

    # --- nvidia ---
    _, nvidia_sub, nvidia_run = _add_backend_subparser(
        subparsers, "nvidia", nvidia_baseline, nvidia_discover, None,
        nvidia_baseline.PLANNED_PROBES,
    )
    nvidia_run.set_defaults(func=_cmd_run_nvidia)
    report_parser = nvidia_sub.add_parser("report")
    report_parser.add_argument("--input", required=True)
    report_parser.add_argument("--out-dir", default="reports")
    report_parser.add_argument("--vendor", default=None)
    report_parser.add_argument("--family", default=None)
    report_parser.add_argument("--sku", default=None)
    report_parser.set_defaults(func=_cmd_report)

    # --- gcom_cuda ---
    from amora.backends.gcom_cuda.gcom import discover_capabilities as gcom_discover
    from amora.backends.gcom_cuda import config as gcfg
    from amora.probes.gcom_cuda import baseline as gcom_baseline

    _, gcom_sub, gcom_run = _add_backend_subparser(
        subparsers, "gcom_cuda", gcom_baseline, gcom_discover, None,
        gcom_baseline.PLANNED_PROBES,
    )
    gcom_run.add_argument("--sku", default=gcfg.DEFAULT_SKU, choices=sorted(gcfg.SKU_PROFILES))
    gcom_run.add_argument("--hw-baseline", default=None,
                          help="real NVIDIA report JSON (provides HW denominators)")
    gcom_run.add_argument("--sim-timeout", type=int, default=1200,
                          help="per-probe simulator wall-clock cap in seconds")
    gcom_run.add_argument("--trace-timeout", type=int, default=1800,
                          help="per-probe trace (instrumented kernel) cap in seconds")
    gcom_run.add_argument("--max-workers", type=int, default=8,
                          help="probes to simulate concurrently (GCoM is a CPU sim)")
    gcom_run.add_argument("--omp-threads", type=int, default=None,
                          help="OMP threads per sim (default: cores//workers, clamped 1..8, unpinned)")
    gcom_run.set_defaults(func=_cmd_run_gcom)

    compare_parser = gcom_sub.add_parser("compare")
    compare_parser.add_argument("--real", required=True)
    compare_parser.add_argument("--sim", required=True)
    compare_parser.add_argument("--out-dir", default="reports/gcom_cuda")
    compare_parser.add_argument("--family", default="hopper")
    compare_parser.add_argument("--sku", default=gcfg.DEFAULT_SKU)
    compare_parser.set_defaults(func=_cmd_compare_gcom)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(sys.stderr)
        return 2
    return int(func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
