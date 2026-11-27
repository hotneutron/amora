"""CLI registration for explicit FlashGPU-Sim feature validation."""
import argparse
import json
from pathlib import Path
import sys

from .build import build_probe
from .runner import capture, run_test, summarize
from amora.probes.nvidia.feature_validation import list_probes, PROBES

def emit(value):
    print(json.dumps(value, indent=2))
    return value

def dispatch(args):
    if args.command == "portable-v2":
        from amora.benchmarking.portable_v2.campaign import dispatch as run_campaign
        return run_campaign(args)
    if args.command == "list":
        emit(dict(probes=list_probes()))
        return 0
    if args.command == "summarize":
        payload = summarize(args.evidence_root)
        emit(payload)
        if args.output:
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
        return 0
    if args.command == "extract-spec":
        from amora.probes.nvidia.feature_validation.spec import extract_spec
        emit(extract_spec(args.evidence_root, cache=args.cache, url=args.url))
        return 0
    if args.command == "build":
        emit(build_probe(args.probe, sim_root=args.sim_root, cuda_root=args.cuda_root,
                         build_root=args.build_root, evidence_root=args.evidence_root))
        return 0
    if args.command == "command":
        command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        if not command:
            raise ValueError("command required after --")
        record = capture(command, label=args.label, evidence_root=args.evidence_root,
                         cwd=args.cwd, timeout=args.timeout)
    else:
        environment = {}
        for item in args.env:
            key, separator, value = item.partition("=")
            if not separator or not key:
                raise ValueError("environment override must have NAME=VALUE form")
            environment[key] = value
        record = run_test(binary=args.binary, sim_root=args.sim_root,
                          cuda_root=args.cuda_root, evidence_root=args.evidence_root,
                          label=args.label, preset=args.preset, mode=args.mode,
                          gtest_filter=args.gtest_filter, overrides=args.config_override,
                          timeout=args.timeout, scratch_root=args.scratch_root,
                          gpu=args.gpu, omp_threads=args.omp_threads,
                          environment_overrides=environment,
                          program_arguments=args.arg)
    emit(record)
    return 0 if record["exit_code"] == 0 else 1

def add_commands(parser):
    sub = parser.add_subparsers(dest="command")
    portable = sub.add_parser("portable-v2", help="Validate source-pinned portable_core v2 CUDA programs")
    from amora.benchmarking.portable_v2.campaign import configure_parser
    configure_parser(portable)
    portable.set_defaults(func=dispatch)
    listing = sub.add_parser("list", help="List independent feature probes and source hashes")
    listing.set_defaults(func=dispatch)
    summary = sub.add_parser("summarize", help="Summarize XML and command receipts")
    summary.add_argument("--evidence-root", type=Path, required=True)
    summary.add_argument("--output", type=Path)
    summary.set_defaults(func=dispatch)
    spec = sub.add_parser("extract-spec", help="Fetch and preserve NVIDIA PTX sections")
    spec.add_argument("--evidence-root", type=Path, required=True)
    spec.add_argument("--cache", type=Path)
    spec.add_argument("--url", default="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html")
    spec.set_defaults(func=dispatch)
    build = sub.add_parser("build", help="Build an independent feature probe")
    build.add_argument("--probe", choices=tuple(PROBES), required=True)
    build.add_argument("--sim-root", type=Path, required=True)
    build.add_argument("--cuda-root", type=Path, required=True)
    build.add_argument("--build-root", type=Path, required=True)
    build.add_argument("--evidence-root", type=Path, required=True)
    build.set_defaults(func=dispatch)
    command = sub.add_parser("command", help="Capture any validation/build command")
    command.add_argument("--label", required=True)
    command.add_argument("--evidence-root", type=Path, required=True)
    command.add_argument("--cwd", type=Path, required=True)
    command.add_argument("--timeout", type=float, default=600)
    command.add_argument("argv", nargs=argparse.REMAINDER)
    command.set_defaults(func=dispatch)
    run = sub.add_parser("run", help="Run an exact GoogleTest or standalone binary on hardware or simulator")
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--sim-root", type=Path, required=True)
    run.add_argument("--cuda-root", type=Path, required=True)
    run.add_argument("--evidence-root", type=Path, required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--preset", default="SM90_H100")
    run.add_argument("--mode", choices=("native", "ptx", "sass", "sass-timing"), default="ptx")
    run.add_argument("--gtest-filter", default="*")
    run.add_argument("--config-override", action="append", default=[])
    run.add_argument("--timeout", type=float, default=600)
    run.add_argument("--scratch-root", type=Path)
    run.add_argument("--gpu", default="0")
    run.add_argument("--omp-threads", type=int, default=4)
    run.add_argument("--env", action="append", default=[])
    run.add_argument("--arg", action="append", default=None,
                     help="Standalone program argument; repeat --arg=VALUE to omit GoogleTest flags")
    run.set_defaults(func=dispatch)

def register_backend(subparsers):
    add_commands(subparsers.add_parser("flashgpu_sim", help="Validate GPU feature semantics and timing in FlashGPU-Sim"))

def main(argv=None):
    parser = argparse.ArgumentParser(prog="amora flashgpu_sim")
    add_commands(parser)
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)
        return 2
    return dispatch(args)

