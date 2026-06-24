"""Command-line interface for AMORA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from amora.backends.nvidia.cuda import discover_capabilities
from amora.probes.nvidia import baseline as baseline_probes
from amora.reports.json_report import render_report, write_report
from amora.reports.markdown_report import write_reports_from_json


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _cmd_list(_: argparse.Namespace) -> int:
    _print_json({"probes": baseline_probes.list_probes()})
    return 0


def _cmd_capabilities(_: argparse.Namespace) -> int:
    _print_json(discover_capabilities().to_dict())
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    capabilities = discover_capabilities()
    if args.all:
        results = baseline_probes.run_all(capabilities)
    else:
        results = baseline_probes.run_probe(args.probe, capabilities)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amora")
    subparsers = parser.add_subparsers(dest="backend")

    nvidia = subparsers.add_parser("nvidia")
    nvidia_sub = nvidia.add_subparsers(dest="command")

    list_parser = nvidia_sub.add_parser("list")
    list_parser.set_defaults(func=_cmd_list)

    capabilities_parser = nvidia_sub.add_parser("inspect-capabilities")
    capabilities_parser.set_defaults(func=_cmd_capabilities)

    run_parser = nvidia_sub.add_parser("run")
    target = run_parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--probe", choices=baseline_probes.PLANNED_PROBES)
    target.add_argument("--all", action="store_true")
    run_parser.add_argument("--output")
    run_parser.set_defaults(func=_cmd_run)

    report_parser = nvidia_sub.add_parser("report")
    report_parser.add_argument("--input", required=True)
    report_parser.add_argument("--out-dir", default="reports")
    report_parser.add_argument("--vendor", default=None)
    report_parser.add_argument("--family", default=None)
    report_parser.add_argument("--sku", default=None)
    report_parser.set_defaults(func=_cmd_report)

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
