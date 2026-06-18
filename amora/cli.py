"""Command-line interface for AMORA."""

from __future__ import annotations

import argparse
from pathlib import Path

from amora.backends.nvidia import CudaToolchain
from amora.probes.nvidia.p0 import (
    arithmetic_source_probe,
    device_attribute_probe,
    occupancy_plan_probe,
    shared_memory_source_probe,
)
from amora.reports.json_report import render_json
from amora.schemas.results import HardwareProfile, ProbeResult


def run_nvidia_p0(*, dry_run: bool) -> HardwareProfile:
    toolchain = CudaToolchain()
    results: list[ProbeResult] = [
        device_attribute_probe(toolchain=toolchain),
        occupancy_plan_probe(),
        arithmetic_source_probe(),
        shared_memory_source_probe(),
    ]
    target = toolchain.target_summary()
    target["dry_run"] = dry_run
    return HardwareProfile(target=target, raw_results=results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amora")
    parser.add_argument("--target", choices=("nvidia",), default="nvidia")
    parser.add_argument("--tier", choices=("p0",), default="p0")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan probes and inspect tools without launching CUDA kernels.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.target == "nvidia" and args.tier == "p0":
        profile = run_nvidia_p0(dry_run=args.dry_run)
    else:
        parser.error("unsupported target/tier combination")

    rendered = render_json(profile.to_dict())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0
