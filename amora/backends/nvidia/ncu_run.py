"""Nsight Compute counter collection for NVIDIA probes.

This is the *profiler* execution path, kept strictly separate from timing
(`runner.run_kernel`): NCU replays each kernel to read counters, which destroys
timing fidelity, so a probe that wants both calls them in distinct passes and
keeps the two payloads in separate evidence layers.

Counters are collected as machine-readable CSV (`--csv --page raw`) so AMORA
does not depend on the binary ``.ncu-rep`` format. Missing/locked-down NCU is a
clean capability gate, never fatal.
"""

from __future__ import annotations

import csv
import io
import subprocess
from dataclasses import dataclass, field

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.ncu import NcuCommand
from amora.backends.nvidia.runner import DEFAULT_ARCH, DEFAULT_BUILD_ROOT, build_executable
from pathlib import Path


class NcuUnavailable(RuntimeError):
    """Raised when NCU counter collection cannot run on this host."""


@dataclass(frozen=True)
class NcuResult:
    metrics: dict[str, float]
    raw_rows: list[dict[str, str]] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


def parse_ncu_csv(text: str) -> tuple[dict[str, float], list[dict[str, str]]]:
    """Parse ``ncu --csv --page raw`` output (wide format).

    NCU emits one CSV *column* per metric (plus identity columns like
    ``Kernel Name``). The probe driver may also print its own JSON to stdout
    before the CSV, so the header row is detected as the first line containing a
    quoted ``Kernel Name`` column.

    Returns (metric_name -> value, raw_rows). Numeric values may carry thousands
    separators; non-numeric cells (units, names) are skipped. When multiple
    kernel rows are present the last numeric value per metric wins.
    """

    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if '"Kernel Name"' in line:
            header_idx = i
            break
    if header_idx is None:
        return {}, []

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    fieldnames = [f for f in (reader.fieldnames or []) if f]
    # Metric columns are those that look like NCU metrics (contain '__' or '.').
    metric_cols = [f for f in fieldnames if "__" in f or "." in f]

    metrics: dict[str, float] = {}
    rows: list[dict[str, str]] = []
    for row in reader:
        # Skip the units row NCU emits right after the header (blank Kernel Name).
        if not (row.get("Kernel Name") or "").strip():
            continue
        rows.append(row)
        for col in metric_cols:
            raw_value = (row.get(col) or "").strip()
            if not raw_value:
                continue
            cleaned = raw_value.replace(",", "")
            try:
                metrics[col] = float(cleaned)
            except ValueError:
                continue
    return metrics, rows


def _ncu_path(capabilities: NvidiaCapabilities) -> str:
    tool = capabilities.tools.get("ncu")
    if not tool or not tool.available or not tool.path:
        raise NcuUnavailable("ncu is not available on PATH")
    return tool.path


def run_kernel_profiled(
    source: Path,
    *,
    capabilities: NvidiaCapabilities,
    metrics: tuple[str, ...],
    args: tuple[str, ...] = (),
    kernel_name: str | None = None,
    launch_count: int = 1,
    timeout: int = 180,
    arch: str = DEFAULT_ARCH,
    build_root: Path = DEFAULT_BUILD_ROOT,
) -> NcuResult:
    """Build (reusing the timing cache) and run the driver under NCU for counters.

    Raises :class:`NcuUnavailable` on missing tool, permission errors, or empty
    counter output so callers can fall back to timing-only cleanly.
    """

    if not metrics:
        raise NcuUnavailable("no metrics requested")
    if any(m is None for m in metrics):
        raise NcuUnavailable("metric list contains an unresolved (None) entry")
    ncu = _ncu_path(capabilities)
    binary, _ = build_executable(
        source, capabilities=capabilities, arch=arch, build_root=build_root
    )
    command = NcuCommand(
        executable=ncu,
        metrics=tuple(metrics),
        target=(str(binary), *args),
        csv=True,
        page="raw",
        launch_count=launch_count,
        kernel_name=kernel_name,
    )
    try:
        completed = subprocess.run(
            command.argv(), check=False, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.SubprocessError as exc:
        raise NcuUnavailable(f"ncu execution failed: {exc}") from exc
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise NcuUnavailable(
            f"ncu rc={completed.returncode}: {stderr[:300] or 'no stderr'}"
        )
    parsed, rows = parse_ncu_csv(completed.stdout)
    if not parsed:
        raise NcuUnavailable("ncu produced no parseable counter rows")
    return NcuResult(
        metrics=parsed,
        raw_rows=rows,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )
