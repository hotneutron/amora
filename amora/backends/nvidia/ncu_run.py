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
import hashlib
import io
import tempfile
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.ncu import NcuCommand, detect_sampling_interval_option
from amora.backends.nvidia.runner import (
    DEFAULT_ARCH,
    DEFAULT_BUILD_ROOT,
    CudaUnavailable,
    build_executable,
)
from amora.backends.nvidia.stall_metrics import STALL_REASONS, parse_numeric_cell


class NcuUnavailable(RuntimeError):
    """Raised when NCU counter collection cannot run on this host."""


@dataclass(frozen=True)
class NcuResult:
    metrics: dict[str, float]
    raw_rows: list[dict[str, str]] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    source_path: Path | None = None
    source_sha256: str | None = None
    binary_path: Path | None = None
    binary_sha256: str | None = None
    command: tuple[str, ...] = ()
    target_command: tuple[str, ...] = ()
    arch: str | None = None
    extra_flags: tuple[str, ...] = ()
    link_flags: tuple[str, ...] = ()

    def provenance(self) -> dict[str, object]:
        """Return the execution details needed to reproduce this NCU run."""

        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "source_sha256": self.source_sha256,
            "binary_path": str(self.binary_path) if self.binary_path else None,
            "binary_sha256": self.binary_sha256,
            "ncu_command": list(self.command),
            "target_command": list(self.target_command),
            "arch": self.arch,
            "extra_flags": list(self.extra_flags),
            "link_flags": list(self.link_flags),
            "returncode": self.returncode,
        }


@dataclass(frozen=True)
class PcStallSample:
    """One source-page row joined into a compact PC stall record."""

    pc_offset: int
    function: str | None
    opcode: str | None
    samples: float
    stalls: dict[str, float]
    raw_address: int | None = None
    address_base: int = 0
    raw_row: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "pc_offset": self.pc_offset,
            "raw_address": self.raw_address,
            "address_base": self.address_base,
            "function": self.function,
            "opcode": self.opcode,
            "samples": self.samples,
            "stalls": dict(self.stalls),
            "raw_row": dict(self.raw_row),
        }


@dataclass(frozen=True)
class NcuPcSamplingResult:
    """Source-correlated PC sampling data extracted from an NCU report."""

    samples: list[PcStallSample]
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    source_path: Path | None = None
    source_sha256: str | None = None
    binary_path: Path | None = None
    binary_sha256: str | None = None
    report_path: Path | None = None
    profile_command: tuple[str, ...] = ()
    import_command: tuple[str, ...] = ()
    target_command: tuple[str, ...] = ()
    arch: str | None = None
    sampling_interval: str | None = None
    sampling_interval_option: str | None = None
    section: str | None = None
    parser_metadata: dict[str, object] = field(default_factory=dict)
    extra_flags: tuple[str, ...] = ()
    link_flags: tuple[str, ...] = ()

    def provenance(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "source_sha256": self.source_sha256,
            "binary_path": str(self.binary_path) if self.binary_path else None,
            "binary_sha256": self.binary_sha256,
            "report_path": str(self.report_path) if self.report_path else None,
            "ncu_profile_command": list(self.profile_command),
            "ncu_import_command": list(self.import_command),
            "target_command": list(self.target_command),
            "arch": self.arch,
            "sampling_interval": self.sampling_interval,
            "sampling_interval_option": self.sampling_interval_option,
            "section": self.section,
            "source_parser": dict(self.parser_metadata),
            "extra_flags": list(self.extra_flags),
            "link_flags": list(self.link_flags),
            "returncode": self.returncode,
        }


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


_PC_COLUMNS = (
    "PC",
    "Address",
    "Source Counters: PC",
    "SASS Address",
)
_FUNCTION_COLUMNS = (
    "Kernel Name",
    "Function Name",
    "Function",
)
_SASS_COLUMNS = (
    "SASS",
    "Instruction",
    "Source",
)
_SOURCE_REASON_ALIASES = {
    "stall_barrier": "barrier",
    "stall_branch_resolving": "branch_resolving",
    "stall_dispatch": "dispatch_stall",
    "stall_drain": "drain",
    "stall_imc": "imc_miss",
    "stall_lg": "lg_throttle",
    "stall_long_sb": "long_scoreboard",
    "stall_math": "math_pipe_throttle",
    "stall_membar": "membar",
    "stall_mio": "mio_throttle",
    "stall_misc": "misc",
    "stall_no_inst": "no_instructions",
    "stall_not_selected": "not_selected",
    "stall_selected": "selected",
    "stall_short_sb": "short_scoreboard",
    "stall_sleep": "sleeping",
    "stall_sleeping": "sleeping",
    "stall_tex": "tex_throttle",
    "stall_wait": "wait",
    "stall_warpgroup_arrive": "warpgroup_arrive",
}
_ABSOLUTE_PC_THRESHOLD = 0x100000


def _find_csv_header(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = next(csv.reader([line]))
        except csv.Error:
            continue
        lowered = {cell.strip().lower() for cell in row}
        if (
            lowered & {"pc", "address", "sass address"}
            or any(" pc" in cell or cell.endswith(": pc") for cell in lowered)
        ):
            return i
    return None


def _pc_offset(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    # NCU can print either "0x60" or cuobjdump-style "/*0060*/".
    text = text.strip("`")
    cuobjdump_hex = False
    if text.startswith("/*") and text.endswith("*/"):
        text = text[2:-2]
        cuobjdump_hex = True
    try:
        if cuobjdump_hex or text.lower().startswith("0x"):
            return int(text, 16)
        if any(ch in text.lower() for ch in "abcdef"):
            return int(text, 16)
        return int(text)
    except ValueError:
        return None


def _first_present(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _reason_from_column(column: str) -> tuple[str | None, bool]:
    lowered = column.strip().lower()
    display_not_issued = lowered.endswith(" (not issued)")
    raw_not_issued = lowered.endswith("_not_issued")
    explicitly_not_issued = display_not_issued or raw_not_issued
    base = lowered.removesuffix(" (not issued)").removesuffix("_not_issued")
    display_reason = _SOURCE_REASON_ALIASES.get(base)
    if display_reason is not None:
        return display_reason, explicitly_not_issued

    prefix = "smsp__pcsamp_warps_issue_stalled_"
    if prefix not in base:
        return None, explicitly_not_issued
    reason = base.split(prefix, 1)[1].split(".", 1)[0]
    reason = {
        "no_instruction": "no_instructions",
        "gmma": "warpgroup_arrive",
    }.get(reason, reason)
    return (reason if reason in STALL_REASONS else None), explicitly_not_issued


def _source_context_function(lines: list[str]) -> str | None:
    """Read the function label emitted ahead of NCU 2026 source tables."""

    for line in reversed(lines):
        try:
            row = next(csv.reader([line]))
        except csv.Error:
            continue
        if len(row) >= 2 and row[0].strip().lower() == "kernel name":
            return row[1].strip() or None
    return None


def parse_ncu_source_pc_sampling_csv_with_metadata(
    text: str,
) -> tuple[list[PcStallSample], dict[str, object]]:
    """Parse source-page PC rows and describe schema/address normalization."""

    lines = text.splitlines()
    header_idx = _find_csv_header(lines)
    if header_idx is None:
        return [], {
            "status": "no_address_header",
            "address_mode": None,
            "address_base": None,
            "address_bases": {},
            "column_mode": None,
            "present_reasons": [],
            "missing_reasons": list(STALL_REASONS),
        }
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    if not reader.fieldnames:
        return [], {
            "status": "no_columns",
            "address_mode": None,
            "address_base": None,
            "address_bases": {},
            "column_mode": None,
            "present_reasons": [],
            "missing_reasons": list(STALL_REASONS),
        }

    recognized = {
        col: parsed
        for col in reader.fieldnames
        if col
        for parsed in [_reason_from_column(col)]
        if parsed[0] is not None
    }
    not_issued = {
        col: reason
        for col, (reason, is_not_issued) in recognized.items()
        if is_not_issued and reason is not None
    }
    reason_cols = not_issued or {
        col: reason
        for col, (reason, _is_not_issued) in recognized.items()
        if reason is not None
    }
    context_function = _source_context_function(lines[:header_idx])
    raw_rows: list[tuple[int, str | None, dict[str, str]]] = []
    for row in reader:
        address = _pc_offset(_first_present(row, _PC_COLUMNS))
        if address is None:
            continue
        function = _first_present(row, _FUNCTION_COLUMNS) or context_function
        raw_rows.append(
            (address, function, {key: value for key, value in row.items() if key is not None})
        )

    if not raw_rows:
        present = sorted(set(reason_cols.values()))
        return [], {
            "status": "no_address_rows",
            "address_mode": None,
            "address_base": None,
            "address_bases": {},
            "column_mode": "not_issued" if not_issued else "all_samples",
            "present_reasons": present,
            "missing_reasons": sorted(set(STALL_REASONS) - set(present)),
        }

    addresses_by_function: dict[str | None, list[int]] = {}
    for address, function, _row in raw_rows:
        addresses_by_function.setdefault(function, []).append(address)
    bases = {
        function: min(addresses) if min(addresses) >= _ABSOLUTE_PC_THRESHOLD else 0
        for function, addresses in addresses_by_function.items()
    }
    modes = {"absolute" if base else "relative" for base in bases.values()}
    address_mode = next(iter(modes)) if len(modes) == 1 else "mixed"

    samples: list[PcStallSample] = []
    for address, function, row in raw_rows:
        stalls: dict[str, float] = {}
        for col, reason in reason_cols.items():
            value = parse_numeric_cell(row.get(col))
            if value is not None and value > 0.0:
                stalls[reason] = stalls.get(reason, 0.0) + value
        if not stalls:
            continue
        base = bases[function]
        samples.append(
            PcStallSample(
                pc_offset=address - base,
                raw_address=address,
                address_base=base,
                function=function,
                opcode=_first_present(row, _SASS_COLUMNS),
                samples=sum(stalls.values()),
                stalls=stalls,
                raw_row=row,
            )
        )

    printable_bases = {
        function or "<unknown>": base for function, base in bases.items()
    }
    unique_bases = set(bases.values())
    present = sorted(set(reason_cols.values()))
    return samples, {
        "status": "pass" if samples else "zero_support",
        "address_mode": address_mode,
        "address_base": next(iter(unique_bases)) if len(unique_bases) == 1 else None,
        "address_bases": printable_bases,
        "column_mode": "not_issued" if not_issued else "all_samples",
        "present_reasons": present,
        "missing_reasons": sorted(set(STALL_REASONS) - set(present)),
    }


def parse_ncu_source_pc_sampling_csv(text: str) -> list[PcStallSample]:
    """Parse ``ncu --import <report> --page source --csv`` PC rows.

    Source-page CSV differs across NCU versions, so this parser uses stable
    cues: a PC/address column, any columns containing
    ``issue_stalled_<reason>``, and optional function/SASS text columns.
    """

    samples, _metadata = parse_ncu_source_pc_sampling_csv_with_metadata(text)
    return samples


def _ncu_path(capabilities: NvidiaCapabilities) -> str:
    tool = capabilities.tools.get("ncu")
    if not tool or not tool.available or not tool.path:
        raise NcuUnavailable("ncu is not available on PATH")
    return tool.path


def _run_ncu(command_argv: tuple[str, ...], *, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command_argv, check=False, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.SubprocessError as exc:
        raise NcuUnavailable(f"ncu execution failed: {exc}") from exc


def _raise_on_ncu_failure(completed: subprocess.CompletedProcess[str]) -> None:
    if completed.returncode == 0:
        return
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    diagnostic = stderr or stdout
    raise NcuUnavailable(
        f"ncu rc={completed.returncode}: {diagnostic[:300] or 'no output'}"
    )


def _validate_target(target: tuple[str, ...]) -> None:
    if not target or not target[0]:
        raise NcuUnavailable("target command must not be empty")


def run_command_profiled(
    target: tuple[str, ...],
    *,
    capabilities: NvidiaCapabilities,
    metrics: tuple[str, ...],
    kernel_name: str | None = None,
    kernel_name_base: str | None = None,
    launch_skip: int | None = None,
    launch_count: int = 1,
    timeout: int = 180,
) -> NcuResult:
    """Run an arbitrary target command under NCU for aggregate counters.

    This is the reusable entry point for Python and JIT launchers. Callers own
    target preparation; AMORA owns NCU invocation, parsing, and provenance.
    """

    _validate_target(target)
    if not metrics:
        raise NcuUnavailable("no metrics requested")
    if any(metric is None for metric in metrics):
        raise NcuUnavailable("metric list contains an unresolved (None) entry")
    command = NcuCommand(
        executable=_ncu_path(capabilities),
        metrics=tuple(metrics),
        target=tuple(target),
        csv=True,
        page="raw",
        launch_skip=launch_skip,
        launch_count=launch_count,
        kernel_name=kernel_name,
        kernel_name_base=kernel_name_base,
    )
    command_argv = tuple(command.argv())
    completed = _run_ncu(command_argv, timeout=timeout)
    _raise_on_ncu_failure(completed)
    parsed, rows = parse_ncu_csv(completed.stdout)
    if not parsed:
        raise NcuUnavailable("ncu produced no parseable counter rows")
    return NcuResult(
        metrics=parsed,
        raw_rows=rows,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
        command=command_argv,
        target_command=tuple(target),
    )


def run_kernel_profiled(
    source: Path,
    *,
    capabilities: NvidiaCapabilities,
    metrics: tuple[str, ...],
    args: tuple[str, ...] = (),
    kernel_name: str | None = None,
    kernel_name_base: str | None = None,
    launch_skip: int | None = None,
    launch_count: int = 1,
    timeout: int = 180,
    arch: str = DEFAULT_ARCH,
    build_root: Path = DEFAULT_BUILD_ROOT,
    extra_flags: tuple[str, ...] = ("-O2",),
    link_flags: tuple[str, ...] = (),
) -> NcuResult:
    """Build (reusing the timing cache) and run the driver under NCU for counters.

    Raises :class:`NcuUnavailable` on missing tool, permission errors, or empty
    counter output so callers can fall back to timing-only cleanly.
    """

    if not metrics:
        raise NcuUnavailable("no metrics requested")
    if any(metric is None for metric in metrics):
        raise NcuUnavailable("metric list contains an unresolved (None) entry")
    _ncu_path(capabilities)
    try:
        binary, source_sha = build_executable(
            source,
            capabilities=capabilities,
            arch=arch,
            build_root=build_root,
            extra_flags=extra_flags,
            link_flags=link_flags,
        )
    except CudaUnavailable as exc:
        raise NcuUnavailable(str(exc)) from exc
    result = run_command_profiled(
        (str(binary), *args),
        capabilities=capabilities,
        metrics=metrics,
        launch_skip=launch_skip,
        launch_count=launch_count,
        kernel_name=kernel_name,
        kernel_name_base=kernel_name_base,
        timeout=timeout,
    )
    return replace(
        result,
        source_path=source,
        source_sha256=source_sha,
        binary_path=binary,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        arch=arch,
        extra_flags=extra_flags,
        link_flags=link_flags,
    )


def run_command_pc_sampling(
    target: tuple[str, ...],
    *,
    capabilities: NvidiaCapabilities,
    kernel_name: str | None = None,
    kernel_name_base: str | None = None,
    launch_skip: int | None = None,
    launch_count: int = 1,
    timeout: int = 300,
    section: str = "SourceCounters",
    sampling_interval: str = "auto",
    sampling_interval_option: str | None = None,
    report_path: Path | None = None,
) -> NcuPcSamplingResult:
    """Collect source-correlated stalls from an arbitrary target command."""

    _validate_target(target)
    ncu = _ncu_path(capabilities)
    if sampling_interval_option is None:
        sampling_interval_option, error = detect_sampling_interval_option(
            ncu, timeout=min(timeout, 30)
        )
        if sampling_interval_option is None:
            raise NcuUnavailable(error or "unable to detect NCU sampling option")
    if report_path is None:
        tmp = tempfile.NamedTemporaryFile(
            prefix="amora-pcsamp-", suffix=".ncu-rep", delete=False
        )
        tmp.close()
        report = Path(tmp.name)
        report.unlink(missing_ok=True)
    else:
        report = report_path
    profile_command = NcuCommand(
        executable=ncu,
        metrics=(),
        target=tuple(target),
        section=section,
        launch_skip=launch_skip,
        launch_count=launch_count,
        kernel_name=kernel_name,
        kernel_name_base=kernel_name_base,
        sampling_interval=sampling_interval,
        sampling_interval_option=sampling_interval_option,
        output=str(report),
        force_overwrite=True,
    )
    profile_argv = tuple(profile_command.argv())
    completed = _run_ncu(profile_argv, timeout=timeout)
    _raise_on_ncu_failure(completed)

    import_command = NcuCommand(
        executable=ncu,
        metrics=(),
        target=(),
        csv=True,
        page="source",
        print_source="sass",
        import_report=str(report),
    )
    import_argv = tuple(import_command.argv())
    imported = _run_ncu(import_argv, timeout=timeout)
    _raise_on_ncu_failure(imported)
    samples, parser_metadata = parse_ncu_source_pc_sampling_csv_with_metadata(
        imported.stdout
    )
    if not samples:
        raise NcuUnavailable(
            "ncu source page produced no PC stall rows: "
            f"{parser_metadata.get('status', 'unknown parser status')}"
        )
    return NcuPcSamplingResult(
        samples=samples,
        stdout=imported.stdout,
        stderr="\n".join(
            part for part in (completed.stderr, imported.stderr) if part
        ),
        returncode=imported.returncode,
        report_path=report,
        profile_command=profile_argv,
        import_command=import_argv,
        target_command=tuple(target),
        sampling_interval=sampling_interval,
        sampling_interval_option=sampling_interval_option,
        section=section,
        parser_metadata=parser_metadata,
    )


def run_kernel_pc_sampling(
    source: Path,
    *,
    capabilities: NvidiaCapabilities,
    args: tuple[str, ...] = (),
    kernel_name: str | None = None,
    kernel_name_base: str | None = None,
    launch_skip: int | None = None,
    launch_count: int = 1,
    timeout: int = 300,
    arch: str = DEFAULT_ARCH,
    build_root: Path = DEFAULT_BUILD_ROOT,
    extra_flags: tuple[str, ...] = ("-O2",),
    link_flags: tuple[str, ...] = (),
    section: str = "SourceCounters",
    sampling_interval: str = "auto",
    sampling_interval_option: str | None = None,
    report_path: Path | None = None,
) -> NcuPcSamplingResult:
    """Collect source-correlated PC stall samples through an exported NCU report."""

    try:
        binary, source_sha = build_executable(
            source,
            capabilities=capabilities,
            arch=arch,
            build_root=build_root,
            extra_flags=extra_flags,
            link_flags=link_flags,
        )
    except CudaUnavailable as exc:
        raise NcuUnavailable(str(exc)) from exc
    result = run_command_pc_sampling(
        (str(binary), *args),
        capabilities=capabilities,
        section=section,
        launch_skip=launch_skip,
        launch_count=launch_count,
        kernel_name=kernel_name,
        kernel_name_base=kernel_name_base,
        sampling_interval=sampling_interval,
        sampling_interval_option=sampling_interval_option,
        report_path=report_path,
        timeout=timeout,
    )
    return replace(
        result,
        source_path=source,
        source_sha256=source_sha,
        binary_path=binary,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        arch=arch,
        extra_flags=extra_flags,
        link_flags=link_flags,
    )
