"""Nsight Compute command helpers."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


NCU_SAMPLING_INTERVAL_OPTIONS = (
    "--warp-sampling-interval",
    "--sampling-interval",
)


@dataclass(frozen=True)
class NcuCommand:
    executable: str
    metrics: tuple[str, ...]
    target: tuple[str, ...]
    output: str | None = None
    csv: bool = False
    page: str | None = None
    launch_skip: int | None = None
    launch_count: int | None = None
    kernel_name: str | None = None
    kernel_name_base: str | None = None
    section: str | None = None
    set_name: str | None = None
    sampling_interval: str | None = None
    sampling_interval_option: str = "--sampling-interval"
    print_source: str | None = None
    force_overwrite: bool = False
    import_report: str | None = None

    def argv(self) -> list[str]:
        args = [self.executable]
        if not self.import_report:
            args.extend(["--target-processes", "all"])
        if self.csv:
            args.append("--csv")
        if self.import_report:
            args.extend(["--import", self.import_report])
        if self.page:
            args.extend(["--page", self.page])
        if self.section:
            args.extend(["--section", self.section])
        if self.set_name:
            args.extend(["--set", self.set_name])
        if self.sampling_interval:
            if self.sampling_interval_option not in NCU_SAMPLING_INTERVAL_OPTIONS:
                raise ValueError(
                    "unsupported NCU sampling interval option: "
                    f"{self.sampling_interval_option}"
                )
            args.extend([self.sampling_interval_option, self.sampling_interval])
        if self.print_source:
            args.extend(["--print-source", self.print_source])
        if self.launch_skip is not None:
            args.extend(["--launch-skip", str(self.launch_skip)])
        if self.launch_count is not None:
            args.extend(["--launch-count", str(self.launch_count)])
        if self.kernel_name_base:
            args.extend(["--kernel-name-base", self.kernel_name_base])
        if self.kernel_name:
            args.extend(["--kernel-name", self.kernel_name])
        if self.metrics:
            args.extend(["--metrics", ",".join(self.metrics)])
        if self.output:
            args.extend(["--export", self.output])
        if self.force_overwrite:
            args.append("--force-overwrite")
        args.extend(self.target)
        return args


def list_metrics(executable: str = "ncu") -> tuple[int, str, str]:
    completed = subprocess.run(
        [executable, "--query-metrics"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stdout, completed.stderr


def parse_sampling_interval_option(help_text: str) -> str | None:
    """Return the warp-sampling interval option advertised by NCU.

    Nsight Compute 2026.2 renamed ``--sampling-interval`` to
    ``--warp-sampling-interval``. Prefer the specific new spelling if a help
    page happens to mention both.
    """

    for option in NCU_SAMPLING_INTERVAL_OPTIONS:
        if option in help_text:
            return option
    return None


def detect_sampling_interval_option(
    executable: str = "ncu",
    *,
    timeout: int = 30,
) -> tuple[str | None, str | None]:
    """Probe NCU help and return ``(option, error)`` without raising."""

    try:
        completed = subprocess.run(
            [executable, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"ncu --help failed: {exc}"
    help_text = "\n".join(
        part for part in (completed.stdout, completed.stderr) if part
    )
    option = parse_sampling_interval_option(help_text)
    if option is not None:
        return option, None
    diagnostic = help_text.strip().splitlines()
    detail = diagnostic[-1] if diagnostic else f"return code {completed.returncode}"
    return None, f"ncu exposes no recognized warp-sampling interval option: {detail}"
