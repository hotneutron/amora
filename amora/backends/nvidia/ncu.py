"""Nsight Compute command helpers."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


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

    def argv(self) -> list[str]:
        args = [self.executable, "--target-processes", "all"]
        if self.csv:
            args.append("--csv")
        if self.page:
            args.extend(["--page", self.page])
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
