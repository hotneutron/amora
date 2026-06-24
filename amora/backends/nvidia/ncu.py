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

    def argv(self) -> list[str]:
        args = [self.executable, "--target-processes", "all"]
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
