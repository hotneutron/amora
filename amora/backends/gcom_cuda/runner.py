"""Run the GCoM simulator on a trace and parse its emitted stats.

``simulate`` invokes ``accel-sim.out`` with the SKU's gpgpusim + trace configs,
tees output to a log, and ``parse_stats`` extracts the ``key = value`` stat
lines (last value per key wins). ``derive_logical_metrics`` translates GCoM stat
keys into AMORA logical metric names via the (versioned) gcom metric map.
Requires a built simulator; not exercised by no-GPU tests except for the pure
``parse_stats`` parser.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from amora.backends.gcom_cuda import config as cfg

# GPGPU-Sim prints "key = value" stat lines.
_STAT_LINE = re.compile(r"^\s*([A-Za-z0-9_:\.\[\]]+)\s*=\s*([-\d.eE+]+)\s*$")

# A run is only meaningful if this core execution stat is present.
REQUIRED_CORE_STAT = "gpu_sim_cycle"


@dataclass(frozen=True)
class SimResult:
    stats: dict[str, float]
    returncode: int
    log_path: Path | None = None
    stdout: str = ""
    stderr: str = ""

    def core_present(self) -> bool:
        return REQUIRED_CORE_STAT in self.stats


class SimulateError(RuntimeError):
    """Raised when the simulator cannot be run."""


def parse_stats(stdout: str) -> dict[str, float]:
    """Extract ``key = value`` numeric stats (last value per key wins)."""

    stats: dict[str, float] = {}
    for line in stdout.splitlines():
        m = _STAT_LINE.match(line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2)
        try:
            stats[key] = float(raw)
        except ValueError:
            continue
    return stats


def _find_trace_pb(trace_dir: Path) -> Path:
    pb = trace_dir / "dynamic_trace.pb"
    if pb.exists():
        return pb
    candidates = list(trace_dir.rglob("dynamic_trace.pb"))
    if not candidates:
        raise SimulateError(f"no dynamic_trace.pb found under {trace_dir}")
    return candidates[0]


def _sim_env() -> dict[str, str]:
    env = dict(os.environ)
    cuda = env.get("CUDA_INSTALL_PATH", "/usr/local/cuda")
    sim_lib = cfg.SIM_BIN.parent.parent.parent / "gpgpu-sim" / "lib"
    lib_paths = [str(p) for p in sim_lib.rglob("*release*") if p.is_dir()]
    lib_paths.append(f"{cuda}/lib64")
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, [*lib_paths, existing]))
    # OMP tuning per accorde reference.
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_PLACES", "cores")
    return env


def simulate(
    profile: cfg.SkuProfile,
    trace_dir: Path,
    *,
    log_path: Path | None = None,
    timeout: int = 7200,
) -> SimResult:
    """Run accel-sim.out on a trace dir with the SKU configs; parse stats."""

    if not cfg.SIM_BIN.exists():
        raise SimulateError(f"simulator binary not built: {cfg.SIM_BIN}")
    trace_pb = _find_trace_pb(trace_dir)
    args = [
        str(cfg.SIM_BIN),
        "-trace", str(trace_pb),
        "-config", str(profile.gpgpusim_config),
        "-config", str(profile.trace_config),
    ]
    completed = subprocess.run(
        args, cwd=str(trace_dir), env=_sim_env(),
        capture_output=True, text=True, timeout=timeout,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr)
    stats = parse_stats(completed.stdout)
    return SimResult(
        stats=stats,
        returncode=completed.returncode,
        log_path=log_path,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def derive_logical_metrics(stats: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Translate GCoM stats into AMORA logical metrics via the gcom metric map.

    Returns ``{logical_name: {"value", "fidelity", "gcom_keys", "ncu_metric"}}``;
    entries whose required GCoM keys are absent are skipped.
    """

    from amora.probes.gcom_cuda.baseline.gcom_metrics_map import GCOM_TO_LOGICAL

    derived: dict[str, dict[str, Any]] = {}
    for entry in GCOM_TO_LOGICAL:
        value = entry.derive(stats)
        if value is None:
            continue
        derived[entry.logical] = {
            "value": value,
            "fidelity": entry.fidelity,
            "gcom_keys": list(entry.gcom_keys),
            "ncu_metric": entry.ncu_metric,
            "architecture_scope": entry.architecture_scope,
        }
    return derived
