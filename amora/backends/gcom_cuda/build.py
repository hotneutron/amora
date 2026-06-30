"""One-time build helpers for the GCoM tracer and simulator.

Both are idempotent: if the artifact already exists the build is skipped. These
shell out to the GCoM checkout's own build scripts (we do not modify GCoM). They
require a Linux host with CUDA + a compiler and are not exercised by the no-GPU
test suite.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from amora.backends.gcom_cuda import config as cfg


class GcomBuildError(RuntimeError):
    """Raised when a GCoM build step fails."""


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("CUDA_INSTALL_PATH", "/usr/local/cuda")
    return env


def _run(cmd: list[str] | str, *, cwd: Path, shell: bool = False, timeout: int = 3600) -> None:
    completed = subprocess.run(
        cmd, cwd=str(cwd), env=_env(), shell=shell,
        capture_output=True, text=True, timeout=timeout,
    )
    if completed.returncode != 0:
        raise GcomBuildError(
            f"command failed (rc={completed.returncode}): {cmd}\n"
            f"stderr: {(completed.stderr or '').strip()[:800]}"
        )


def tracer_so_paths() -> list[Path]:
    if not cfg.TRACER_DIR.exists():
        return []
    return list(cfg.TRACER_DIR.rglob("*.so"))


def ensure_tracer_built() -> Path:
    """Build the NVBit tracer if its .so is not present; return its directory."""

    if tracer_so_paths():
        return cfg.TRACER_DIR
    if not cfg.TRACER_DIR.exists():
        raise GcomBuildError(f"tracer dir missing: {cfg.TRACER_DIR}")
    install = cfg.TRACER_DIR / "install_nvbit.sh"
    if install.exists():
        _run(["bash", str(install)], cwd=cfg.TRACER_DIR)
    _run(["make", "-C", str(cfg.TRACER_DIR)], cwd=cfg.GCOM_ROOT)
    if not tracer_so_paths():
        raise GcomBuildError("tracer build completed but no .so produced")
    return cfg.TRACER_DIR


def ensure_sim_built() -> Path:
    """Build accel-sim.out if absent; return the simulator binary path."""

    if cfg.SIM_BIN.exists():
        return cfg.SIM_BIN
    if not cfg.SIM_SETUP_SCRIPT.exists():
        raise GcomBuildError(f"simulator setup script missing: {cfg.SIM_SETUP_SCRIPT}")
    nproc = os.cpu_count() or 4
    script = (
        f"source {cfg.SIM_SETUP_SCRIPT} && "
        f"make -j{nproc} -C {cfg.GCOM_ROOT / 'gpu-simulator'}"
    )
    _run(["bash", "-c", script], cwd=cfg.GCOM_ROOT)
    if not cfg.SIM_BIN.exists():
        raise GcomBuildError(f"simulator build completed but binary missing: {cfg.SIM_BIN}")
    return cfg.SIM_BIN
