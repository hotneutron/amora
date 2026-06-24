"""Reusable build + launch helper for AMORA NVIDIA baseline kernels.

The baseline cutline does not require Python CUDA bindings. Each probe ships a
small `.cu` source containing both the device kernel and a host driver that
runs the kernel and prints a single JSON line to stdout. This module wraps
`nvcc` to compile that source into a host executable, executes it with a
configurable timeout, parses the JSON, and returns it.

Build artifacts are cached under ``out/build/nvidia/baseline/<probe>/`` keyed
by the source SHA-256 so repeated probe runs avoid recompilation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities


DEFAULT_BUILD_ROOT = Path(
    os.environ.get(
        "AMORA_BUILD_ROOT",
        os.path.join(os.path.expanduser("~"), ".cache", "amora", "build", "nvidia", "baseline"),
    )
)
DEFAULT_ARCH = os.environ.get("AMORA_NVCC_ARCH", "sm_80")


class CudaUnavailable(RuntimeError):
    """Raised when the host cannot build or launch a CUDA driver."""


@dataclass(frozen=True)
class CudaRunResult:
    binary_path: Path
    binary_sha256: str
    stdout: str
    stderr: str
    returncode: int
    payload: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _ensure_cuda(capabilities: NvidiaCapabilities) -> tuple[str, str]:
    """Return (nvcc_path, nvidia_smi_path) or raise CudaUnavailable."""

    nvcc_tool = capabilities.tools.get("nvcc")
    if not nvcc_tool or not nvcc_tool.available or not nvcc_tool.path:
        raise CudaUnavailable("nvcc is not available on PATH")
    if not capabilities.gpu_available or not capabilities.devices:
        raise CudaUnavailable("no CUDA-capable GPU reported by nvidia-smi")
    smi_tool = capabilities.tools.get("nvidia-smi")
    smi_path = smi_tool.path if smi_tool and smi_tool.available else "nvidia-smi"
    return nvcc_tool.path, smi_path


def build_executable(
    source: Path,
    *,
    capabilities: NvidiaCapabilities,
    arch: str = DEFAULT_ARCH,
    build_root: Path = DEFAULT_BUILD_ROOT,
    extra_flags: tuple[str, ...] = ("-O2",),
) -> tuple[Path, str]:
    """Compile ``source`` into a host executable and return ``(path, source_sha256)``.

    The compilation is cached: if a binary already exists under
    ``build_root/<source.stem>/<source_sha256>`` it is reused.
    """

    nvcc_path, _ = _ensure_cuda(capabilities)
    src_hash = _sha256(source)
    target_dir = build_root / source.stem / src_hash
    target_dir.mkdir(parents=True, exist_ok=True)
    binary = target_dir / source.stem
    if binary.exists():
        return binary, src_hash
    args = [
        nvcc_path,
        "-arch",
        arch,
        "-std=c++14",
        *extra_flags,
        str(source),
        "-o",
        str(binary),
    ]
    completed = subprocess.run(args, check=False, capture_output=True, text=True, timeout=120)
    if completed.returncode != 0 or not binary.exists():
        # Clean up so future calls retry instead of returning a stale path.
        if binary.exists():
            binary.unlink()
        raise CudaUnavailable(
            f"nvcc failed for {source.name}: rc={completed.returncode} "
            f"stderr={(completed.stderr or '').strip()[:400]}"
        )
    return binary, src_hash


def run_kernel(
    source: Path,
    *,
    capabilities: NvidiaCapabilities,
    args: tuple[str, ...] = (),
    timeout: int = 30,
    arch: str = DEFAULT_ARCH,
    build_root: Path = DEFAULT_BUILD_ROOT,
) -> CudaRunResult:
    """Build (if needed) and execute the CUDA driver compiled from ``source``.

    The host driver is expected to print a single JSON document to stdout.
    """

    binary, _ = build_executable(
        source,
        capabilities=capabilities,
        arch=arch,
        build_root=build_root,
    )
    completed = subprocess.run(
        [str(binary), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    binary_sha = _sha256(binary)
    if completed.returncode != 0:
        raise CudaUnavailable(
            f"{source.stem} runtime exited with rc={completed.returncode}: "
            f"stderr={(completed.stderr or '').strip()[:400]}"
        )
    payload_str = (completed.stdout or "").strip()
    if not payload_str:
        raise CudaUnavailable(f"{source.stem} produced no stdout payload")
    try:
        # Some drivers may print log lines before the JSON; keep the last JSON object.
        payload = json.loads(payload_str.splitlines()[-1])
    except json.JSONDecodeError as exc:
        raise CudaUnavailable(
            f"{source.stem} stdout is not valid JSON: {exc!s}; raw={payload_str[:200]!r}"
        ) from exc
    return CudaRunResult(
        binary_path=binary,
        binary_sha256=binary_sha,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
        payload=payload,
    )


def cleanup_build_root(build_root: Path = DEFAULT_BUILD_ROOT) -> None:
    """Remove cached build artifacts; primarily useful for tests."""

    if build_root.exists():
        shutil.rmtree(build_root)
