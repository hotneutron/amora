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
from amora.backends.nvidia.sass import SassExpectation, SassValidation, validate_sass


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
    sass_validation: SassValidation | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _compile_contract_hash(
    source_hash: str,
    *,
    arch: str,
    extra_flags: tuple[str, ...],
    link_flags: tuple[str, ...],
) -> str:
    payload = "\0".join((source_hash, arch, *extra_flags, "--link-flags--", *link_flags))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    link_flags: tuple[str, ...] = (),
) -> tuple[Path, str]:
    """Compile ``source`` into a host executable and return ``(path, source_sha256)``.

    The compilation is cached under a hash of source, architecture, compiler
    flags, and linker flags so benchmark contracts cannot reuse a binary built
    with incompatible libraries.
    """

    nvcc_path, _ = _ensure_cuda(capabilities)
    src_hash = _sha256(source)
    contract_hash = _compile_contract_hash(
        src_hash,
        arch=arch,
        extra_flags=extra_flags,
        link_flags=link_flags,
    )
    target_dir = build_root / source.stem / contract_hash
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
        *link_flags,
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


def build_cubin(
    source: Path,
    *,
    capabilities: NvidiaCapabilities,
    arch: str = DEFAULT_ARCH,
    build_root: Path = DEFAULT_BUILD_ROOT,
) -> tuple[Path, str]:
    """Compile ``source`` to a device-only cubin and return ``(path, source_sha256)``.

    Cached next to the host-binary cache, keyed by source SHA-256.
    """

    nvcc_path, _ = _ensure_cuda(capabilities)
    src_hash = _sha256(source)
    target_dir = build_root / source.stem / src_hash
    target_dir.mkdir(parents=True, exist_ok=True)
    cubin = target_dir / f"{source.stem}.cubin"
    if cubin.exists():
        return cubin, src_hash
    args = [nvcc_path, "-arch", arch, "-cubin", "-std=c++14", str(source), "-o", str(cubin)]
    completed = subprocess.run(args, check=False, capture_output=True, text=True, timeout=120)
    if completed.returncode != 0 or not cubin.exists():
        if cubin.exists():
            cubin.unlink()
        raise CudaUnavailable(
            f"nvcc -cubin failed for {source.name}: rc={completed.returncode} "
            f"stderr={(completed.stderr or '').strip()[:400]}"
        )
    return cubin, src_hash


def _disassembler(capabilities: NvidiaCapabilities) -> str | None:
    """Prefer cuobjdump (-sass) over nvdisasm for cubin disassembly."""

    for name in ("cuobjdump", "nvdisasm"):
        tool = capabilities.tools.get(name)
        if tool and tool.available and tool.path:
            return tool.path
    return None


def validate_kernel_sass(
    source: Path,
    expectation: SassExpectation,
    *,
    capabilities: NvidiaCapabilities,
    arch: str = DEFAULT_ARCH,
    build_root: Path = DEFAULT_BUILD_ROOT,
) -> SassValidation | None:
    """Build a cubin, disassemble it, and validate against ``expectation``.

    Returns ``None`` (never raises) when the toolchain cannot produce SASS, so a
    missing disassembler degrades gracefully instead of failing the probe.
    """

    disasm = _disassembler(capabilities)
    if disasm is None:
        return None
    try:
        cubin, _ = build_cubin(
            source, capabilities=capabilities, arch=arch, build_root=build_root
        )
        flag = "-sass" if disasm.endswith("cuobjdump") else "-c"
        completed = subprocess.run(
            [disasm, flag, str(cubin)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return None
        sass_text = completed.stdout
        disassembly_hash = hashlib.sha256(sass_text.encode("utf-8")).hexdigest()
        return validate_sass(sass_text, expectation, disassembly_hash=disassembly_hash)
    except (CudaUnavailable, subprocess.SubprocessError, OSError):
        return None


def run_kernel(
    source: Path,
    *,
    capabilities: NvidiaCapabilities,
    args: tuple[str, ...] = (),
    timeout: int = 30,
    arch: str = DEFAULT_ARCH,
    build_root: Path = DEFAULT_BUILD_ROOT,
    expectation: SassExpectation | None = None,
) -> CudaRunResult:
    """Build (if needed) and execute the CUDA driver compiled from ``source``.

    The host driver is expected to print a single JSON document to stdout. When
    ``expectation`` is provided, the kernel's SASS is validated (best effort) and
    attached to the result; SASS-tooling failures do not abort the run.
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
    sass = None
    if expectation is not None:
        sass = validate_kernel_sass(
            source, expectation, capabilities=capabilities, arch=arch, build_root=build_root
        )
    return CudaRunResult(
        binary_path=binary,
        binary_sha256=binary_sha,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
        payload=payload,
        sass_validation=sass,
    )


def cleanup_build_root(build_root: Path = DEFAULT_BUILD_ROOT) -> None:
    """Remove cached build artifacts; primarily useful for tests."""

    if build_root.exists():
        shutil.rmtree(build_root)
