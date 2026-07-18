"""Compile probe kernels and produce GCoM/NVBit traces.

Replicates the env mechanism used by GCoM's ``util/tracer_nvbit/run_hw_trace.py``
(``CUDA_INJECTION64_PATH`` + ``LD_PRELOAD`` of ``tracer_tool.so``, with
``TRACES_FOLDER`` as the output dir), then runs the tracer's
``post-traces-processing-compressed`` to produce the ``.traceg`` + ``kernelslist.g``
the simulator consumes. Requires a real GPU; not exercised by no-GPU tests.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from amora.backends.gcom_cuda import config as cfg


class TraceError(RuntimeError):
    """Raised when compilation or tracing of a probe kernel fails."""


def _tracer_tool_so() -> Path:
    so = cfg.TRACER_DIR / "tracer_tool" / "tracer_tool.so"
    if so.exists():
        return so
    candidates = list(cfg.TRACER_DIR.rglob("tracer_tool.so"))
    if not candidates:
        raise TraceError(f"tracer_tool.so not found under {cfg.TRACER_DIR}")
    return candidates[0]


def _trace_env(
    trace_dir: Path,
    tracer_so: Path,
    *,
    kernel_limit: int | None,
    kernel_start: int | None,
    kernel_end: int | None,
    active_from_start: bool,
) -> dict[str, str]:
    """Build the NVBit environment for an optional measured-launch selection."""

    env = dict(os.environ)
    env.pop("DYNAMIC_KERNEL_LIMIT_START", None)
    env.pop("DYNAMIC_KERNEL_LIMIT_END", None)
    env["USER_DEFINED_FOLDERS"] = "1"
    env["TRACES_FOLDER"] = str(trace_dir)
    env["CUDA_INJECTION64_PATH"] = str(tracer_so)
    env["LD_PRELOAD"] = str(tracer_so)
    env["ACTIVE_FROM_START"] = "1" if active_from_start else "0"
    if kernel_start is not None:
        env["DYNAMIC_KERNEL_LIMIT_START"] = str(kernel_start)
    if kernel_end is not None:
        env["DYNAMIC_KERNEL_LIMIT_END"] = str(kernel_end)
    else:
        env["DYNAMIC_KERNEL_LIMIT_END"] = str(kernel_limit if kernel_limit is not None else 0)
    return env


def compile_probe(
    src: Path,
    out_dir: Path,
    *,
    defines: tuple[str, ...] = (),
    extra_flags: tuple[str, ...] = ("-O2",),
    link_flags: tuple[str, ...] = (),
    timeout: int = 300,
) -> Path:
    """Compile a probe ``.cu`` to a static executable; return the binary path.

    ``defines`` (e.g. ``("AMORA_WORKING_SET_KIB=64",)``) enables sweep variants.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    label = "_".join(d.replace("=", "") for d in defines) or "default"
    binary = out_dir / f"{src.stem}__{label}"
    args = ["nvcc", "-arch", cfg.NVCC_ARCH, "-std=c++14", *extra_flags]
    for d in defines:
        args.append(f"-D{d}")
    args += [str(src), "-o", str(binary), *link_flags]
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise TraceError(f"nvcc timed out after {timeout}s for {src.name}") from exc
    if completed.returncode != 0 or not binary.exists():
        raise TraceError(
            f"nvcc failed for {src.name}: rc={completed.returncode} "
            f"stderr={(completed.stderr or '').strip()[:400]}"
        )
    return binary


def trace_probe(
    probe_id: str,
    src: Path,
    out_dir: Path,
    *,
    defines: tuple[str, ...] = (),
    argv: tuple[str, ...] = (),
    kernel_limit: int | None = None,
    kernel_start: int | None = None,
    kernel_end: int | None = None,
    active_from_start: bool = True,
    extra_flags: tuple[str, ...] = ("-O2",),
    link_flags: tuple[str, ...] = (),
    timeout: int = 1800,
) -> Path:
    """Compile + trace one probe variant; return the trace directory.

    ``timeout`` bounds the (instrumented) kernel execution; on expiry a
    ``TraceError`` is raised so callers can degrade to ``missing_stat`` rather
    than crash. The returned directory contains ``dynamic_trace.pb``.
    """

    binary = compile_probe(
        src,
        out_dir,
        defines=defines,
        extra_flags=extra_flags,
        link_flags=link_flags,
    )
    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    tracer_so = _tracer_tool_so()
    # Mirror util/tracer_nvbit/run_hw_trace.py: USER_DEFINED_FOLDERS makes the
    # tracer honor TRACES_FOLDER; dynamic start/end select one measured launch.
    env = _trace_env(
        trace_dir,
        tracer_so,
        kernel_limit=kernel_limit,
        kernel_start=kernel_start,
        kernel_end=kernel_end,
        active_from_start=active_from_start,
    )

    try:
        run = subprocess.run(
            [str(binary), *argv], env=env, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise TraceError(f"tracing {probe_id} exceeded {timeout}s timeout") from exc
    if run.returncode != 0:
        raise TraceError(
            f"tracing {probe_id} exited rc={run.returncode}: "
            f"stderr={(run.stderr or '').strip()[:400]}"
        )

    # This tracer version emits a protobuf trace (dynamic_trace.pb) directly;
    # the simulator consumes it (no kernelslist post-processing needed).
    pb = trace_dir / "dynamic_trace.pb"
    if not pb.exists():
        raise TraceError(f"no dynamic_trace.pb produced for {probe_id} in {trace_dir}")
    return trace_dir


def trace_probe_sweep(
    probe_id: str,
    src: Path,
    out_dir: Path,
    variants: list[tuple[str, tuple[str, ...], tuple[str, ...]]],
) -> list[tuple[str, Path]]:
    """Trace one variant per (label, defines, argv); return [(label, trace_dir)].

    Used by sweep / multi-kernel probes (Multi-trace sweep in the plan).
    """

    results: list[tuple[str, Path]] = []
    for label, defines, argv in variants:
        variant_dir = out_dir / label
        trace_dir = trace_probe(probe_id, src, variant_dir, defines=defines, argv=argv)
        results.append((label, trace_dir))
    return results
