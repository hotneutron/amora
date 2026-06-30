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


def _post_processor() -> Path:
    proc = cfg.TRACER_DIR / "tracer_tool" / "traces-processing" / "post-traces-processing-compressed"
    if not proc.exists():
        candidates = list(cfg.TRACER_DIR.rglob("post-traces-processing-compressed"))
        if not candidates:
            raise TraceError("post-traces-processing-compressed not found")
        proc = candidates[0]
    return proc


def compile_probe(src: Path, out_dir: Path, *, defines: tuple[str, ...] = ()) -> Path:
    """Compile a probe ``.cu`` to a static executable; return the binary path.

    ``defines`` (e.g. ``("AMORA_WORKING_SET_KIB=64",)``) enables sweep variants.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    label = "_".join(d.replace("=", "") for d in defines) or "default"
    binary = out_dir / f"{src.stem}__{label}"
    args = ["nvcc", "-arch", cfg.NVCC_ARCH, "-std=c++14", "-O2"]
    for d in defines:
        args.append(f"-D{d}")
    args += [str(src), "-o", str(binary)]
    completed = subprocess.run(args, capture_output=True, text=True, timeout=300)
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
) -> Path:
    """Compile + trace one probe variant; return the trace directory.

    The returned directory contains the simulator-ready ``kernelslist.g`` and
    per-kernel ``.traceg`` files.
    """

    binary = compile_probe(src, out_dir, defines=defines)
    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    tracer_so = _tracer_tool_so()
    env = dict(os.environ)
    env["TRACES_FOLDER"] = str(trace_dir)
    env["CUDA_INJECTION64_PATH"] = str(tracer_so)
    env["LD_PRELOAD"] = str(tracer_so)
    if kernel_limit is not None:
        env["DYNAMIC_KERNEL_LIMIT_END"] = str(kernel_limit)

    run = subprocess.run(
        [str(binary), *argv], env=env, capture_output=True, text=True, timeout=600
    )
    if run.returncode != 0:
        raise TraceError(
            f"tracing {probe_id} exited rc={run.returncode}: "
            f"stderr={(run.stderr or '').strip()[:400]}"
        )

    kernelslist = trace_dir / "kernelslist"
    if kernelslist.exists():
        post = subprocess.run(
            [str(_post_processor()), str(kernelslist)],
            capture_output=True, text=True, timeout=600,
        )
        if post.returncode != 0:
            raise TraceError(
                f"post-processing {probe_id} failed rc={post.returncode}: "
                f"stderr={(post.stderr or '').strip()[:400]}"
            )

    if not any(trace_dir.glob("*.traceg")) and not (trace_dir / "kernelslist.g").exists():
        raise TraceError(f"no traces produced for {probe_id} in {trace_dir}")
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
