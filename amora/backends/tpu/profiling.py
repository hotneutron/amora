"""Optional trace capture, separate from unprofiled timing."""
from __future__ import annotations

from pathlib import Path

from .runner import drain


def capture(invoke, directory):
    import jax
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        with jax.profiler.trace(str(directory), create_perfetto_trace=True):
            for _ in range(3):
                with jax.profiler.TraceAnnotation("amora_tpu_program"):
                    drain(invoke())
    except (ImportError, RuntimeError, ValueError) as error:
        return {"state": "unsupported", "reason": str(error)}
    artifacts = [str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()]
    return {"state": "observed" if artifacts else "unsupported", "artifacts": artifacts,
            "device_duration_ns": None,
            "reason": "trace captured; device interval semantics need explicit trace validation"}
