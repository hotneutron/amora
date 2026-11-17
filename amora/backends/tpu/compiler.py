"""Compile before timing; HLO is recorded as IR, never as ISA disassembly."""
from __future__ import annotations

from pathlib import Path
import time

from .artifacts import file_hash


def compile_program(function, arguments, directory):
    import jax
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter_ns()
    lowered = jax.jit(function).lower(*arguments)
    executable = lowered.compile()
    elapsed = time.perf_counter_ns() - start
    artifacts = {}
    for kind, text in (("stablehlo", str(lowered.compiler_ir("stablehlo"))),
                       ("optimized_hlo", executable.as_text())):
        if text:
            path = directory / (kind + ".txt")
            path.write_text(text)
            artifacts[kind] = {"path": str(path.name), "sha256": file_hash(path)}
    return executable, {"compile_ns": elapsed, "compiler_artifacts": artifacts,
                        "dynamic_argument_count": len(arguments)}
