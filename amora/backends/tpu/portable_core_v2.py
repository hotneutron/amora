"""Single-device JAX lowering for frozen Portable-core V2 programs."""
from __future__ import annotations

from hashlib import sha256
import os

from amora.benchmarking.portable_v2.contracts import Blocked, device_index
from benchmark_generators.portable_core.expanded.v2.execution import step as portable_step
from benchmark_generators.portable_core.expanded.v2.ops import GROUPED


class _JaxMath:
    def __init__(self, jnp, lax):
        self.xp = jnp
        self.lax = lax
        self.dtypes = {"fp32": jnp.float32, "fp16": jnp.float16, "int32": jnp.int32}

    def cast(self, value, dtype):
        return value.astype(self.dtypes[dtype])

    def array(self, value):
        return self.xp.asarray(value)

    def sum(self, x, axis, keep=False):
        return self.xp.sum(x, axis=axis, keepdims=keep)

    def amax(self, x, axis, keep=False):
        return self.xp.max(x, axis=axis, keepdims=keep)

    def max_scan(self, x):
        return self.lax.cummax(x, axis=1)

    def argsort_desc(self, x):
        return self.xp.argsort(-x, axis=1, stable=True)


def _step(node, args, types, math):
    if node.op not in GROUPED | {"half_scale_contract", "grouped_threshold_count", "affine_state_scan", "indexed_scatter_add"}:
        # The shared CPU V2 algebra is pure for these ops, including the explicit store casts.
        return portable_step(node, args, types, math)
    x = math.cast(args[0], "fp32")
    xp = math.xp
    m, n = x.shape
    op = node.op
    if op in GROUPED:
        row = xp.arange(m//4)[:, None] * 4
        col = xp.arange(n)[None, :]
        aux = math.cast(args[1], "fp32") if len(args) == 2 else None
        mixed = op in ("multi_hash_embedding_sum", "dual_hash_gather_sum")
        indices = [((row + i) * (17 if mixed and i >= 2 else 13) + col * (5 if mixed and i >= 2 else 7)) % m
                   for i in range(4)]
        vals = [(aux if op == "multi_hash_embedding_sum" and i >= 2 else x)[idx, col]
                for i, idx in enumerate(indices)]
        if op in ("weighted_embedding_bag_mean", "weighted_embedding_bag_sum"):
            weights = [aux[idx, col] for idx in indices]
            if op == "weighted_embedding_bag_mean":
                weights = [xp.abs(w) + 0.25 for w in weights]
            core = sum(v*w for v, w in zip(vals, weights))
            if op == "weighted_embedding_bag_mean":
                core /= sum(weights)
        elif op in ("indexed_group_min", "indexed_group_max"):
            core = (xp.min if op == "indexed_group_min" else xp.max)(xp.stack(vals), axis=0)
        elif op == "indexed_group_l2":
            core = xp.sqrt(sum(v*v for v in vals) + 1e-6)
        else:
            core = sum(vals)
            if op == "embedding_bag_mean":
                core /= 4
        y = xp.pad(core, ((0, m-m//4), (0, 0)))
    elif op == "half_scale_contract":
        # Do not rely on the platform's default matmul precision for FP32 working values.
        y = xp.matmul(x, xp.eye(n, dtype=x.dtype)*0.5, precision=math.lax.Precision.HIGHEST)
    elif op == "grouped_threshold_count":
        count = (x.reshape(m, n//4, 4) > 0).sum(axis=2)
        y = xp.pad(count, ((0, 0), (0, n-n//4)))
    elif op == "affine_state_scan":
        add = math.cast(args[1], "fp32")

        def carry(previous, values):
            a, b = values
            out = a*previous+b
            return out, out

        _, col_major = math.lax.scan(carry, xp.zeros(m, dtype=x.dtype), (x.T, add.T))
        y = col_major.T
    else:
        # Duplicates were rejected before staging; assignment to zero equals scatter add here.
        y = xp.zeros_like(x).at[args[1]].set(x, unique_indices=True)
    return (math.cast(y, types[0].dtype),)


def build_program(case, types, math, capture=False):
    """Build pure lowering before JIT; this function has no device ownership effects."""
    def execute(*arguments):
        values = {p.id: (arg,) for p, arg in zip(case.program.inputs, arguments)}
        for n in case.program.nodes:
            args = tuple(values[p.producer][p.port] for p in n.inputs)
            values[n.id] = _step(n, args, types[n.id], math)
        out = tuple(values[p.producer][p.port] for p in case.program.outputs)
        if capture:
            return out, tuple(values[n.id] for n in case.program.nodes)
        return out

    return execute


class Target:
    def __init__(self, device, *, expected_generation=None):
        from .capabilities import discover_in_process
        from .compatibility import runtime_identity
        from .ownership import snapshot

        index = device_index("tpu_jax", device)
        self.before = snapshot()
        if self.before["state"] != "exclusive":
            raise Blocked("ownership_unavailable", "TPU devices are busy or ownership is unverified")
        runtime = runtime_identity()
        if not all(runtime["packages"].get(k) for k in ("jax", "jaxlib", "libtpu")):
            raise Blocked("runtime_missing", "JAX, jaxlib and libtpu are required")
        if not runtime["runtime_lock_versions_verified"]:
            raise Blocked("runtime_unqualified", "TPU correctness requires a verified pinned runtime lock")
        os.environ["JAX_PLATFORMS"] = "tpu"
        try:
            caps = discover_in_process(device)
        except (ImportError, RuntimeError, ValueError) as error:
            raise Blocked("device_unavailable", str(error)) from error
        if caps.get("status") != "ok":
            raise Blocked("device_unavailable", str(caps.get("unsupported_reasons")))
        generation = caps["architecture"]["generation"]
        if generation == "unknown" or expected_generation and generation != expected_generation:
            raise Blocked("unexpected_generation", f"discovered TPU generation {generation}")
        if caps["target"]["process_count"] != 1:
            raise Blocked("unsupported_distribution", "initial adapter requires a single process")
        import jax
        import jax.numpy as jnp
        self.jax = jax
        self.device = jax.local_devices(backend="tpu")[index]
        if self.device.platform != "tpu" or self.device.id != caps["selected_device"]["id"]:
            raise Blocked("device_mismatch", "selected device is not the verified local TPU")
        self.math = _JaxMath(jnp, jax.lax)
        config = dict(caps["target"])
        config["devices"] = [{k: v for k, v in d.items() if k != "memory_stats"} for d in config["devices"]]
        config["visibility"] = {k: v for k, v in config["visibility"].items() if k != "TPU_LOG_DIR"}
        config.pop("target_config_digest", None)
        config.pop("instance_config_sha256", None)  # raw config (including log path) is separate evidence
        self.identity = {"platform": "tpu", "generation": generation, "target": config,
                         "runtime": caps["runtime"], "compiler": "jax.jit", "placement": str(self.device)}
        self.evidence = {"instance_config_sha256": caps["target"]["instance_config_sha256"], "ownership_before": self.before}

    def finish(self):
        from .ownership import snapshot
        result = snapshot(exclude_pids=(os.getpid(),))
        return {"ownership_after": result}, result["state"] == "exclusive"

    def prepare(self, case, types, raw, directory):
        import numpy as np
        jax = self.jax
        args = tuple(jax.device_put(raw[p.id], self.device) for p in case.program.inputs)
        compiled = {}
        artifacts = {}
        for label in ("output", "node_capture"):
            fn = build_program(case, types, self.math, capture=(label == "node_capture"))
            lowered = jax.jit(fn).lower(*args)
            exe = lowered.compile()
            compiled[label] = exe
            artifacts[label] = {}
            for kind, text in (("stablehlo", str(lowered.compiler_ir("stablehlo"))), ("optimized_hlo", exe.as_text())):
                if text:
                    path = directory / f"{label}-{kind}.txt"
                    with path.open("x") as stream:
                        stream.write(text)
                    artifacts[label][kind] = {"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}
                else:
                    artifacts[label][kind] = {"status": "not_available"}

        def check_placement(value):
            for leaf in jax.tree.leaves(value):
                if leaf.devices() != {self.device}:
                    raise Blocked("device_mismatch", "TPU output placement does not match selected device")

        def run():
            out = compiled["output"](*args)
            _, observed = compiled["node_capture"](*args)
            check_placement((out, observed))
            return tuple(map(np.asarray, out)), {n.id: tuple(map(np.asarray, p)) for n, p in zip(case.program.nodes, observed)}

        return run, {"ir": artifacts, "native_executable_sha256": "not_available"}
