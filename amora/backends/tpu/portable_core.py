# SPDX-License-Identifier: Apache-2.0
"""Pure JAX node-graph lowering of canonical Apache-2.0 portable workloads.

Mathematical contracts remain in benchmark_generators.portable_core. This bridge
reads their graph directly; it does not fork the workload generator or oracle.
"""
from __future__ import annotations

from .artifacts import digest, file_hash


def verify_semantics():
    import json
    from pathlib import Path
    from benchmark_generators import portable_core as core
    from benchmark_generators.portable_core.base.core import digest as payload_digest
    path = Path(core.__file__).parent / "SEMANTIC_FINGERPRINTS.json"
    expected = json.loads(path.read_text())["fingerprints"]
    rows = []
    for tier in (20, 24, 28):
        for profile in ("smoke", "native"):
            cases = core.generate_cases(tier, profile)
            item = expected[f"{tier}/{profile}"]
            actual = payload_digest(cases)
            rows.append({"tier": tier, "profile": profile, "cases": len(cases),
                         "expected_sha256": item["case_payload_sha256"], "actual_sha256": actual,
                         "passed": actual == item["case_payload_sha256"] and len(cases) == item["cases"]})
    return {"fingerprint_file_sha256": file_hash(path), "results": rows,
            "passed": all(r["passed"] for r in rows), "bridge_source_sha256": file_hash(__file__)}


def prepare(case, device, raw=None):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from benchmark_generators import portable_core as core
    from benchmark_generators.portable_core.expanded.core import prepare as validate
    raw = core.make_inputs(case) if raw is None else raw
    # Canonical preparation performs all dtype/range/offset/domain checks on CPU.
    validate(case, core.create_backend("numpy", "cpu"), raw)
    dtypes = {"fp32": jnp.float32, "bf16": jnp.bfloat16, "int32": jnp.int32}
    arguments = tuple(jax.device_put(np.asarray(raw[name]), device).astype(dtypes[spec["dtype"]])
                      for name, spec in case["inputs"].items())
    from .runner import drain
    drain(arguments)
    expected = core.oracle(case, raw)
    input_identity = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                             "sha256": __import__("hashlib").sha256(value.tobytes()).hexdigest()}
                      for name, value in raw.items()}
    return arguments, expected, digest(input_identity)


def program(case):
    import jax
    import jax.numpy as jnp
    names = tuple(case["inputs"])
    dtypes = {"fp32": jnp.float32, "bf16": jnp.bfloat16, "int32": jnp.int32}

    def execute(*arguments):
        values = dict(zip(names, arguments))
        for node in case["nodes"]:
            args = [values[name] for name in node["inputs"]]
            x = args[0]
            op, attrs = node["op"], node["attrs"]
            if op == "reshape": out = x.reshape(tuple(node["shape"]))
            elif op == "cast": out = x.astype(dtypes[attrs["dtype"]])
            elif op == "broadcast":
                axis = attrs.get("axis")
                vector = x if axis is None else x[:, None] if axis == 0 else x[None, :]
                out = jnp.broadcast_to(vector, tuple(attrs["shape"]))
            elif op == "transpose": out = x.T
            elif op == "gather": out = jnp.take(x, args[1], axis=0)
            elif op == "matmul":
                out = jax.lax.dot_general(x.astype(jnp.bfloat16), args[1].astype(jnp.bfloat16),
                    (((1,), (0,)), ((), ())), precision=jax.lax.Precision.HIGHEST,
                    preferred_element_type=jnp.float32)
            elif op == "scan": out = jnp.cumsum(x, axis=attrs["axis"], dtype=jnp.float32)
            elif op == "segment_sum":
                offsets = attrs["offsets"]
                out = jnp.stack([jnp.sum(x[a:b], axis=0, dtype=jnp.float32)
                                 for a, b in zip(offsets, offsets[1:])])
            elif op == "scatter_add":
                out = jnp.zeros((attrs["rows"], x.shape[1]), jnp.float32).at[args[1]].add(x)
            elif op == "stencil":
                padded = jnp.pad(x, ((1, 1), (1, 1), (0, 0)))
                out = jnp.zeros_like(x)
                for i in range(3):
                    for k in range(3):
                        out = out + padded[i:i+x.shape[0], k:k+x.shape[1]] * args[1][:, i, k]
            else:
                f = [arg.astype(jnp.float32) for arg in args]
                a = f[0]
                if op in ("add", "sub", "mul", "div"):
                    out = {"add": jnp.add, "sub": jnp.subtract, "mul": jnp.multiply, "div": jnp.divide}[op](*f)
                elif op == "square": out = jnp.square(a)
                elif op == "relu": out = jnp.maximum(a, jnp.float32(0))
                elif op == "exp": out = jnp.exp(a)
                elif op == "rsqrt": out = jax.lax.rsqrt(a)
                elif op == "silu": out = a / (jnp.float32(1) + jnp.exp(-a))
                elif op == "sum": out = jnp.sum(a, axis=attrs["axis"], dtype=jnp.float32)
                elif op == "max": out = jnp.max(a, axis=attrs["axis"])
                else: raise ValueError(f"unimplemented opcode: {op}")
            if tuple(out.shape) != tuple(node["shape"]):
                raise ValueError(f"compiled graph shape mismatch: {node['id']}")
            values[node["id"]] = out
        return tuple(values[name] for name in case.get("outputs", [case["output"]]))
    return execute


def check_outputs(actual, expected, *, matrix=False, output_dtypes=None):
    import numpy as np
    atol, rtol = (5e-4, 5e-3) if matrix else (2e-5, 2e-4)
    rows = []
    if len(actual) != len(expected):
        return {"passed": False, "reason": "output_count_mismatch"}
    for index, (got, want) in enumerate(zip(actual, expected)):
        dtype = str(got.dtype)
        wanted_dtype = output_dtypes[index] if output_dtypes else "float32"
        wanted_dtype = {"fp32": "float32", "bf16": "bfloat16", "int32": "int32"}.get(wanted_dtype, wanted_dtype)
        got = np.asarray(got, dtype=np.float32)
        shape = got.shape == want.shape
        passed = shape and dtype == wanted_dtype and np.all(np.isfinite(got)) and np.allclose(
            got, want, atol=atol, rtol=rtol)
        rows.append({"passed": bool(passed), "shape": list(got.shape), "dtype": dtype,
                     "max_abs_error": float(np.max(np.abs(got-want))) if shape and got.size else 0.0})
    return {"passed": all(r["passed"] for r in rows), "outputs": rows,
            "tolerance_policy": "portable-core-r1", "atol": atol, "rtol": rtol}
