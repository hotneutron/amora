"""Shared host-side validation for the correctness-only accelerator adapters."""
from __future__ import annotations

from hashlib import sha256
from math import prod
import re

from benchmark_generators.portable_core.expanded.v2.ops import DISCRETE, GROUPED
from benchmark_generators.portable_core.expanded.v2.spec import ORACLE, RECIPE, digest

PROTOCOL = "portable-v2-device-correctness-v1"
MODE = {"cuda_torch": "torch-eager", "tpu_jax": "single-device-jax-jit"}
PLATFORMS = {"cuda_torch": "cuda", "tpu_jax": "tpu"}
MAX_CASES = 32
MAX_PROGRAM_BYTES = 256 * 2**20

# Explicit allowlist: never inherit new CPU operations without a backend decision.
OPS = GROUPED | frozenset({
    "cast", "add", "mul", "relu", "row_roll", "half_scale_contract", "row_stat_broadcast",
    "rotary_pair_transform", "row_logsumexp", "cumulative_max_scan", "bilinear_resample",
    "weighted_stencil", "grouped_threshold_count", "dequantize_bias_gelu", "affine_state_scan",
    "indexed_embedding_sum", "indexed_scatter_add", "segment_sum4", "silu_gated_product",
    "scaled_round_clip", "causal_masked_softmax", "row_variance_normalize", "row_topk_select",
})


class Blocked(RuntimeError):
    def __init__(self, reason, message):
        self.reason = reason
        super().__init__(message)


def device_index(backend, device):
    prefix = PLATFORMS.get(backend)
    match = re.fullmatch(r"([a-z]+):([0-9]{1,4})", device)
    if prefix is None or not match or match.group(1) != prefix:
        raise ValueError(f"{backend} requires an explicit {prefix or 'supported'}:<local-index> device")
    return int(match.group(2))


def inspect():
    return {"protocol": PROTOCOL, "hardware_opened": False, "backends": {
        k: {"mode": v, "device": PLATFORMS[k] + ":<local-index>", "operators_declared": sorted(OPS),
            "packed_only": True, "scatter": "unique destinations only", "native_status": "not_checked"}
        for k, v in MODE.items()}, "cpu_fallback": False, "native_performance_qualified": False}


def prepare_case(case, raw=None):
    """Bind the CPU oracle and physical input contract before any device transfer."""
    try:
        import numpy as np
        from benchmark_generators.portable_core.expanded.v2.execution import Backend, make_inputs, prepare
    except ImportError as error:
        raise Blocked("runtime_missing", "NumPy is required for the independent V2 reference") from error

    unsupported = sorted({n.op for n in case.program.nodes} - OPS)
    if unsupported:
        raise Blocked("unsupported_operation", ", ".join(unsupported))
    if any(p.row_padding or p.offset for p in case.program.inputs):
        raise Blocked("unsupported_layout", "physical row stride/offset is not admitted by the packed adapters")
    types = case.program.types(dict(case.shape))
    # Bound capture storage and the largest explicit temporary (e.g. half-scale identity).
    bound = sum(4 * prod(t.shape) for ports in types.values() for t in ports)
    for node in case.program.nodes:
        if node.op == "half_scale_contract":
            bound += 4 * types[node.id][0].shape[-1] ** 2
    if bound > MAX_PROGRAM_BYTES:
        raise Blocked("input_budget", "capture and explicit temporary budget exceeded")
    raw = make_inputs(case) if raw is None else raw
    # Existing strict CPU prepare validates dtypes, layout, aliasing, and bounds. It does not repack.
    cpu = prepare(case, Backend("numpy"), raw)
    for node in case.program.nodes:
        if node.op == "indexed_scatter_add":
            idx = raw[node.inputs[1].producer]
            if np.unique(idx).size != idx.size:
                raise Blocked("unsupported_duplicate_indices", "scatter destinations must be unique on this route")
    inputs = [{"role": inp.role, "dtype": str(raw[inp.id].dtype), "shape": list(raw[inp.id].shape),
               "byte_strides": list(raw[inp.id].strides), "offset": 0,
               "sha256": sha256(raw[inp.id].tobytes(order="C")).hexdigest()} for inp in case.program.inputs]
    return cpu.types, raw, {"input_recipe": RECIPE, "inputs": inputs, "input_sha256": digest(inputs), "oracle": ORACLE}


def compare(values, expected, exact):
    import numpy as np

    if len(values) != len(expected):
        return "arity mismatch"
    for i, (got, want) in enumerate(zip(values, expected)):
        if got.shape != want.shape or got.dtype != want.dtype or not np.isfinite(got).all() or not np.isfinite(want).all():
            return f"shape, dtype or finiteness mismatch on port {i}"
        # Match the pinned CPU V2 oracle thresholds; integer/discrete output is exact.
        atol, rtol = (0.001, 0.005) if want.dtype == np.float16 else (2e-5, 2e-4)
        ok = (np.array_equal(got, want) if want.dtype.kind == "i" or exact[i]
              else np.allclose(got, want, atol=atol, rtol=rtol))
        if not ok:
            return f"oracle value mismatch on port {i}"
    return None


def check_case(target, case, types, raw, directory):
    from benchmark_generators.portable_core.expanded.v2.reference import oracle

    prepared, compiler = target.prepare(case, types, raw, directory)
    got, captures = prepared()
    want, reference = oracle(case, raw, capture=True)
    if set(captures) != {n.id for n in case.program.nodes}:
        return "missing or unexpected node outputs", compiler
    for node in case.program.nodes:
        ports = captures[node.id]
        error = compare(ports, reference[node.id], [node.op in DISCRETE] * len(ports))
        if error:
            return f"{node.id}: {error}", compiler
    ops = {n.id: n.op for n in case.program.nodes}
    return compare(got, want, [ops.get(r.producer) in DISCRETE for r in case.program.outputs]), compiler
