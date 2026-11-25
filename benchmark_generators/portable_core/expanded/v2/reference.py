# SPDX-License-Identifier: Apache-2.0
"""Independent FP64/scalar reference with explicit rounding at storage boundaries."""
from __future__ import annotations

import math

from .ops import DISCRETE, GROUPED


def _indexed_quartet(op, x, aux):
    import numpy as np

    r, c = x.shape
    count = r // 4
    columns = np.broadcast_to(np.arange(c), (count, c)).reshape(-1)
    start = np.repeat(np.arange(count), c)*4
    lane = np.arange(4)[:, None]
    alt = (lane >= 2) & (op in ("multi_hash_embedding_sum", "dual_hash_gather_sum"))
    indices = ((start+lane)*np.where(alt, 17, 13) + columns*np.where(alt, 5, 7)) % r
    samples = x[indices, columns]
    if op == "multi_hash_embedding_sum":
        samples[2:] = aux[indices[2:], columns]
    if op == "weighted_embedding_bag_mean":
        weights = np.abs(aux[indices, columns]) + 0.25
        y = np.average(samples, axis=0, weights=weights)
    elif op == "weighted_embedding_bag_sum":
        y = np.einsum("ki,ki->i", samples, aux[indices, columns])
    elif op in ("indexed_group_min", "indexed_group_max"):
        y = (np.min if op == "indexed_group_min" else np.max)(samples, axis=0)
    elif op == "indexed_group_l2":
        y = np.hypot(np.linalg.norm(samples, axis=0), 0.001)
    elif op == "embedding_bag_mean":
        y = samples.mean(axis=0)
    else:
        y = samples.sum(axis=0)
    return np.pad(y.reshape(count, c), ((0, r-count), (0, 0)))


def _step(op, inputs, params):
    import numpy as np

    x = np.asarray(inputs[0], dtype=np.float64)
    y = np.asarray(inputs[1], dtype=np.float64) if len(inputs) == 2 else None
    if op in GROUPED:
        return (_indexed_quartet(op, x, y),)
    if op == "cast":
        return (x,)
    if op == "add":
        return (np.add(x, y),)
    if op == "mul":
        return (np.multiply(x, y),)
    if op == "relu":
        return (np.clip(x, 0, None),)
    r, c = x.shape
    if op == "row_roll":
        return (np.concatenate((x[-1:], x[:-1])),)
    if op == "half_scale_contract":
        # The original fixture right-multiplies by 0.5 times the identity.
        return (np.ldexp(x, -1),)
    if op == "row_stat_broadcast":
        stat = x.mean(1) if params["stat"] == "mean" else x.max(1)
        stored = stat.astype(np.float16 if params["temporary_dtype"] == "fp16" else np.float32)
        return (np.repeat(stored[:, None], c, axis=1),)
    if op == "rotary_pair_transform":
        pair = x.reshape(r, c//2, 2)
        angle = y.reshape(r, c//2, 2)
        out = np.empty_like(pair)
        out[..., 0] = pair[..., 0]*np.cos(angle[..., 0]) - pair[..., 1]*np.sin(angle[..., 0])
        out[..., 1] = pair[..., 0]*np.cos(angle[..., 1]) + pair[..., 1]*np.sin(angle[..., 1])
        return (out.reshape(x.shape),)
    if op == "row_logsumexp":
        return (np.repeat(np.logaddexp.reduce(x, 1)[:, None], c, axis=1),)
    if op == "cumulative_max_scan":
        # Explicit prefix recurrence; intentionally not the backend's cummax implementation.
        out = x.copy()
        for i in range(1, c):
            out[:, i] = np.where(out[:, i-1] > x[:, i], out[:, i-1], x[:, i])
        return (out,)
    if op in ("bilinear_resample", "weighted_stencil"):
        a = np.pad(x, 1, mode="edge")
        offsets = ((0, 0, .375), (0, 1, .125), (1, 0, .375), (1, 1, .125)) if op == "bilinear_resample" else (
            (0, 0, .5), (-1, 0, .125), (1, 0, .125), (0, -1, .125), (0, 1, .125))
        return (sum(gain*a[1+i:1+i+r, 1+j:1+j+c] for i, j, gain in offsets),)
    if op == "grouped_threshold_count":
        counts = sum((x[:, i::4] > 0).astype(np.int32) for i in range(4))
        return (np.pad(counts, ((0, 0), (0, c-c//4))),)
    if op in ("scaled_round_clip", "dequantize_bias_gelu"):
        z = np.clip(x * 32, -127, 127)
        value = np.floor(z)
        u = (value + (z-value >= .5))/32
        if op == "scaled_round_clip":
            return (u,)
        u = u + y
        return (u / (1 + np.exp(-2*math.sqrt(2/math.pi)*u*(1 + .044715*u**2))),)
    if op == "affine_state_scan":
        carry = np.zeros(r)
        out = np.empty_like(x)
        for i, (a, b) in enumerate(zip(x.T, y.T)):
            carry = np.add(np.multiply(a, carry), b)
            out[:, i] = carry
        return (out,)
    if op == "indexed_embedding_sum":
        return (sum(np.take(x, inputs[1][:, i], axis=0) for i in range(4)),)
    if op == "indexed_scatter_add":
        idx = inputs[1]
        return (np.stack([x[idx == i].sum(0) for i in range(r)]),)
    if op == "segment_sum4":
        return (sum(x[:, i::4] for i in range(4)),)
    if op == "silu_gated_product":
        return ((x / (1+np.exp(-x)))*y,)
    if op == "causal_masked_softmax":
        out = np.zeros_like(x)
        for i, row in enumerate(x):
            allowed = row[:i+1]
            out[i, :i+1] = np.exp(allowed-np.logaddexp.reduce(allowed))
        return (out,)
    if op == "row_variance_normalize":
        return ((x - np.mean(x, axis=1, keepdims=True)) / np.sqrt(np.var(x, axis=1, keepdims=True)+1e-5),)
    if op == "row_topk_select":
        indices = np.array([sorted(range(c), key=lambda i: (-row[i], i))[:params["k"]] for row in x])
        return np.take_along_axis(x, indices, axis=1), indices
    raise ValueError(op)


def oracle(case, inputs, *, capture=False):
    import numpy as np

    store = {"fp16": np.float16, "fp32": np.float32, "int32": np.int32}
    outputs = case.program.types(dict(case.shape))
    values = {k: (v,) for k, v in inputs.items()}
    for node in case.program.nodes:
        args = [values[r.producer][r.port] for r in node.inputs]
        want = _step(node.op, args, dict(node.attributes))
        types = outputs[node.id]
        if len(want) != len(types):
            raise ValueError("reference arity mismatch")
        values[node.id] = tuple(np.asarray(v, dtype=store[t.dtype]) for v, t in zip(want, types))
    result = tuple(values[r.producer][r.port] for r in case.program.outputs)
    return (result, values) if capture else result


def check(prepared, raw):
    from .execution import execute_prepared
    import numpy as np

    _, actual = execute_prepared(prepared, capture=True)
    _, expected = oracle(prepared.case, raw, capture=True)
    for node in prepared.case.program.nodes:
        if len(actual[node.id]) != len(expected[node.id]):
            return False, f"arity at {node.id}"
        for a, b in zip(actual[node.id], expected[node.id]):
            got = prepared.backend.numpy(a)
            if got.shape != b.shape or got.dtype != b.dtype or not np.isfinite(got).all() or not np.isfinite(b).all():
                return False, f"shape, dtype or finiteness at {node.id}"
            if b.dtype.kind == "i" or node.op in DISCRETE:
                ok = np.array_equal(got, b)
            else:
                # Applied to each rounded boundary of the portable realization, not the archival oracle.
                atol, rtol = (0.001, 0.005) if b.dtype == np.float16 else (2e-5, 2e-4)
                ok = np.allclose(got, b, atol=atol, rtol=rtol)
            if not ok:
                return False, f"oracle mismatch at {node.id} ({node.op})"
    return True, None
