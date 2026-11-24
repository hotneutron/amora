# SPDX-License-Identifier: Apache-2.0
"""Prepared CPU realization with explicit working dtypes and store points."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .ops import GROUPED, MODULAR
from .spec import REALIZATION, digest


class BackendUnavailable(RuntimeError):
    pass


class Backend:
    def __init__(self, name="numpy", device="cpu"):
        if device != "cpu" or name not in ("numpy", "torch"):
            raise BackendUnavailable(f"v2 backend {name}:{device} is not qualified")
        try:
            import numpy as np
            self.np = np
            if name == "torch":
                import torch
                self.xp = torch
            else:
                self.xp = np
        except ImportError as err:
            raise BackendUnavailable(f"missing CPU dependency: {err.name}") from err
        self.name = name
        self.device = device
        self.dtypes = {"fp16": self.xp.float16, "fp32": self.xp.float32, "int32": self.xp.int32}

    def cast(self, value, dtype):
        return value.to(self.dtypes[dtype]) if self.name == "torch" else value.astype(self.dtypes[dtype], copy=False)

    def from_numpy(self, value):
        return self.xp.from_numpy(value) if self.name == "torch" else value

    def numpy(self, value):
        return value.detach().numpy() if self.name == "torch" else value

    def sum(self, x, axis, keep=False):
        return self.xp.sum(x, dim=axis, keepdim=keep) if self.name == "torch" else x.sum(axis=axis, keepdims=keep)

    def amax(self, x, axis, keep=False):
        return self.xp.amax(x, dim=axis, keepdim=keep) if self.name == "torch" else x.max(axis=axis, keepdims=keep)

    def array(self, value):
        return self.xp.as_tensor(value) if self.name == "torch" else self.np.asarray(value)

    def max_scan(self, x):
        if self.name == "torch":
            return self.xp.cummax(x, dim=1).values
        return self.xp.maximum.accumulate(x, axis=1)

    def argsort_desc(self, x):
        if self.name == "torch":
            return self.xp.argsort(x, dim=1, descending=True, stable=True)
        return self.np.argsort(-x, axis=1, kind="stable")


@dataclass(frozen=True)
class Prepared:
    case: object
    backend: Backend
    types: dict
    inputs: dict
    realization_sha256: str


def make_inputs(case):
    import numpy as np

    rng = np.random.Generator(np.random.PCG64(case.seed))
    result = {}
    dims = dict(case.shape)
    for item in case.program.inputs:
        tensor = item.tensor.bind(dims)
        shape = tensor.shape
        if math.prod(shape) > 2**24:
            raise ValueError("input exceeds the portable slice budget")
        if tensor.dtype == "int32":
            if "M" not in dims:
                raise ValueError("indexed fixture requires M")
            size = math.prod(shape)
            rows = dims["M"]
            if item.domain == "distinct":
                vals = (np.arange(size) % rows).astype(np.int32)
            elif item.domain == "ramp":
                vals = ((np.arange(size) * 13 + 7) % max(rows // 4, 1)).astype(np.int32)
            elif item.domain == "positive":
                vals = rng.integers(0, max(rows // 8, 1), size, dtype=np.int32)
            else:
                vals = rng.integers(0, rows, size, dtype=np.int32)
            vals = vals.reshape(shape)
        else:
            lo, hi, scale = (16, 48, 64) if item.domain == "positive" else (-12, 13, 128)
            if item.domain == "signed":
                lo, hi, scale = -63, 64, 32
            vals = rng.integers(lo, hi, shape, dtype=np.int32) / scale
            vals = vals.astype(np.float16 if tensor.dtype == "fp16" else np.float32)
        if item.row_padding or item.offset:
            width = shape[-1] + item.row_padding
            size = (shape[0] * width if len(shape) == 2 else math.prod(shape)) + item.offset
            if size > 2**24:
                raise ValueError("physical input exceeds the portable slice budget")
            backing = np.zeros(size, dtype=vals.dtype)
            strides = (width * vals.itemsize, vals.itemsize) if len(shape) == 2 else None
            view = np.ndarray(shape, dtype=vals.dtype, buffer=backing, offset=item.offset * vals.itemsize,
                              strides=strides)
            view[...] = vals
            vals = view
        result[item.id] = vals
    return result


def prepare(case, backend, raw=None):
    np = backend.np
    raw = make_inputs(case) if raw is None else raw
    types = case.program.types(dict(case.shape))
    if set(raw) != {entry.id for entry in case.program.inputs}:
        raise ValueError("input ports do not match case")
    dtype = {"fp16": np.dtype("float16"), "fp32": np.dtype("float32"), "int32": np.dtype("int32")}
    for item in case.program.inputs:
        array = raw[item.id]
        tensor = types[item.id][0]
        if not isinstance(array, np.ndarray) or array.shape != tensor.shape or array.dtype != dtype[tensor.dtype]:
            raise ValueError(f"invalid tensor shape or dtype at {item.role}")
        if not array.flags.writeable or not np.all(np.isfinite(array)):
            raise ValueError("non-finite or read-only inputs are not supported")
        expected = tuple(math.prod(tensor.shape[i+1:]) * array.itemsize for i in range(len(tensor.shape)))
        if item.row_padding:
            expected = ((tensor.shape[1] + item.row_padding) * array.itemsize, array.itemsize)
        base = array
        while isinstance(base.base, np.ndarray):
            base = base.base
        if array.strides != expected or not base.flags.c_contiguous or (
                array.ctypes.data - base.ctypes.data != item.offset * array.itemsize):
            raise ValueError(f"undeclared strides or offset at {item.role}")
    for node in case.program.nodes:
        if node.op in MODULAR:
            ref = node.inputs[1]
            if ref.producer not in raw or ref.port:
                raise ValueError("indexed controls require prepared external indices")
            index = raw[ref.producer]
            rows = types[node.inputs[0].producer][node.inputs[0].port].shape[0]
            if (index < 0).any() or (index >= rows).any():
                raise ValueError("index outside the declared table/destination domain")
    # The alias class is part of the declared contract. Fork/join sharing uses graph ports,
    # not accidental aliasing between otherwise independent external bindings.
    arrays = list(raw.values())
    if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[:i]):
        raise ValueError("undeclared aliased external inputs")
    values = {k: (backend.from_numpy(v),) for k, v in raw.items()}
    receipt = {"case": case.case_sha256, "backend": backend.name, "device": backend.device,
               "runtime": str(backend.xp.__version__), "realization": REALIZATION, "protocol": "cpu_correctness_only"}
    return Prepared(case, backend, types, values, digest(receipt))


def _grouped(op, v, w, backend):
    xp = backend.xp
    m, n = v.shape
    rows = xp.arange(m // 4)[:, None] * 4
    cols = xp.arange(n)[None, :]
    second_hash = op in ("multi_hash_embedding_sum", "dual_hash_gather_sum")
    idx = [((rows+i) * (17 if second_hash and i >= 2 else 13) +
            cols * (5 if second_hash and i >= 2 else 7)) % m for i in range(4)]
    values = [(w if op == "multi_hash_embedding_sum" and i >= 2 else v)[r, cols]
              for i, r in enumerate(idx)]
    if op in ("weighted_embedding_bag_mean", "weighted_embedding_bag_sum"):
        weights = [w[r, cols] for r in idx]
        if op == "weighted_embedding_bag_mean":
            weights = [xp.abs(weight) + 0.25 for weight in weights]
        core = sum(a*b for a, b in zip(values, weights))
        if op == "weighted_embedding_bag_mean":
            core = core / sum(weights)
    elif op == "indexed_group_min":
        core = xp.minimum(xp.minimum(values[0], values[1]), xp.minimum(values[2], values[3]))
    elif op == "indexed_group_max":
        core = xp.maximum(xp.maximum(values[0], values[1]), xp.maximum(values[2], values[3]))
    elif op == "indexed_group_l2":
        core = xp.sqrt(sum(x*x for x in values) + 1e-6)
    else:
        core = sum(values)
        if op == "embedding_bag_mean":
            core = core / 4
    out = xp.zeros_like(v)
    out[:m//4] = core
    return out


def step(node, args, result_type, b):
    xp = b.xp
    op = node.op
    v = b.cast(args[0], "fp32")
    w = b.cast(args[1], "fp32") if len(args) == 2 and op not in MODULAR else None
    a = dict(node.attributes)
    m, n = v.shape if len(v.shape) == 2 else (0, 0)
    if op in GROUPED:
        out = _grouped(op, v, w, b)
    elif op == "cast":
        out = v
    elif op == "add":
        out = v + w
    elif op == "mul":
        out = v * w
    elif op == "relu":
        out = xp.maximum(v, b.array(0.))
    elif op == "row_roll":
        out = v[(xp.arange(m) - 1) % m]
    elif op == "half_scale_contract":
        out = v @ (xp.eye(n, dtype=b.dtypes["fp32"]) * 0.5)
    elif op == "row_stat_broadcast":
        temp = b.sum(v, 1, True) / n if a["stat"] == "mean" else b.amax(v, 1, True)
        temp = b.cast(temp, a["temporary_dtype"])
        out = xp.broadcast_to(temp, (m, n))
    elif op == "rotary_pair_transform":
        mate = v[:, xp.arange(n) ^ 1]
        odd = (xp.arange(n) % 2) == 1
        out = xp.where(odd, v*xp.sin(w) + mate*xp.cos(w), v*xp.cos(w) - mate*xp.sin(w))
    elif op == "row_logsumexp":
        peak = b.amax(v, 1, True)
        out = xp.broadcast_to(peak + xp.log(b.sum(xp.exp(v-peak), 1, True)), (m, n))
    elif op == "cumulative_max_scan":
        out = b.max_scan(v)
    elif op in ("bilinear_resample", "weighted_stencil"):
        r, c = xp.arange(m), xp.arange(n)
        down = xp.minimum(r+1, b.array(m-1)); right = xp.minimum(c+1, b.array(n-1))
        if op == "bilinear_resample":
            out = 0.375*v + 0.125*v[:, right] + 0.375*v[down] + 0.125*v[down[:, None], right[None, :]]
        else:
            up = xp.maximum(r-1, b.array(0)); left = xp.maximum(c-1, b.array(0))
            out = 0.5*v + 0.125*(v[up] + v[down] + v[:, left] + v[:, right])
    elif op == "grouped_threshold_count":
        out = xp.zeros_like(v)
        out[:, :n//4] = b.sum((v.reshape(m, n//4, 4) > 0), 2)
    elif op == "dequantize_bias_gelu":
        u = xp.floor(xp.clip(v*32, -127, 127)+0.5)/32 + w
        out = 0.5*u*(1 + xp.tanh(0.7978845608028654*(u + 0.044715*u*u*u)))
    elif op == "affine_state_scan":
        carry = xp.zeros_like(w[:, 0])
        out = xp.empty_like(w)
        for i in range(n):
            carry = v[:, i]*carry + w[:, i]
            out[:, i] = carry
    elif op == "indexed_embedding_sum":
        out = b.sum(v[args[1]], 1)
    elif op == "indexed_scatter_add":
        out = xp.zeros_like(v)
        if b.name == "torch":
            out.index_add_(0, args[1], v)
        else:
            xp.add.at(out, args[1], v)
    elif op == "segment_sum4":
        out = b.sum(v.reshape(m, n//4, 4), 2)
    elif op == "silu_gated_product":
        out = v * (w / (1 + xp.exp(-v)))
    elif op == "scaled_round_clip":
        out = xp.floor(xp.clip(v * 32, -127, 127) + 0.5) / 32
    elif op == "causal_masked_softmax":
        active = xp.arange(n)[None, :] <= xp.arange(m)[:, None]
        masked = xp.where(active, v, b.array(float("-inf")))
        q = xp.exp(masked - b.amax(masked, 1, True))
        out = q / b.sum(q, 1, True)
    elif op == "row_variance_normalize":
        centered = v - b.sum(v, 1, True)/n
        out = centered / xp.sqrt(b.sum(centered*centered, 1, True)/n + 1e-5)
    elif op == "row_topk_select":
        idx = b.argsort_desc(v)[:, :a["k"]]
        return b.cast(v[xp.arange(m)[:, None], idx], result_type[0].dtype), b.cast(idx, "int32")
    else:
        raise ValueError(op)
    return (b.cast(out, result_type[0].dtype),)


def execute_prepared(prepared, *, capture=False):
    values = dict(prepared.inputs)
    for node in prepared.case.program.nodes:
        args = [values[r.producer][r.port] for r in node.inputs]
        values[node.id] = step(node, args, prepared.types[node.id], prepared.backend)
    out = tuple(values[r.producer][r.port] for r in prepared.case.program.outputs)
    return (out, values) if capture else out
