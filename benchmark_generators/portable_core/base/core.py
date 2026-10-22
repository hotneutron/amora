# SPDX-License-Identifier: Apache-2.0
# Adapted from Accorde: package namespace relocation only (2026-10-22).
"""Backend-neutral core20 contracts, deterministic inputs and independent oracles.

This module defines mathematical programs, not ISA-specific schedules. CPU
preflight is not a hardware performance label or an OOD predictor qualification.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

SCHEMA = "cross-isa-core20-v1"
FAMILIES = tuple([f"A{i:02}" for i in range(1, 9)] + [f"C{i:02}" for i in range(1, 9)] + [f"D{i:02}" for i in range(1, 5)])
DTYPE_BYTES = {"fp32": 4, "bf16": 2, "int32": 4}
PROFILES = {
    "smoke": [
        ("dev_small", 8, 16, "source_development"),
        ("dev_medium", 16, 32, "source_development"),
        ("tail", 7, 17, "sealed_shape"),
        ("scale", 32, 64, "sealed_shape"),
    ],
    "native": [
        ("dev_small", 32, 128, "source_development"),
        ("dev_square", 128, 128, "source_development"),
        ("dev_medium", 128, 256, "source_development"),
        ("dev_large", 256, 256, "source_development"),
        ("tail", 127, 129, "sealed_shape"),
        ("scale", 512, 512, "sealed_shape"),
        ("wide", 32, 512, "sealed_shape"),
        ("tall", 512, 32, "sealed_shape"),
    ],
}
VARIANTS = {
    "A02": ("affine", "add", "sub", "mul", "div", "square", "to_bf16", "to_fp32"),
    "A03": ("relu", "exp", "rsqrt", "silu"),
    "A04": ("axis0", "axis1"), "A05": ("axis0", "axis1"), "A06": ("axis0", "axis1"),
    "D03": ("PP", "MP", "MM", "RR", "GP", "TT"),
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def numel(shape: list[int] | tuple[int, ...]) -> int:
    return math.prod(shape)


def _bf16(x: Any) -> np.ndarray:
    # A reference representation of BF16 values in FP32; never a native-BF16 claim.
    a = np.asarray(x, dtype=np.float32)
    bits = a.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    special = (bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    kept = bits & np.uint32(0xFFFF0000)
    kept = np.where((bits & np.uint32(0x007FFFFF)) != 0, kept | np.uint32(0x00400000), kept)
    return np.where(special, kept, rounded).astype(np.uint32).view(np.float32)


class Builder:
    def __init__(self, case_id: str):
        self.case_id = case_id
        self.inputs: dict[str, dict[str, Any]] = {}
        self.nodes: list[dict[str, Any]] = []
        self.specs: dict[str, dict[str, Any]] = {}

    def inp(self, name: str, shape: tuple[int, ...], dtype: str = "fp32", domain: str = "uniform", **extra: Any) -> str:
        spec = {"shape": list(shape), "dtype": dtype, "domain": domain, **extra}
        self.inputs[name] = spec
        self.specs[name] = spec
        return name

    def scalar(self, name: str, value: float) -> str:
        return self.inp(name, (), value=value, domain="constant")

    def op(self, op: str, *args: str, **attrs: Any) -> str:
        specs = [self.specs[a] for a in args]
        shapes = [tuple(s["shape"]) for s in specs]
        dtype = "fp32"
        if op == "matmul":
            if any(s["dtype"] != "bf16" for s in specs):
                raise ValueError("M requires explicit BF16 operands")
            a, w = shapes
            if len(a) != 2 or len(w) != 2 or a[1] != w[0]:
                raise ValueError("incompatible matrix ports")
            shape = (a[0], w[1])
        elif op in ("add", "sub", "mul", "div"):
            if any(s["dtype"] != "fp32" for s in specs):
                raise ValueError("pointwise arithmetic requires explicit FP32 ports")
            shape = np.broadcast_shapes(*shapes)
        elif op == "cast":
            dtype = attrs["dtype"]
            shape = shapes[0]
        elif op == "reshape":
            dtype = specs[0]["dtype"]
            shape = tuple(attrs["shape"])
            if numel(shape) != numel(shapes[0]):
                raise ValueError("invalid view")
        elif op in ("sum", "max"):
            axis = attrs["axis"]
            shape = tuple(v for i, v in enumerate(shapes[0]) if i != axis)
        elif op == "broadcast":
            shape = tuple(attrs["shape"])
            source = shapes[0]
            if attrs.get("axis") is not None:
                if len(source) != 1 or len(shape) != 2 or source[0] != shape[attrs["axis"]]:
                    raise ValueError("invalid vector broadcast")
            elif np.broadcast_shapes(source, shape) != shape:
                raise ValueError("invalid batched broadcast")
        elif op == "transpose":
            if len(shapes[0]) != 2:
                raise ValueError("T is 2-D")
            shape = shapes[0][::-1]
        elif op == "gather":
            if len(shapes[0]) != 2 or len(shapes[1]) != 1 or specs[1]["dtype"] != "int32":
                raise ValueError("G requires table and INT32 row-index vector")
            shape = (shapes[1][0], shapes[0][1])
        else:
            if specs[0]["dtype"] != "fp32":
                raise ValueError("nonlinear input requires explicit FP32")
            shape = shapes[0]
        name = f"n{len(self.nodes)}"
        node = {"id": name, "op": op, "inputs": list(args), "attrs": attrs, "shape": list(shape), "dtype": dtype,
                "materialized": op != "reshape"}
        self.nodes.append(node)
        self.specs[name] = node
        return name

    def broad(self, value: str, shape: tuple[int, ...], axis: int | None = None) -> str:
        return self.op("broadcast", value, shape=list(shape), axis=axis)

    def affine(self, x: str) -> str:
        return self.op("add", self.op("mul", x, self.scalar("alpha", 0.7)), self.scalar("beta", 0.2))

    def norm(self, x: str, rows: int, cols: int, layer: bool = False) -> str:
        divisor = self.scalar("width", float(cols))
        eps = self.scalar("epsilon", 1e-5)
        gamma = self.inp("gamma", (cols,), domain="positive")
        z = x
        if layer:
            mean = self.op("div", self.op("sum", x, axis=1), divisor)
            z = self.op("sub", x, self.broad(mean, (rows, cols), axis=0))
        variance = self.op("div", self.op("sum", self.op("square", z), axis=1), divisor)
        inverse = self.op("rsqrt", self.op("add", variance, eps))
        y = self.op("mul", z, self.broad(inverse, (rows, cols), axis=0))
        y = self.op("mul", y, self.broad(gamma, (rows, cols), axis=1))
        if layer:
            beta = self.inp("norm_beta", (cols,))
            y = self.op("add", y, self.broad(beta, (rows, cols), axis=1))
        return y


def build_case(family: str, variant: str, scenario: str, rows: int, cols: int, role: str, profile: str) -> dict[str, Any]:
    case_id = f"{profile}/{family}/{variant}/{scenario}"
    b = Builder(case_id)
    tags = {family}
    r, c = rows, cols
    if family == "A01":
        y = b.op("matmul", b.inp("A", (r, c), "bf16"), b.inp("W", (c, c), "bf16", "weight", scale=0.5 / math.sqrt(c)))
    elif family == "A02":
        x = b.inp("X", (r, c), "bf16" if variant == "to_fp32" else "fp32")
        if variant == "affine":
            y = b.affine(x)
        elif variant in ("to_bf16", "to_fp32"):
            y = b.op("cast", x, dtype="bf16" if variant == "to_bf16" else "fp32")
        elif variant == "square":
            y = b.op("square", x)
        else:
            other = b.inp("Y", (r, c), domain="positive" if variant == "div" else "uniform")
            y = b.op(variant, x, other)
    elif family == "A03":
        y = b.op(variant, b.inp("X", (r, c), domain="positive" if variant == "rsqrt" else "uniform"))
    elif family in ("A04", "A05"):
        y = b.op("sum" if family == "A04" else "max", b.inp("X", (r, c)), axis=int(variant[-1]))
    elif family == "A06":
        axis = int(variant[-1])
        y = b.broad(b.inp("v", ((r, c)[axis],)), (r, c), axis=axis)
    elif family == "A07":
        y = b.op("transpose", b.inp("X", (r, c)))
    elif family == "A08":
        y = b.op("gather", b.inp("table", (2*r+3, c)), b.inp("index", (r,), "int32", "index", upper=2*r+3))
    elif family == "C01":
        x = b.inp("X", (r, c))
        centered = b.op("sub", x, b.broad(b.op("max", x, axis=1), (r,c), axis=0))
        e = b.op("exp", centered)
        y = b.op("div", e, b.broad(b.op("sum", e, axis=1), (r,c), axis=0))
    elif family in ("C02", "C03", "C07"):
        x = b.inp("X", (r, c))
        y = b.norm(x, r, c, layer=family == "C03")
        # Centered LayerNorm contains RMS normalization of Z; close this tag
        # through all parent-derived edge supervision as well as whole graphs.
        if family in ("C03", "C07"):
            tags.add("C02")
        if family == "C07":
            y = b.op("matmul", b.op("cast", y, dtype="bf16"), b.inp("W", (c,c), "bf16", "weight", scale=0.5/math.sqrt(c)))
    elif family == "C04":
        groups, lookups = max(1,r//2), r
        table = b.inp("table", (2*r+3,c))
        index = b.inp("index", (groups,lookups), "int32", "index", upper=2*r+3)
        flat = b.op("reshape", index, shape=[groups*lookups])
        z = b.op("reshape", b.op("gather", table, flat), shape=[groups,lookups,c])
        a = b.inp("a", (groups,lookups), domain="positive")
        av = b.op("reshape", a, shape=[groups,lookups,1])
        product = b.op("mul", z, b.broad(av, (groups,lookups,c)))
        numerator = b.op("sum", product, axis=1)
        denominator = b.broad(b.op("sum", a, axis=1), (groups,c), axis=0)
        y = b.op("div", numerator, denominator)
    elif family == "C05":
        mm = b.op("matmul", b.inp("A", (r,c), "bf16"), b.inp("W", (c,c), "bf16", "weight", scale=0.5/math.sqrt(c)))
        bias = b.broad(b.inp("bias", (c,)), (r,c), axis=1)
        y = b.op("silu", b.op("add", mm, bias))
    elif family == "C06":
        a = b.op("cast", b.op("relu", b.inp("X", (r,c))), dtype="bf16")
        y = b.op("matmul", a, b.inp("W", (c,c), "bf16", "weight", scale=0.5/math.sqrt(c)))
    elif family == "C08":
        g = b.op("gather", b.inp("table", (2*r+3,c)), b.inp("index", (r,), "int32", "index", upper=2*r+3))
        y = b.op("matmul", b.op("cast", g, dtype="bf16"), b.inp("W", (c,c), "bf16", "weight", scale=0.5/math.sqrt(c)))
    elif family in ("D01", "D02"):
        x = b.inp("X", (r,c))
        u = b.op("relu", b.affine(x))
        g = b.inp("g_weight", (r,c), domain="positive")
        h = b.inp("h_weight", (r,c), domain="positive")
        v = b.op("silu", b.op("mul", u, g))
        w = b.op("silu", b.op("mul", v if family == "D01" else u, h))
        y = b.op("add", w, x if family == "D01" else v)
    elif family == "D03":
        if variant == "PP":
            u = b.affine(b.inp("X0", (r,c)))
            v = b.op("silu", b.inp("X1", (r,c)))
        elif variant in ("MP", "MM"):
            u = b.op("matmul", b.inp("A0", (r,c), "bf16"), b.inp("W0", (c,c), "bf16", "weight", scale=0.5/math.sqrt(c)))
            if variant == "MM":
                v = b.op("matmul", b.inp("A1", (r,c), "bf16"), b.inp("W1", (c,c), "bf16", "weight", scale=0.5/math.sqrt(c)))
            else:
                v = b.op("silu", b.inp("X1", (r,c)))
        elif variant == "RR":
            u = b.op("sum", b.inp("X0", (r,c)), axis=1)
            v = b.op("max", b.inp("X1", (r,c)), axis=1)
        elif variant == "GP":
            u = b.op("gather", b.inp("table", (2*r+3,c)), b.inp("index", (r,), "int32", "index", upper=2*r+3))
            v = b.op("silu", b.inp("X1", (r,c)))
        elif variant == "TT":
            u = b.op("transpose", b.inp("X0", (c,r)))
            v = b.op("transpose", b.inp("X1", (c,r)))
        else:
            raise ValueError(variant)
        y = b.op("add", u, v)
    elif family == "D04":
        h = 2*c
        u = b.op("matmul", b.inp("A", (r,c), "bf16"), b.inp("W1", (c,h), "bf16", "weight", scale=0.5/math.sqrt(c)))
        activation = b.op("cast", b.op("silu", u), dtype="bf16")
        y = b.op("matmul", activation, b.inp("W2", (h,c), "bf16", "weight", scale=0.5/math.sqrt(h)))
    else:
        raise ValueError(family)
    leaves = {node["op"] for node in b.nodes}
    if "silu" in leaves:
        leaves.update(("exp", "div", "add", "mul"))
    if "square" in leaves:
        leaves.add("mul")
    case = {"schema": SCHEMA, "case_id": case_id, "family": family, "variant": variant,
            "scenario": scenario, "role": role, "profile": profile, "rows": r, "width": c,
            "inputs": b.inputs, "nodes": b.nodes, "output": y, "semantic_tags": sorted(tags),
            "semantic_leaves": sorted(leaves), "realization": "decomposed-portable-array-program"}
    case["sha256"] = digest(case)
    return case


def generate_cases(profile: str = "smoke") -> list[dict[str, Any]]:
    if profile not in PROFILES:
        raise ValueError(profile)
    return [build_case(f, v, s, r, c, role, profile) for f in FAMILIES
            for v in VARIANTS.get(f, ("base",)) for s,r,c,role in PROFILES[profile]]


def make_inputs(case: dict[str, Any]) -> dict[str, np.ndarray]:
    seed = int.from_bytes(hashlib.sha256(case["case_id"].encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    arrays = {}
    for name, spec in case["inputs"].items():
        shape, domain = tuple(spec["shape"]), spec["domain"]
        if domain == "constant":
            array = np.asarray(spec["value"], dtype=np.float32)
        elif domain == "index":
            array = rng.integers(0, spec["upper"], size=shape, dtype=np.int32)
        elif domain == "positive":
            array = rng.uniform(0.25, 1.25, size=shape).astype(np.float32)
        elif domain == "weight":
            array = rng.normal(0, spec["scale"], size=shape).astype(np.float32)
        else:
            array = rng.uniform(-1, 1, size=shape).astype(np.float32)
        arrays[name] = _bf16(array) if spec["dtype"] == "bf16" else array
    return arrays


class PreparedInputs:
    """Validated/staged inputs; staging is outside any future measured region."""
    def __init__(self, case_sha256: str, values: dict[str, Any]):
        self.case_sha256 = case_sha256
        self.values = values


def prepare_inputs(case: dict[str, Any], backend: Any, inputs: dict[str, Any] | None = None) -> PreparedInputs:
    raw = make_inputs(case) if inputs is None else inputs
    if set(raw) != set(case['inputs']):
        raise ValueError('input names do not match the frozen contract')
    for name,spec in case['inputs'].items():
        a = np.asarray(raw[name])
        if tuple(a.shape) != tuple(spec['shape']):
            raise ValueError(f'input shape mismatch: {name}')
        expected = np.int32 if spec['dtype']=='int32' else np.float32
        if a.dtype != expected:
            raise ValueError(f'input storage dtype mismatch: {name}')
        if not np.all(np.isfinite(a)):
            raise ValueError(f'nonfinite input: {name}')
        if spec['domain']=='index' and (np.any(a<0) or np.any(a>=spec['upper'])):
            raise ValueError(f'index range violation: {name}')
        if spec['domain']=='positive' and np.any(a<=0):
            raise ValueError(f'positive-domain violation: {name}')
        if spec['dtype']=='bf16' and not np.array_equal(a.view(np.uint32), _bf16(a).view(np.uint32)):
            raise ValueError(f'BF16 input is not rounded to its declared storage: {name}')
    nodes={n['id']:n for n in case['nodes']}
    specs={**case['inputs'],**nodes}
    for n in case['nodes']:
        if n['op']=='gather':
            ref=n['inputs'][1]
            while ref in nodes and nodes[ref]['op']=='reshape':
                ref=nodes[ref]['inputs'][0]
            idx=case['inputs'].get(ref)
            if idx is None or idx['domain']!='index' or idx['upper']>specs[n['inputs'][0]]['shape'][0]:
                raise ValueError('gather needs a statically validated input-index proof')
    return PreparedInputs(case['sha256'],{name:backend.array(raw[name],spec['dtype'])
                                        for name,spec in case['inputs'].items()})


def execute_prepared(case: dict[str, Any], backend: Any, prepared: PreparedInputs) -> Any:
    """Execute only the mathematical program, without host validation/staging."""
    if not isinstance(prepared,PreparedInputs) or prepared.case_sha256!=case['sha256']:
        raise ValueError('prepared inputs do not match this frozen case')
    values=dict(prepared.values)
    for node in case['nodes']:
        args=[values[n] for n in node['inputs']]
        if node['op']=='reshape':
            out=args[0].reshape(tuple(node['shape']))
        else:
            attrs=dict(node['attrs'])
            if node['op']=='gather':
                attrs['indices_validated']=True
            out=backend.execute(node['op'],args,attrs)
        if tuple(out.shape)!=tuple(node['shape']):
            raise ValueError(f"shape mismatch at {node['id']}: {out.shape}")
        values[node['id']]=out
    return values[case['output']]


def evaluate(case: dict[str, Any], backend: Any, inputs: dict[str, Any] | None = None) -> Any:
    return execute_prepared(case,backend,prepare_inputs(case,backend,inputs))


def oracle(case: dict[str, Any], x: dict[str, np.ndarray]) -> np.ndarray:
    """Independent template formulas, with FP64 work and explicit FP32 boundaries."""
    f = lambda a: np.asarray(a, dtype=np.float32)
    d = lambda a: np.asarray(a, dtype=np.float64)
    add = lambda a,b: f(d(a)+d(b))
    mul = lambda a,b: f(d(a)*d(b))
    div = lambda a,b: f(d(a)/d(b))
    total = lambda a,axis: f(np.sum(d(a),axis=axis))
    mm = lambda a,w: f(d(a) @ d(w))
    silu = lambda a: f(d(a)/(1+np.exp(-d(a))))
    affine = lambda a: add(mul(a,x['alpha']),x['beta'])
    def norm(a: np.ndarray, layer: bool = False) -> np.ndarray:
        z = f(d(a)-d(div(total(a,1),x['width']))[:,None]) if layer else a
        var = div(total(mul(z,z),1),x['width'])
        inverse = f(1/np.sqrt(d(add(var,x['epsilon']))))
        y = mul(mul(z,inverse[:,None]),x['gamma'][None,:])
        return add(y,x['norm_beta'][None,:]) if layer else y
    family, variant = case['family'], case['variant']
    if family == 'A01': return mm(x['A'],x['W'])
    if family == 'A02':
        a=x['X']
        if variant=='affine': return affine(a)
        if variant=='to_bf16': return _bf16(a)
        if variant=='to_fp32': return f(a)
        if variant=='square': return mul(a,a)
        return {'add':add,'sub':lambda a,b:f(d(a)-d(b)),'mul':mul,'div':div}[variant](a,x['Y'])
    if family == 'A03':
        a=x['X']
        return {'relu':lambda a:f(np.maximum(a,0)), 'exp':lambda a:f(np.exp(d(a))),
                'rsqrt':lambda a:f(1/np.sqrt(d(a))), 'silu':silu}[variant](a)
    if family=='A04': return total(x['X'],int(variant[-1]))
    if family=='A05': return f(np.max(x['X'],axis=int(variant[-1])))
    if family=='A06':
        v=x['v']
        return np.broadcast_to(v[:,None] if variant=='axis0' else v[None,:],(case['rows'],case['width'])).copy()
    if family=='A07': return x['X'].T.copy()
    if family=='A08': return x['table'][x['index']].copy()
    if family=='C01':
        a=x['X']; centered=f(d(a)-d(np.max(a,axis=1))[:,None]); e=f(np.exp(d(centered)))
        return div(e,total(e,1)[:,None])
    if family in ('C02','C03'): return norm(x['X'],family=='C03')
    if family=='C04':
        z=x['table'][x['index']]
        return div(total(mul(z,x['a'][:,:,None]),1),total(x['a'],1)[:,None])
    if family=='C05': return silu(add(mm(x['A'],x['W']),x['bias'][None,:]))
    if family=='C06': return mm(_bf16(np.maximum(x['X'],0)),x['W'])
    if family=='C07': return mm(_bf16(norm(x['X'])),x['W'])
    if family=='C08': return mm(_bf16(x['table'][x['index']]),x['W'])
    if family in ('D01','D02'):
        u=f(np.maximum(affine(x['X']),0)); v=silu(mul(u,x['g_weight']))
        w=silu(mul(v if family=='D01' else u,x['h_weight']))
        return add(w,x['X'] if family=='D01' else v)
    if family=='D03':
        if variant=='PP': u,v=affine(x['X0']),silu(x['X1'])
        elif variant=='MP': u,v=mm(x['A0'],x['W0']),silu(x['X1'])
        elif variant=='MM': u,v=mm(x['A0'],x['W0']),mm(x['A1'],x['W1'])
        elif variant=='RR': u,v=total(x['X0'],1),np.max(x['X1'],axis=1)
        elif variant=='GP': u,v=x['table'][x['index']],silu(x['X1'])
        elif variant=='TT': u,v=x['X0'].T,x['X1'].T
        else: raise ValueError(variant)
        return add(u,v)
    if family=='D04': return mm(_bf16(silu(mm(x['A'],x['W1']))),x['W2'])
    raise ValueError(family)


def features(case: dict[str, Any]) -> dict[str, Any]:
    """Structural/numeric serving facts only: no case/family/scenario identifiers."""
    nodes = case['nodes']; specs={**case['inputs'], **{n['id']:n for n in nodes}}
    depth={k:0 for k in case['inputs']}; consumers={n['id']:0 for n in nodes}
    records=[]; edges=[]
    for i,n in enumerate(nodes):
        depth[n['id']]=1+max(depth[a] for a in n['inputs'])
        for a in n['inputs']:
            if a in consumers:
                consumers[a]+=1
                edges.append([int(a[1:]),i])
        kind={'matmul':'M','sum':'R','max':'R','broadcast':'B','gather':'G','transpose':'T','reshape':'view'}.get(n['op'],'P')
        load=0 if n['op']=='reshape' else sum(numel(specs[a]['shape'])*DTYPE_BYTES[specs[a]['dtype']] for a in n['inputs'])
        store=0 if n['op']=='reshape' else numel(n['shape'])*DTYPE_BYTES[n['dtype']]
        records.append({'kind':kind,'opcode':n['op'],'shape':n['shape'],'dtype':n['dtype'],
                        'axis':n['attrs'].get('axis'),'load_bytes_logical':load,
                        'store_bytes_logical':store,'depth':depth[n['id']],'input_count':len(n['inputs']),
                        'materialized_intent':n['materialized']})
    return {'schema':SCHEMA,'nodes':records,'edges':edges,'node_count':len(nodes),
            'logical_materialized_node_count':sum(n['materialized'] for n in nodes),
            'view_node_count':sum(n['op']=='reshape' for n in nodes),
            'edge_count':len(edges),'max_depth':max(depth.values()),
            'branch_count':sum(v>1 for v in consumers.values()),
            'join_count':sum(sum(a in consumers for a in n['inputs'])>1 for n in nodes),
            'hardware_facts':None,'compiler_facts':None,'native_realization_verified':False}


def split_manifest(cases: list[dict[str, Any]]) -> dict[str, Any]:
    folds={}
    for held in ('C01','C02','C03','C04','C07','C08','D02','D04'):
        train=[c['case_id'] for c in cases if c['role']=='source_development' and held not in c['semantic_tags']]
        op=[c['case_id'] for c in cases if c['role']=='source_development' and held in c['semantic_tags']]
        joint=[c['case_id'] for c in cases if c['role']=='sealed_shape' and held in c['semantic_tags']]
        shape=[c['case_id'] for c in cases if c['role']=='sealed_shape' and held not in c['semantic_tags']]
        folds[held]={'excluded_tag':held,'train':train,'operation_only':op,'shape_only':shape,'joint':joint}
    return {'schema':SCHEMA,'holdout_rule':'canonical semantic-tag closure including parent-derived edges',
            'folds':folds,'shape_source':[c['case_id'] for c in cases if c['role']=='source_development'],
            'shape_sealed':[c['case_id'] for c in cases if c['role']=='sealed_shape'],
            'native_performance_labels_opened':0}


def edge_plans(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plans=[]
    for c in cases:
        if c['role']!='source_development': continue
        node_positions={node['id']:position for position,node in enumerate(c['nodes'])}
        for i,n in enumerate(c['nodes']):
            for a in n['inputs']:
                if a in node_positions:
                    source=node_positions[a]
                    if c['nodes'][source]['op']=='reshape' or n['op']=='reshape': continue
                    plans.append({'edge_id':f"{c['case_id']}/edge/{source}-{i}",
                                  'parent_case_id':c['case_id'],'parent_semantic_tags':c['semantic_tags'],
                                  'source_node':source,'consumer_node':i,
                                  'source_opcode':c['nodes'][source]['op'],'consumer_opcode':n['op'],
                                  'status':'PLANNED_NOT_MEASURED'})
    return plans
