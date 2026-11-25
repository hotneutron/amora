# SPDX-License-Identifier: Apache-2.0
"""Immutable v2 semantic programs and independently versioned case identities."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Union


Dim = Union[int, str]
Scalar = Union[int, str, float, bool]
DTYPES = {"fp16", "fp32", "int32"}
SCHEMA = "portable-core-case-v2"
RECIPE = "pcg64-dyadic-v1"
ORACLE = "portable-independent-v2.1"
REALIZATION = "cpu-explicit-boundaries-v1"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def _name(value):
    if not isinstance(value, str) or not value or any(not (x.isalnum() or x in "-_.") for x in value):
        raise ValueError(f"invalid identifier {value!r}")


@dataclass(frozen=True)
class Tensor:
    shape: tuple[Dim, ...]
    dtype: str

    def __post_init__(self):
        if self.dtype not in DTYPES or not isinstance(self.shape, tuple) or not self.shape:
            raise ValueError("unsupported tensor dtype or rank")
        for dim in self.shape:
            if type(dim) is int and dim > 0:
                continue
            if isinstance(dim, str):
                _name(dim)
                continue
            raise ValueError("shape requires positive integer or named dimensions")

    def bind(self, dimensions):
        return Tensor(tuple(dimensions[x] if isinstance(x, str) else x for x in self.shape), self.dtype)

    def record(self):
        return {"shape": list(self.shape), "dtype": self.dtype}

    @classmethod
    def read(cls, value):
        return cls(tuple(value["shape"]), value["dtype"])


@dataclass(frozen=True)
class Input:
    id: str
    role: str
    tensor: Tensor
    domain: str = "dyadic"
    row_padding: int = 0
    offset: int = 0

    def __post_init__(self):
        _name(self.id)
        _name(self.role)
        if self.domain not in ("dyadic", "positive", "ramp", "distinct", "signed"):
            raise ValueError("unknown input domain")
        if any(type(x) is not int or x < 0 for x in (self.row_padding, self.offset)):
            raise ValueError("input strides and offset must be nonnegative")
        if self.row_padding and len(self.tensor.shape) != 2:
            raise ValueError("row padding requires rank two")

    def record(self):
        return {"id": self.id, "role": self.role, "tensor": self.tensor.record(),
                "domain": self.domain, "row_padding": self.row_padding, "offset": self.offset}


@dataclass(frozen=True)
class Ref:
    producer: str
    port: int = 0

    def __post_init__(self):
        _name(self.producer)
        if type(self.port) is not int or self.port < 0:
            raise ValueError("invalid output port")

    def record(self):
        return [self.producer, self.port]


@dataclass(frozen=True)
class Node:
    id: str
    op: str
    inputs: tuple[Ref, ...]
    attributes: tuple[tuple[str, Scalar], ...] = ()

    def __post_init__(self):
        _name(self.id)
        _name(self.op)
        if not isinstance(self.inputs, tuple) or not all(isinstance(x, Ref) for x in self.inputs):
            raise ValueError("node inputs must be ordered ports")
        if tuple(sorted(self.attributes)) != self.attributes or len(dict(self.attributes)) != len(self.attributes):
            raise ValueError("attributes must be unique and canonically ordered")
        for name, value in self.attributes:
            _name(name)
            if type(value) not in (int, str, float, bool):
                raise ValueError("attributes must be JSON scalars")
        encoded(dict(self.attributes))

    def record(self):
        return {"id": self.id, "op": self.op, "inputs": [x.record() for x in self.inputs],
                "attributes": dict(self.attributes)}


@dataclass(frozen=True)
class Program:
    inputs: tuple[Input, ...]
    nodes: tuple[Node, ...]
    outputs: tuple[Ref, ...]
    unroll: int = 1
    state_edges: tuple[tuple[Ref, str, int], ...] = ()

    def types(self, dimensions):
        from .ops import infer

        values = {x.id: (x.tensor.bind(dimensions),) for x in self.inputs}
        if not self.inputs or not self.nodes or not self.outputs or len(values) != len(self.inputs):
            raise ValueError("program requires unique inputs, nodes and outputs")
        if len({x.role for x in self.inputs}) != len(self.inputs):
            raise ValueError("program input roles must be unique")
        if type(self.unroll) is not int or not 1 <= self.unroll <= 16:
            raise ValueError("invalid bounded unroll")
        if bool(self.state_edges) != (self.unroll > 1):
            raise ValueError("state edges require bounded explicit unroll")
        state = {(consumer, i): source for source, consumer, i in self.state_edges}
        if len(state) != len(self.state_edges):
            raise ValueError("duplicate state edge")
        used_state = set()
        for node in self.nodes:
            if node.id in values:
                raise ValueError("duplicate producer")
            ins = []
            for i, ref in enumerate(node.inputs):
                if ref.producer not in values or ref.port >= len(values[ref.producer]):
                    raise ValueError("input port is missing or does not precede its consumer")
                if (node.id, i) in state:
                    if state[node.id, i] != ref or ref.producer in {x.id for x in self.inputs}:
                        raise ValueError("invalid state edge")
                    used_state.add((node.id, i))
                ins.append(values[ref.producer][ref.port])
            values[node.id] = infer(node.op, ins, dict(node.attributes))
        if set(state) != used_state or any(x.producer not in values or x.port >= len(values[x.producer]) for x in self.outputs):
            raise ValueError("invalid state edge or output port")
        self.canonical()  # also rejects dead work, which cannot silently disappear from a trace
        return values

    def canonical(self):
        """Traverse ordered output/operand ports, preserving shared node identity."""
        source = {x.id: i for i, x in enumerate(self.inputs)}
        nodes = {x.id: x for x in self.nodes}
        seen = {}
        ordered = []
        stack = set()
        state = {(consumer, i): ref for ref, consumer, i in self.state_edges}

        def visit(ref):
            if ref.producer in source:
                return ["input", source[ref.producer], ref.port]
            if ref.producer in stack or ref.producer not in nodes:
                raise ValueError("cyclic or unknown graph reference")
            if ref.producer not in seen:
                node = nodes[ref.producer]
                stack.add(node.id)
                ins = [visit(x) + ["state" if (node.id, i) in state else "data"]
                       for i, x in enumerate(node.inputs)]
                stack.remove(node.id)
                seen[node.id] = len(ordered)
                ordered.append([node.op, dict(node.attributes), ins])
            return ["node", seen[ref.producer], ref.port]

        outputs = [visit(x) for x in self.outputs]
        if len(seen) != len(self.nodes):
            raise ValueError("unreachable or duplicate program nodes")
        return {"inputs": [[x.role, x.tensor.record()] for x in self.inputs],
                "nodes": ordered, "outputs": outputs, "unroll": self.unroll}

    def record(self):
        return {"inputs": [x.record() for x in self.inputs], "nodes": [x.record() for x in self.nodes],
                "outputs": [x.record() for x in self.outputs], "unroll": self.unroll,
                "state_edges": [[r.record(), n, i] for r, n, i in self.state_edges]}

    @classmethod
    def read(cls, value):
        return cls(tuple(Input(**{**x, "tensor": Tensor.read(x["tensor"])}) for x in value["inputs"]),
                   tuple(Node(x["id"], x["op"], tuple(Ref(*r) for r in x["inputs"]),
                              tuple(sorted(x["attributes"].items()))) for x in value["nodes"]),
                   tuple(Ref(*r) for r in value["outputs"]), value["unroll"],
                   tuple((Ref(*r), n, i) for r, n, i in value["state_edges"]))


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    program: Program
    shape: tuple[tuple[str, int], ...]
    seed: int = 20261123
    intent: str = "control"
    upstream_id: str | None = None
    upstream_sha256: str | None = None

    def __post_init__(self):
        _name(self.id)
        _name(self.family)
        if self.intent not in ("reference", "control", "diagnostic", "development"):
            raise ValueError("unknown case intent")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("invalid case seed")
        if tuple(sorted(self.shape)) != self.shape or len(dict(self.shape)) != len(self.shape):
            raise ValueError("shape bindings must be unique and ordered")
        for name, dim in self.shape:
            _name(name)
            if type(dim) is not int or dim <= 0:
                raise ValueError("invalid dimension binding")
        if bool(self.upstream_id) != bool(self.upstream_sha256) or self.intent == "reference" and not self.upstream_id:
            raise ValueError("reference cases require a source crosswalk")
        if self.upstream_sha256 is not None and (len(self.upstream_sha256) != 64 or
                any(x not in "0123456789abcdef" for x in self.upstream_sha256)):
            raise ValueError("invalid upstream case hash")
        needed = {d for item in self.program.inputs for d in item.tensor.shape if isinstance(d, str)}
        if set(dict(self.shape)) != needed:
            raise ValueError("shape bindings must match symbolic program axes")
        self.program.types(dict(self.shape))

    @property
    def program_sha256(self):
        return digest(self.program.canonical())

    @property
    def case_sha256(self):
        return digest({"program_sha256": self.program_sha256, "shape": dict(self.shape), "seed": self.seed,
                       "recipe": RECIPE, "access": [[x.domain, x.row_padding, x.offset] for x in self.program.inputs]})

    def record(self):
        return {"schema": SCHEMA, "id": self.id, "family": self.family, "intent": self.intent,
                "shape": dict(self.shape), "seed": self.seed, "recipe": RECIPE,
                "program": self.program.record(), "program_sha256": self.program_sha256,
                "case_sha256": self.case_sha256, "upstream_id": self.upstream_id,
                "upstream_sha256": self.upstream_sha256}

    @classmethod
    def read(cls, value):
        case = cls(value["id"], value["family"], Program.read(value["program"]),
                   tuple(sorted(value["shape"].items())), value["seed"], value["intent"],
                   value["upstream_id"], value["upstream_sha256"])
        if case.record() != value:
            raise ValueError("unknown case version, structure, or checksum mismatch")
        return case
