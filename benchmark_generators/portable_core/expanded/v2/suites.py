# SPDX-License-Identifier: Apache-2.0
"""Exact, source-only Gate-v2 graphs and separately identified H3 controls."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import gzip
from importlib.resources import files
import json
import itertools

from .ops import DIAGNOSTIC
from .spec import Input, Tensor, Ref, Node, Program, Case, digest

SUITES = ("h3-semantics", "h3-programs", "operation-gate-v2", "mechanism-controls")
PROFILES = ("smoke", "reference", "development")
GATE_SOURCE = "9e3d8f458755c0d3c5556dd240140a639231bc716ec1b9595b51b359caadf014"
H3 = ("affine_state_scan", "indexed_embedding_sum", "weighted_stencil", "indexed_scatter_add",
      "segment_sum4", "silu_gated_product", "scaled_round_clip")
KINDS = {"gather": "row_roll", "contraction": "half_scale_contract", "reduction": "row_stat_broadcast",
         "broadcast": "row_stat_broadcast", "pointwise": "relu"}


class BlockedSource(ValueError):
    pass


@lru_cache(maxsize=1)
def _source():
    data = json.loads(gzip.decompress(files(__package__).joinpath("data", "gate-source.json.gz").read_bytes()))
    if data.get("schema") != "portable-core-gate-source-v1" or digest(data) != GATE_SOURCE:
        raise ValueError("pinned v2 source registry changed")
    return data


def source_provenance():
    data = _source()
    return {"source_bundle_sha256": GATE_SOURCE, "accorde_commit": data["commit"],
            "source_blobs_sha256": dict(data["source_blobs_sha256"])}


def _gate_case(source):
    p = source["executable_program"]
    if source["case_sha256"] != digest({k: v for k, v in source.items() if k not in ("case_id", "case_sha256")}):
        raise ValueError("upstream case hash drift")
    inputs = tuple(Input(name, name, Tensor(tuple(axes), "fp16")) for name, axes in p["external_shapes"])
    nodes = []
    for row in p["operations"]:
        op = row["kind"]
        attrs = ()
        metadata = dict(row["attributes"])
        if op == "pointwise":
            op = metadata.get("op", "relu")
            if op not in ("add", "mul", "relu"):
                raise ValueError("unexpected upstream pointwise variant")
        elif op == "gather":
            if metadata.get("op") != "row_permutation":
                raise ValueError("unexpected upstream gather variant")
            op = "row_roll"
        elif op == "contraction":
            op = "half_scale_contract"
        elif op in ("reduction", "broadcast"):
            # The pinned fixture is authoritative over historical motif shorthand:
            # sum_broadcast actually means row mean; first_column_broadcast means row max.
            if metadata.get("op") != ("sum_broadcast" if op == "reduction" else "first_column_broadcast"):
                raise ValueError("unexpected upstream row variant")
            attrs = (("stat", "mean" if op == "reduction" else "max"), ("temporary_dtype", "fp16"))
            op = "row_stat_broadcast"
        nodes.append(Node(row["name"], op, tuple(Ref(x) for x in row["inputs"]), attrs))
    state = ()
    if p["unroll_count"] == 2:
        second = next(node for node in nodes if node.id.endswith("_u1"))
        state = ((second.inputs[0], second.id, 0),)
    dims = tuple((axis, source["shape"][axis]) for axis in ("M", "N"))
    return Case("v2-" + source["case_id"], source["lane"],
                Program(inputs, tuple(nodes), (Ref(p["output"]),), p["unroll_count"], state),
                dims, intent="reference" if source["axis"] == "gate_v2_external_operation" else "development",
                upstream_id=source["case_id"], upstream_sha256=source["case_sha256"])


def _selected_gate():
    rows = _source()["reference"]
    earliest = {}
    for r in rows:
        key = (r["lane"], r["family"])
        earliest[key] = min(earliest.get(key, r["motif_id"]), r["motif_id"])
    witnesses = [row for row in rows if row["motif_id"] == earliest[row["lane"], row["family"]]]
    if len(witnesses) != 240 or set(Counter((r["lane"], r["family"]) for r in witnesses).values()) != {3}:
        raise ValueError("mandatory Torch witness matrix is incomplete")
    return witnesses


def _inputs(dtype="fp32"):
    return (Input("x", "primary", Tensor(("M", "N"), dtype)),)


def _control(family, m, n, *, variant="base", intent="control", padding=0, offset=0):
    base = Input("x", "primary", Tensor(("M", "N"), "fp32"), "dyadic", padding, offset)
    inputs = [base]
    if family in ("affine_state_scan", "silu_gated_product"):
        if family == "affine_state_scan":
            inputs[0] = Input("x", "coefficient", base.tensor, "positive", padding, offset)
        inputs.append(Input("y", "auxiliary", base.tensor))
    if family in ("indexed_embedding_sum", "indexed_scatter_add"):
        rank = ("M", 4) if family == "indexed_embedding_sum" else ("M",)
        domain = {"distinct": "distinct", "reuse": "positive", "collide": "ramp"}.get(variant, "dyadic")
        inputs.append(Input("idx", "indices", Tensor(rank, "int32"), domain))
    attrs = (("k", 4), ("ties", "lowest_index")) if family == "row_topk_select" else ()
    out = (Ref("z"), Ref("z", 1)) if attrs else (Ref("z"),)
    return Case(f"control-{family}-{variant}-{m}x{n}", family,
                Program(tuple(inputs), (Node("z", family, tuple(Ref(x.id) for x in inputs), attrs),), out),
                (("M", m), ("N", n)), intent=intent)


def _h3_semantics(profile):
    if profile == "smoke":
        return [_control(family, 16, 32) for family in H3]
    rows = [_control(family, m, n) for family in H3 for m, n in ((32, 64), (24, 128), (64, 48), (17, 96))]
    for family in ("indexed_embedding_sum", "indexed_scatter_add"):
        rows.extend(_control(family, 64, 96, variant=variant, padding=padding, offset=offset, intent="development")
                    for variant, padding, offset in (("distinct", 0, 0), ("reuse", 0, 0), ("collide", 0, 0),
                        ("strided", 13, 0), ("offset", 0, 3), ("unchanged", 0, 0)))
    return rows


def _edge_op(name, kind, source):
    attrs = ()
    if kind in ("reduction", "broadcast"):
        attrs = (("stat", "mean" if kind == "reduction" else "max"), ("temporary_dtype", "fp32"))
    return Node(name, KINDS[kind], (Ref(source),), attrs)


def _fork_join(n, fork):
    nodes = (Node("a", "relu", (Ref("x"),)), Node("b", "row_roll", (Ref("x" if fork else "a"),)),
             Node("c", "add", (Ref("a"), Ref("b"))))
    return Case(f"control-{'fork-join' if fork else 'chain'}-{n}", "fork-join" if fork else "chain",
                Program(_inputs(), nodes, (Ref("c"),)), (("M", n), ("N", n)))


TOPOLOGIES = {
    "cross-ladder": (('x',), ('x',), ('0', '1'), ('2',), ('1', '3'), ('2', '4')),
    "asymmetric-hourglass-tail": (('x',), ('x',), ('x',), ('0', '1'), ('3', '2'), ('4',), ('5',)),
    "dual-residual-weave": (('x',), ('0',), ('0', '1'), ('2',), ('1', '3'), ('0', '4')),
    "tri-fork-cascade": (('x',), ('x',), ('x',), ('0', '1'), ('3', '2'), ('4',), ('1', '5')),
    "deep-residual-spine": (('x',),) + tuple((str(i-1), '0') if i % 3 == 0 else (str(i-1),) for i in range(1, 16)),
    "dual-trunk-cross-weave": (('x',), ('x',)) + tuple((str(i-2), str(i-1)) if i % 4 > 1 else (str(i-2),)
                                                     for i in range(2, 16)) + (('14', '15'),),
    "nested-diamond-cascade": tuple(edge for i in range(4) for edge in (
        ('x' if i == 0 else str(4*i-1),), ('x' if i == 0 else str(4*i-1),), (str(4*i+1),), (str(4*i), str(4*i+2)))),
    "wide-four-branch-pyramid": tuple(('x' if i < 4 else str(i-4),) for i in range(16)) +
                               (('12', '13'), ('14', '15'), ('16', '17'), ('18',)),
}


def _topology(family, m, n):
    nodes = []
    for i, edges in enumerate(TOPOLOGIES[family]):
        ref = lambda s: Ref("x" if s == "x" else "n"+s)
        if len(edges) == 1:
            nodes.append(_edge_op(f"n{i}", ("gather", "reduction", "pointwise", "broadcast")[i % 4], ref(edges[0]).producer))
        else:
            nodes.append(Node(f"n{i}", "add", tuple(map(ref, edges))))
    consumed = {ref.producer for node in nodes for ref in node.inputs}
    outputs = tuple(Ref(node.id) for node in nodes if node.id not in consumed)
    return Case(f"control-{family}-{m}x{n}", family, Program(_inputs(), tuple(nodes), outputs),
                (("M", m), ("N", n)))


def _h3_programs(profile):
    rows = [_fork_join(32, b) for b in (False, True)]
    sizes = (160, 320) if profile == "development" else (16,)
    pairs = tuple(itertools.product(KINDS, repeat=2)) if profile == "development" else (
        ("gather", "reduction"), ("reduction", "gather"), ("contraction", "broadcast"), ("broadcast", "pointwise"))
    for n, (a, b) in itertools.product(sizes, pairs):
        rows.append(Case(f"control-edge-{a}-{b}-{n}", f"edge-{a}-{b}",
                         Program(_inputs(), (_edge_op("a", a, "x"), _edge_op("b", b, "a")), (Ref("b"),)),
                         (("M", n), ("N", n))))
    if profile == "development":
        for family in TOPOLOGIES:
            rows += [_topology(family, m, n) for m, n in ((224, 224), (448, 448), (112, 448), (448, 112))]
    return rows


def generate(suite, profile="smoke"):
    if suite not in SUITES or profile not in PROFILES:
        raise ValueError("unknown suite or profile")
    if suite.startswith("h3-"):
        if profile == "reference":
            raise BlockedSource("blocked_missing_source: approved per-case H3 reference bundle not available")
        return _h3_semantics(profile) if suite == "h3-semantics" else _h3_programs(profile)
    source = _source()
    if suite == "operation-gate-v2":
        if profile == "reference":
            return [_gate_case(r) for r in source["reference"]]
        if profile == "development":
            return [_gate_case(r) for r in source["development"]]
        rows = _selected_gate()
        first = {lane: next(r for r in rows if r["lane"] == lane and r["shape"]["M"] == 128)
                 for lane in dict.fromkeys(r["lane"] for r in rows)}
        return [_gate_case(r) for r in first.values()]
    if profile == "reference":
        return [_gate_case(r) for r in source["development"]]
    sizes = ((16, 32),) if profile == "smoke" else ((32, 64), (32, 96), (24, 128))
    return [_control(op, m, n, intent="diagnostic") for op in sorted(DIAGNOSTIC) for m, n in sizes]


def required(case, suite, profile):
    if suite == "operation-gate-v2" and profile == "reference":
        return ("numpy", "torch") if case.upstream_id in _witness_ids() else ("numpy",)
    if suite.startswith("h3-") or suite == "operation-gate-v2" and profile == "smoke":
        return ("numpy", "torch")
    return ("numpy",)


@lru_cache(maxsize=1)
def _witness_ids():
    return {x["case_id"] for x in _selected_gate()}


def coverage(suite):
    if suite not in SUITES:
        raise ValueError("unknown suite")
    if suite.startswith("h3-"):
        status = _source()["h3_exact"]
        panels = [p for p in status["report_only_panels"] if
                  (p["panel"] == "seven_scoped_semantic_routes") == (suite == "h3-semantics")]
        return {"reference_status": "blocked_missing_source", "owner": status["owner_for_source_export"],
                "pending_panels": panels, "spec_parity": "not_evaluated"}
    return {"reference_status": "source_pinned", "historical_input_byte_parity": "not_verified",
            "spec_parity": "archived_graph_port_and_shape_crosswalk"}
