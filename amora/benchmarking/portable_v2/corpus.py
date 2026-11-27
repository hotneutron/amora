# SPDX-License-Identifier: Apache-2.0
"""Seal v2 cases and physical inputs for an explicit CUDA node realization."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path

OPS = ("cast", "add", "mul", "relu", "row_roll", "half_scale_contract",
       "row_stat_broadcast", "rotary_pair_transform", "row_logsumexp",
       "cumulative_max_scan", "bilinear_resample", "grouped_threshold_count",
       "dequantize_bias_gelu", "affine_state_scan", "indexed_embedding_sum",
       "indexed_scatter_add", "segment_sum4", "silu_gated_product",
       "scaled_round_clip", "causal_masked_softmax", "row_variance_normalize",
       "row_topk_select", "weighted_embedding_bag_mean", "multi_hash_embedding_sum",
       "indexed_group_min", "indexed_group_l2", "weighted_embedding_bag_sum",
       "embedding_bag_mean", "dual_hash_gather_sum", "indexed_group_max", "weighted_stencil")
DTYPES = {"fp16": 0, "fp32": 1, "int32": 2}
REALIZATION = "cuda-explicit-node-boundaries-v1"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")

def prepare_corpus(root, *, selections=None):
    import numpy as np
    from benchmark_generators.portable_core.expanded.v2 import suites, runner
    from benchmark_generators.portable_core.expanded.v2.execution import Backend, make_inputs, prepare
    from benchmark_generators.portable_core.expanded.v2.reference import oracle
    from benchmark_generators.portable_core.expanded.v2.ops import DISCRETE
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    bank = root / "banks"
    bank.mkdir()
    unique, aliases, blocked, snapshots = {}, {}, {}, []
    panels = selections if selections is not None else [(s,p) for s in suites.SUITES for p in suites.PROFILES]
    for suite, profile in panels:
        try:
            receipt = runner.freeze(root / "frozen" / (suite + "-" + profile), suite, profile)
            snapshots.append(dict(suite=suite, profile=profile, receipt=receipt))
            for case in suites.generate(suite, profile):
                if case.id in unique and unique[case.id].case_sha256 != case.case_sha256:
                    raise ValueError("case ID collision")
                unique[case.id] = case
                aliases.setdefault(case.id, []).append([suite, profile])
        except suites.BlockedSource as exc:
            blocked[suite + "-" + profile] = str(exc)
    def blob(array):
        data = array.tobytes(order="C")
        digest = hashlib.sha256(data).hexdigest()
        target = bank / (digest + ".bin")
        if not target.exists():
            target.write_bytes(data)
        return digest

    cases, lines, groups = [], [], {}
    pilot = []
    for index, case in enumerate(sorted(unique.values(), key=lambda c: c.id)):
        raw = make_inputs(case)
        prepare(case, Backend(), raw)  # validates layout, aliases and index domains
        _, expected = oracle(case, raw, capture=True)
        types = case.program.types(dict(case.shape))
        values, ports, bindings = [], {}, []
        for item in case.program.inputs:
            array = raw[item.id]
            backing = array
            while isinstance(backing.base, np.ndarray):
                backing = backing.base
            m, n = array.shape if array.ndim == 2 else (array.shape[0], 1)
            stride = array.strides[0] // array.itemsize if array.ndim == 2 else 1
            entry = dict(dtype=DTYPES[item.tensor.dtype], m=m, n=n, stride=stride,
                         offset=item.offset, count=backing.size, exact=1,
                         blob=blob(backing), producer=item.id, port=0, input=True,
                         logical_sha256=hashlib.sha256(array.tobytes()).hexdigest())
            ports[item.id, 0] = len(values)
            values.append(entry)
            bindings.append(dict(producer=item.id, port=0, input=True,
                                 logical_sha256=entry["logical_sha256"],
                                 physical_sha256=entry["blob"], strides=list(array.strides),
                                 offset=item.offset))
        nodes, mask = [], 0
        for node in case.program.nodes:
            if node.op not in OPS:
                raise ValueError("unsupported CUDA op " + node.op)
            attrs = dict(node.attributes)
            if node.op == "row_topk_select" and not 1 <= attrs["k"] <= 8:
                raise ValueError("CUDA top-k realization supports k <= 8")
            op = OPS.index(node.op)
            mask |= 1 << op
            args = [ports[r.producer, r.port] for r in node.inputs]
            output_ids = []
            for port, tensor in enumerate(types[node.id]):
                a = np.ascontiguousarray(expected[node.id][port])
                if not np.isfinite(a).all():
                    raise ValueError("non-finite oracle boundary")
                m, n = a.shape
                entry = dict(dtype=DTYPES[tensor.dtype], m=m, n=n, stride=n, offset=0,
                             count=a.size, exact=int(tensor.dtype == "int32" or node.op in DISCRETE),
                             blob=blob(a), producer=node.id, port=port, input=False)
                ports[node.id, port] = len(values)
                output_ids.append(len(values))
                values.append(entry)
                bindings.append(dict(producer=node.id, port=port, input=False,
                                     oracle_sha256=entry["blob"], shape=list(a.shape), dtype=tensor.dtype))
            flags = int(attrs.get("stat") == "max") | (2 * int(attrs.get("temporary_dtype") == "fp16"))
            nodes.append([op, args[0], args[1] if len(args) > 1 else -1,
                          output_ids[0], output_ids[1] if len(output_ids) > 1 else -1,
                          flags, attrs.get("k", 0)])
        final_ports = [ports[r.producer, r.port] for r in case.program.outputs]
        case_record = dict(index=index, **case.record(), aliases=aliases[case.id],
                           operator_mask=mask, bindings=bindings, final_value_indices=final_ports)
        cases.append(case_record)
        groups.setdefault(str(mask), []).append(index)
        if any(profile == "smoke" for _, profile in aliases[case.id]):
            pilot.append(index)
        lines.append(f"{case.id} {len(values)} {len(case.program.inputs)} {len(nodes)} {len(final_ports)}")
        for value in values:
            lines.append(" ".join(str(value[k]) for k in
                                  ("dtype", "m", "n", "stride", "offset", "count", "exact", "blob")))
        lines.extend(" ".join(map(str, node)) for node in nodes)
        lines.append(" ".join(map(str, final_ports)))
    data = (str(len(cases)) + "\n" + "\n".join(lines) + "\n").encode()
    (root / "cases.txt").write_bytes(data)
    manifest = dict(schema="portable-v2-cuda-corpus-v1", realization=REALIZATION,
                    cases=cases, groups=groups, pilot_indices=pilot, blocked=blocked,
                    snapshots=snapshots, numpy_version=np.__version__,
                    files={"cases.txt": hashlib.sha256(data).hexdigest()},
                    banks={p.name: sha(p) for p in sorted(bank.glob("*.bin"))},
                    family_counts=dict(Counter(c["family"] for c in cases)),
                    shape_counts=dict(Counter(str(c["shape"]) for c in cases)))
    write_json(root / "manifest.json", manifest)
    write_json(root / "seal.json", dict(manifest_sha256=sha(root / "manifest.json"),
                                       cases=len(cases), realization=REALIZATION))
    return manifest

def verify_corpus(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    seal = json.loads((root / "seal.json").read_text())
    if sha(root / "manifest.json") != seal["manifest_sha256"]:
        raise ValueError("changed corpus manifest")
    if manifest["schema"] != "portable-v2-cuda-corpus-v1" or manifest["realization"] != REALIZATION:
        raise ValueError("unknown corpus realization")
    for name, digest in manifest["files"].items():
        if sha(root / name) != digest:
            raise ValueError("changed corpus file " + name)
    for name, digest in manifest["banks"].items():
        if sha(root / "banks" / name) != digest:
            raise ValueError("changed input/oracle bank " + name)
    from benchmark_generators.portable_core.expanded.v2.runner import verify
    for item in manifest["snapshots"]:
        verify(root / "frozen" / (item["suite"] + "-" + item["profile"]))
    return manifest

