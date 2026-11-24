# SPDX-License-Identifier: Apache-2.0
"""Offline export from the immutable Accorde Git objects listed in the evidence lock."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / ".plan/20261123-1522-portable-core-h3-operation-evidence.json"
COMMIT = "7c85e7541a3d8dfbb25fec46d96ff644c1ae72b5"
PREFIX = "tools/ppp_profiling/"


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def select(module, src, names):
    """Evaluate only the audited graph definitions, never upstream imports or CLIs."""
    approved = []
    for node in ast.parse(src).body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else (
            node.targets[0].id if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) else None)
        if name in names:
            approved.append(node)
    assert len(approved) == len(names)
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.fix_missing_locations(ast.Module(body=future + approved, type_ignores=[])),
                 "<pinned Accorde graph definitions>", "exec"), module.__dict__)


def development(inputs):
    mod = ModuleType("_accorde_source_export")
    sys.modules[mod.__name__] = mod  # required by stdlib dataclass for postponed annotations
    mod.__dict__.update(dataclass=dataclass, asdict=asdict, itertools=itertools, hashlib=hashlib,
                        json=json, Counter=Counter, defaultdict=defaultdict, lru_cache=lru_cache)
    select(mod, inputs["native_b_corpus"], {"NativeOperation"})
    select(mod, inputs["native_b_meta_motif_ood"], {"MotifGraph", "_label_permutations", "canonical_text", "signature"})
    select(mod, inputs["hardware_384_meta_motif_corpus"], {
        "FAMILIES", "LEAVES", "MERGES", "SHAPES", "MOTIFS_PER_FAMILY", "digest", "MetaMotif", "ExecutableProgram",
        "_graph", "_build", "_parameters", "motifs"})
    select(mod, inputs["operation_gate_v2_support_feasibility"], {"CANDIDATES", "BINARY"})
    select(mod, inputs["operation_gate_v2_support_corpus"], {"BASE_START", "support_motifs", "_order", "_operation", "executable", "registry", "validate"})
    assert mod.validate()["status"] == "pass"
    return mod.registry()


def export(repo, out):
    lock = json.loads(LOCK.read_bytes())
    refs = lock["operation_gate_v2"]["references"] + lock["older_operation_and_development_references"]
    lookup = {x["path"]: x["file_sha256"] for x in refs if x["commit"] == COMMIT}
    paths = {}

    def load(path):
        data = subprocess.check_output(["git", "-C", str(repo), "show", f"{COMMIT}:{path}"])
        got = hashlib.sha256(data).hexdigest()
        if lookup[path] != got:
            raise ValueError(f"source drift: {path}")
        paths[path] = got
        return data

    registry_path, = [path for path in lookup if path.endswith("/operation_gate_v2_v1/registry.json")]
    cases = json.loads(load(registry_path))
    assert hashlib.sha256(canonical(cases)).hexdigest() == lock["operation_gate_v2"]["case_registry_canonical_sha256"]
    helpers = ("native_b_corpus", "native_b_meta_motif_ood", "hardware_384_meta_motif_corpus",
               "operation_gate_v2_support_corpus", "operation_gate_v2_support_feasibility")
    controls = development({name: load(f"{PREFIX}{name}.py").decode() for name in helpers})
    for name in ("operation_gate_v2_external_fixture", "operation_gate_v2_external_feasibility", "multi_axis_ood_fixture", "native_b_fixture"):
        load(f"{PREFIX}{name}.py")
    assert len(cases) == 720 and len(controls) == 288
    for row in cases + controls:
        assert set(row) == {"axis", "case_id", "case_sha256", "executable_program", "family", "implementation",
                            "lane", "meta_graph", "motif_id", "shape"}
    data = {"schema": "portable-core-gate-source-v1", "commit": COMMIT,
            "source_blobs_sha256": paths, "reference": cases, "development": controls,
            "h3_exact": lock["h3_exact_reference_availability"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as handle:
        with gzip.GzipFile(fileobj=handle, filename="", mode="wb", compresslevel=9, mtime=0) as packed:
            packed.write(canonical(data))
    print(f"{len(cases)} reference / {len(controls)} development specs; source sha256 {hashlib.sha256(canonical(data)).hexdigest()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--accorde", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    export(args.accorde, args.out)