# SPDX-License-Identifier: Apache-2.0
"""Immutable v2 snapshots, bounded CPU checks and complete-run aggregation."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

from .spec import ORACLE, REALIZATION, encoded
from .suites import coverage, generate, required, source_provenance

SCHEMA = "portable-core-snapshot-v2"
RUN_SCHEMA = "portable-core-correctness-v2"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(path):
    return json.loads(path.read_bytes())


def _write(path, data):
    payload = encoded(data) + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
    return _sha(payload)


def _runtime_sources():
    parent = Path(__file__).parent
    return {p.name: _sha(p.read_bytes()) for p in sorted(parent.glob("*.py"))}


def _records(suite, profile):
    cases = generate(suite, profile)
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("empty or duplicate case registry")
    return cases, [dict(case.record(), required_backends=list(required(case, suite, profile))) for case in cases]


def _metadata(suite, profile, cases):
    groups = defaultdict(list)
    for case in cases:
        groups[case.program_sha256].append(case.id)
    crosswalk = {case.id: {"source_case_id": case.upstream_id, "source_case_sha256": case.upstream_sha256,
                          "spec_parity": "source_graph_ports_and_shape" if case.upstream_id else "not_applicable",
                          "historical_input_byte_parity": "not_verified" if case.upstream_id else "not_applicable"}
                 for case in cases}
    return {
        "scope": "portable correctness only",
        "source_coverage": coverage(suite),
        "suite": suite, "profile": profile, "cases": len(cases),
        "family_counts": dict(Counter(case.family for case in cases)),
        "required_backends": dict(Counter(b for case in cases for b in required(case, suite, profile))),
        "source_crosswalk": crosswalk,
        "exposure": {"historical_or_generated": "not_sealed", "scientific_split": "not_assigned",
                     "program_groups": dict(groups)},
        "provenance": source_provenance(),
    }


def freeze(folder, suite, profile="smoke"):
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError("refuse frozen overwrite")
    cases, records = _records(suite, profile)  # fail closed before writing a partial blocked snapshot
    metadata = _metadata(suite, profile, cases)
    folder.mkdir(parents=True, exist_ok=True)
    checksums = {"manifest.json": _write(folder/"manifest.json", {"schema": SCHEMA, "cases": records}),
                 "coverage.json": _write(folder/"coverage.json", metadata)}
    receipt = {"schema": SCHEMA, "status": "FROZEN", "suite": suite, "profile": profile,
               "cases": len(cases), "files": checksums, "runtime_source_sha256": _runtime_sources(),
               "oracle": ORACLE, "realization": REALIZATION, "native_performance_qualified": False}
    _write(folder/"freeze_receipt.json", receipt)
    return receipt


def verify(folder, *, return_cases=False):
    folder = Path(folder)
    receipt = _read(folder/"freeze_receipt.json")
    if (receipt.get("schema") != SCHEMA or receipt.get("oracle") != ORACLE or receipt.get("realization") != REALIZATION
            or receipt.get("status") != "FROZEN" or receipt.get("native_performance_qualified") is not False
            or receipt.get("runtime_source_sha256") != _runtime_sources()
            or set(receipt.get("files", {})) != {"manifest.json", "coverage.json"}):
        raise ValueError("invalid v2 receipt, unknown version or changed sources")
    for name, checksum in receipt["files"].items():
        path = folder/name
        if path.is_symlink() or _sha(path.read_bytes()) != checksum:
            raise ValueError(f"invalid v2 snapshot checksum: {name}")
    suite, profile = receipt["suite"], receipt["profile"]
    cases, expected = _records(suite, profile)
    if (receipt["cases"] != len(cases) or _read(folder/"manifest.json") != {"schema": SCHEMA, "cases": expected}
            or _read(folder/"coverage.json") != _metadata(suite, profile, cases)):
        raise ValueError("snapshot differs from canonical pinned catalogue")
    return (receipt, cases) if return_cases else receipt


def check(folder, output, backends=None, device="cpu", *, start=0, stop=None, timeout=120, profile=None, suite=None):
    from .execution import Backend, BackendUnavailable, make_inputs, prepare
    from .reference import check as compare

    receipt, cases = verify(folder, return_cases=True)
    frozen_suite, frozen_profile = receipt["suite"], receipt["profile"]
    if suite is not None and suite != frozen_suite or profile is not None and profile != frozen_profile:
        raise ValueError("suite/profile differs from the frozen v2 snapshot")
    stop = len(cases) if stop is None else stop
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(cases) or not 0 < timeout <= 600:
        raise ValueError("invalid correctness shard or timeout")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refuse correctness receipt overwrite")
    selected = cases[start:stop]
    matrix = {c.id: required(c, frozen_suite, frozen_profile) for c in selected}
    wanted = set(backend for names in matrix.values() for backend in names)
    allowed = wanted if backends is None else set(backends)
    missing = {name: "excluded by caller" for name in wanted - allowed}
    contexts = {}
    for name in sorted(allowed):
        try:
            contexts[name] = Backend(name, device)
        except BackendUnavailable as error:
            missing[name] = str(error)
    rows = []
    started = time.monotonic()
    timed_out = False
    for case in selected:
        if time.monotonic() - started >= timeout:
            timed_out = True
            break
        raw = make_inputs(case)
        for name in sorted(set(matrix[case.id]) | (allowed - wanted)):
            if name in missing:
                continue
            obj = prepare(case, contexts[name], raw)
            passed, reason = compare(obj, raw)
            rows.append({"case_id": case.id, "backend": name, "passed": passed, "reason": reason,
                         "realization_sha256": obj.realization_sha256})
    required_rows = {(case.id, name) for case in selected for name in matrix[case.id]}
    pass_rows = {(r["case_id"], r["backend"]) for r in rows if r["passed"]}
    output.mkdir(parents=True, exist_ok=True)
    rows_sha = _write(output/"checks.json", rows)
    result = {"schema": RUN_SCHEMA, "status": "PASS" if required_rows <= pass_rows and not timed_out else "FAIL",
              "suite": frozen_suite, "profile": frozen_profile, "case_start": start, "case_stop": stop,
              "complete_snapshot": start == 0 and stop == len(cases), "required_checks": len(required_rows),
              "passed_required": len(pass_rows & required_rows), "unavailable": missing, "timed_out": timed_out,
              "manifest_sha256": receipt["files"]["manifest.json"], "checks_sha256": rows_sha,
              "native_performance_qualified": False}
    _write(output/"result.json", result)
    return result


def aggregate(folder, shards):
    """Require exactly one successful validated row for each frozen required check."""
    receipt, cases = verify(folder, return_cases=True)
    needed = {(c.id, backend) for c in cases for backend in required(c, receipt["suite"], receipt["profile"])}
    seen, coverage = set(), set()
    for path in map(Path, shards):
        r = _read(path/"result.json")
        raw = (path/"checks.json").read_bytes()
        if (r.get("schema") != RUN_SCHEMA or r.get("status") != "PASS" or r.get("timed_out") is not False
                or r.get("manifest_sha256") != receipt["files"]["manifest.json"] or r.get("checks_sha256") != _sha(raw)):
            raise ValueError("failed, mismatched or changed correctness shard")
        start, stop = r["case_start"], r["case_stop"]
        if not 0 <= start < stop <= len(cases):
            raise ValueError("invalid shard range")
        indices = set(range(start, stop))
        if coverage & indices:
            raise ValueError("duplicate correctness shard range")
        coverage.update(indices)
        expected = {(c.id, b) for c in cases[start:stop] for b in required(c, receipt["suite"], receipt["profile"])}
        keys = []
        for row in json.loads(raw):
            key = row["case_id"], row["backend"]
            if key in expected and row["passed"] is True:
                keys.append(key)
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError("incomplete or duplicated required checks")
        seen.update(keys)
    return {"status": "PASS" if seen == needed and coverage == set(range(len(cases))) else "FAIL",
            "required_checks": len(needed), "passed_required": len(seen), "cases": len(cases)}
