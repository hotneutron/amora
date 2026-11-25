"""Bind native TPU V2 correctness receipts to a frozen host-latency campaign."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import median

from amora.benchmarking.portable_v2.contracts import prepare_case, compare
from amora.benchmarking.portable_v2.runner import _source_hashes
from benchmark_generators.portable_core.expanded.v2.runner import verify
from benchmark_generators.portable_core.expanded.v2.spec import digest

SCHEMA = "amora-v2-tpu-latency-v1"
DOMAIN = "synchronized_host_dispatch_to_ready_ns"
PROTOCOL = {"warmup": 20, "samples": 30, "repeats": 3, "batch": 1, "max_relative_mad": 0.05}
MIN_MEDIAN_NS = 1000
HLO_IDENTITY = "optimized_hlo_except_source_stack_tables_v1"


def optimized_hlo_identity(text):
    # Recompilation changes call-site stack tables and numeric stack_frame_id
    # metadata. Keep instructions, ordering, layouts, and backend configs exact.
    body = re.sub(r"(?m)^(?:FileNames|FunctionNames|FileLocations|StackFrames)\n(?:[0-9]+ [^\n]*\n)+\n?", "", text)
    body = re.sub(r"\bstack_frame_id=\d+", "stack_frame_id=0", body)
    return hashlib.sha256(body.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(root):
    evidence = root / "evidence"
    pins, summary, verification = (read(root / "full-pins.json"), read(evidence / "full-summary.json"),
                                    read(root / "tpu-verification.json"))
    pins_sha = sha(root / "full-pins.json")
    if not (verification["complete"] and summary["complete"] and verification["pins_sha256"] == pins_sha
            and summary["source_manifest_sha256"] == pins_sha and not verification["mismatches"]):
        raise ValueError("missing verified full correctness campaign")
    verified = {}
    for line in (evidence / "artifact-export/files.sha256").read_text().splitlines():
        h, path = line.split("  ", 1)
        if path in verified or sha(evidence / path) != h:
            raise ValueError(f"wrong archived file hash: {path}")
        verified[path] = h
    cohort, targets, sources = {}, set(), {}
    for shard in summary["shards"]:
        path = evidence / shard["receipt"]
        r = read(path)
        if (sha(path) != shard["receipt_sha256"] or r["manifest_sha256"] != pins["suites"][shard["suite"]]["manifest_sha256"]
                or r["backend"] != "tpu_jax" or r["cpu_fallback"] or r["native_performance_qualified"]
                or shard["ownership_after_exit"]["state"] != "exclusive"):
            raise ValueError("incompatible correctness shard")
        for c in r["cases"]:
            key = c["case_id"]
            bound = pins["suites"][shard["suite"]]["cases"][key]
            if key in cohort or c["case_sha256"] != bound["case_sha256"]:
                raise ValueError("invalid case crosswalk")
            row = dict(case_id=key, case_sha256=c["case_sha256"], suite=shard["suite"],
                       manifest_sha256=r["manifest_sha256"], correctness_receipt_sha256=sha(path),
                       status="needs_measurement" if c["qualified"] else "not_admissible", reason=c.get("reason"))
            if c["qualified"]:
                if bound["expected"] != "passed" or bound["input_sha256"] != c["binding"]["input_sha256"]:
                    raise ValueError("frozen inputs differ from observed correctness inputs")
                target = r["hardware"]["target"]
                key_sha = r["hardware"]["target_sha256"]
                if digest(target) != key_sha or not all(x["passed"] for x in r["controls"].values()):
                    raise ValueError("invalid target or boundary witness")
                targets.add(key_sha)
                row.update(program_sha256=c["program_sha256"], input_sha256=c["binding"]["input_sha256"],
                           realization_sha256=c["realization_sha256"], target_sha256=key_sha,
                           generation=target["generation"], runtime_sha256=target["runtime"]["runtime_digest"])
                for kind in ("stablehlo", "optimized_hlo"):
                    ir = c["compiler"]["ir"]["output"][kind]
                    file = path.parent / c["artifact_directory"] / ir["path"]
                    if verified[str(file.relative_to(evidence))] != ir["sha256"]:
                        raise ValueError("wrong IR hash")
                    row[kind + "_sha256"] = ir["sha256"]
                    row[kind + "_path"] = str(file.relative_to(evidence))
                    if kind == "optimized_hlo":
                        row["optimized_hlo_identity_sha256"] = optimized_hlo_identity(file.read_text())
                if sources and sources != r["adapter_sources_sha256"]:
                    raise ValueError("multiple adapter versions")
                sources = r["adapter_sources_sha256"]
            elif c.get("reason") != bound["expected"]:
                raise ValueError("unexpected admission state in correctness evidence")
            cohort[key] = row
    if len(cohort) != pins["count"] or len(targets) != 1:
        raise ValueError("incomplete or mixed-target corpus")
    value = {"schema": SCHEMA, "label_domain": DOMAIN, "protocol": PROTOCOL, "minimum_median_ns": MIN_MEDIAN_NS,
             "pins_sha256": pins_sha, "source_hashes": sources, "hlo_identity": HLO_IDENTITY, "rows": list(cohort.values())}
    return dict(value, plan_sha256=digest(value))


def checked(plan_file):
    value = read(plan_file)
    key = value.pop("plan_sha256")
    if (key != digest(value) or value["schema"] != SCHEMA or value["label_domain"] != DOMAIN
            or value["protocol"] != PROTOCOL or value["minimum_median_ns"] != MIN_MEDIAN_NS
            or value.get("hlo_identity") != HLO_IDENTITY):
        raise ValueError("latency plan changed or is incompatible")
    return dict(value, plan_sha256=key)


def collect(plan_file, frozen_root, suite, start, stop, out, device):
    from amora.backends.tpu.portable_core_v2 import Target, build_program
    from amora.backends.tpu.runner import time_program
    from benchmark_generators.portable_core.expanded.v2.reference import oracle
    import numpy as np

    spec = checked(plan_file)
    if spec["source_hashes"] != _source_hashes("tpu_jax"):
        raise ValueError("timing executor must reuse the correctness adapter source")
    rows = [r for r in spec["rows"] if r["suite"] == suite and r["status"] == "needs_measurement"]
    if not 0 <= start < stop <= len(rows) or stop - start > 32 or out.exists():
        raise ValueError("specify a new output and a bounded, existing shard (max 32)")
    selected = rows[start:stop]
    receipt, cases = verify(frozen_root / suite, return_cases=True)
    by_id = {c.id: c for c in cases}
    target = Target(device, expected_generation=selected[0]["generation"])
    if any(r["target_sha256"] != digest(target.identity) for r in selected):
        raise ValueError("target/runtime differs from the admitted realization")
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for number, r in enumerate(selected):
        c = by_id[r["case_id"]]
        row = {"case_id": c.id, "case_sha256": c.case_sha256, "status": "not_measured"}
        if r["manifest_sha256"] != receipt["files"]["manifest.json"] or r["case_sha256"] != c.case_sha256:
            raise ValueError("frozen case identity drift")
        try:
            types, raw, binding = prepare_case(c)
            if binding["input_sha256"] != r["input_sha256"]:
                raise ValueError("input bytes differ from successful correctness run")
            args = tuple(target.jax.device_put(raw[p.id], target.device) for p in c.program.inputs)
            lowered = target.jax.jit(build_program(c, types, target.math)).lower(*args)
            executable = lowered.compile()
            evidence = {"stablehlo": str(lowered.compiler_ir("stablehlo")), "optimized_hlo": executable.as_text()}
            hashes = {kind: hashlib.sha256(text.encode()).hexdigest() for kind, text in evidence.items() if text}
            row["recompiled_ir_sha256"] = hashes
            if hashes.get("stablehlo") != r["stablehlo_sha256"] or not evidence["optimized_hlo"]:
                raise ValueError("StableHLO identity differs or optimized HLO missing")
            h = optimized_hlo_identity(evidence["optimized_hlo"])
            if h != r["optimized_hlo_identity_sha256"]:
                raise ValueError("optimized HLO differs beyond source-stack metadata")
            row["optimized_hlo_identity_sha256"] = h
            want = oracle(c, raw)

            def invoke():
                return executable(*args)

            def correctness():
                vals = invoke()
                if any(v.devices() != {target.device} for v in vals):
                    raise ValueError("output placement differs from admitted TPU")
                if error := compare(tuple(map(np.asarray, vals)), want, [False]*len(want)):
                    raise ValueError(error)

            correctness()
            timing = time_program(invoke, **PROTOCOL)
            correctness()
            median = timing["median_ns_per_invocation"]
            ok = timing["stable"] and median >= MIN_MEDIAN_NS
            row.update(status="labelled" if ok else "unstable", timing=timing,
                       realization_sha256=r["realization_sha256"], input_sha256=binding["input_sha256"])
            if ok:
                row["label_ns"] = median
        except (ValueError, RuntimeError, TypeError) as error:
            row.update(reason=str(error))
        write(out / f"case-{number:02d}.json", row)
        records.append(row)
    closeout, valid = target.finish()
    write(out / "shard.json", {"plan_sha256": spec["plan_sha256"], "suite": suite, "start": start, "stop": stop,
                               "target_sha256": digest(target.identity), "ownership": closeout,
                               "ownership_valid": valid, "rows": records})
    return int(not valid or any(r["status"] != "labelled" for r in records))


def validated_median(timing):
    """Recompute the gate from raw samples; never accept the worker flag alone."""
    if (timing["warmup_iterations"] != PROTOCOL["warmup"] or
            timing["measured_samples_per_repeat"] != PROTOCOL["samples"] or
            timing["repeats"] != PROTOCOL["repeats"] or timing["batch_size"] != PROTOCOL["batch"] or
            timing["timed_unit"] != "single_invocation"):
        raise ValueError("timing repeats, warmup or batch differ from the frozen plan")
    grid = {(repeat, sample) for repeat in range(PROTOCOL["repeats"]) for sample in range(PROTOCOL["samples"])}
    values = {}
    for row in timing["samples"]:
        key = (row["repeat"], row["sample"])
        ns = row["dispatch_to_completion_ns"]
        if key not in grid or key in values or type(ns) is not int or ns <= 0 or row["invocations"] != 1:
            raise ValueError("duplicate, invalid or non-singleton timing sample")
        values[key] = ns
    if set(values) != grid:
        raise ValueError("missing timing sample")

    def centre(series):
        middle = median(series)
        if median(abs(x-middle) for x in series) > PROTOCOL["max_relative_mad"]*middle:
            raise ValueError("raw timing samples fail relative MAD")
        return middle

    return centre([centre([values[(r, s)] for s in range(PROTOCOL["samples"])])
                   for r in range(PROTOCOL["repeats"])])


def assemble(spec, shard_paths):
    expected = {r["case_id"]: r.copy() for r in spec["rows"]}
    seen = set()
    for path in shard_paths:
        s = read(path)
        if s["plan_sha256"] != spec["plan_sha256"] or s["ownership_valid"] is not True:
            raise ValueError("wrong timing plan or failed ownership gate")
        eligible = [r["case_id"] for r in spec["rows"] if r["suite"] == s["suite"] and r["status"] == "needs_measurement"]
        if (not 0 <= s["start"] < s["stop"] <= len(eligible) or
                [r["case_id"] for r in s["rows"]] != eligible[s["start"]:s["stop"]]):
            raise ValueError("missing or wrong shard case IDs")
        for r in s["rows"]:
            key = r["case_id"]
            e = expected[key]
            if (key in seen or e["case_sha256"] != r["case_sha256"] or e["target_sha256"] != s["target_sha256"]
                    or e["status"] != "needs_measurement"):
                raise ValueError("duplicate, mismatched or non-admissible latency row")
            if r["status"] == "labelled":
                t = r["timing"]
                if (r["realization_sha256"] != e["realization_sha256"] or r["input_sha256"] != e["input_sha256"]
                        or r.get("optimized_hlo_identity_sha256") != e["optimized_hlo_identity_sha256"]
                        or r.get("recompiled_ir_sha256", {}).get("stablehlo") != e["stablehlo_sha256"]
                        or not t["stable"] or t["time_domain"] != "synchronized_host"
                        or r["label_ns"] != t["median_ns_per_invocation"]
                        or r["label_ns"] != validated_median(t) or r["label_ns"] < MIN_MEDIAN_NS):
                    raise ValueError("incompatible or unstable latency label")
                e["label_ns"] = r["label_ns"]
            elif r["status"] not in ("not_measured", "unstable") or "label_ns" in r:
                raise ValueError("invalid latency status")
            e.update(status=r["status"], timing_shard_sha256=sha(path))
            seen.add(key)
    counts = Counter(r["status"] for r in expected.values())
    return {"schema": SCHEMA, "label_domain": DOMAIN, "protocol": PROTOCOL, "plan_sha256": spec["plan_sha256"],
            "counts": dict(counts), "complete": not any(counts[k] for k in ("needs_measurement", "unstable", "not_measured")),
            "rows": list(expected.values())}


def recover(spec, original, retry):
    """Merge at most one extra attempt, only for cases rejected as unstable."""
    first, second = assemble(spec, original), assemble(spec, retry)
    by_id = {r["case_id"]: r for r in first["rows"]}
    recovered = 0
    retried = 0
    for r in second["rows"]:
        if "timing_shard_sha256" not in r:
            continue
        prev = by_id[r["case_id"]]
        if prev["status"] != "unstable" or "label_ns" in prev:
            raise ValueError("retry may only replace a rejected unstable measurement")
        by_id[r["case_id"]] = {**r, "original_unstable_shard_sha256": prev["timing_shard_sha256"]}
        recovered += int(r["status"] == "labelled")
        retried += 1
    counts = Counter(r["status"] for r in by_id.values())
    return {**first, "counts": dict(counts), "complete": not any(counts[k] for k in (
        "needs_measurement", "unstable", "not_measured")), "rows": list(by_id.values()),
        "recovery": {"policy": "same protocol; one extra attempt for unstable rows only",
                     "attempted": retried, "labelled": recovered}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="action", required=True)
    prep = subs.add_parser("plan")
    prep.add_argument("--artifacts", type=Path, required=True)
    run = subs.add_parser("run")
    run.add_argument("--frozen-root", type=Path, required=True)
    run.add_argument("--suite", required=True)
    run.add_argument("--start", type=int, required=True)
    run.add_argument("--stop", type=int, required=True)
    run.add_argument("--device", default="tpu:0")
    join = subs.add_parser("assemble")
    join.add_argument("--shard", type=Path, action="append", default=[])
    join.add_argument("--retry-shard", type=Path, action="append", default=[])
    for action in (run, join):
        action.add_argument("--plan", type=Path, required=True)
    for action in (prep, run, join):
        action.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.action == "run":
        return collect(a.plan, a.frozen_root, a.suite, a.start, a.stop, a.out, a.device)
    if a.action == "plan":
        value = plan(a.artifacts)
    else:
        spec = checked(a.plan)
        value = recover(spec, a.shard, a.retry_shard) if a.retry_shard else assemble(spec, a.shard)
    write(a.out, value)
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
