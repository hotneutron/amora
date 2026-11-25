"""Validated recipes and immutable TPU campaign execution."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

from .artifacts import digest, file_hash, finish_manifest, identifier, stamp, validate_manifest, write_json
from .capabilities import device_index


def load_recipe(path):
    return validate_recipe(json.loads(Path(path).read_text()))


def validate_recipe(recipe):
    if not isinstance(recipe, dict) or type(recipe.get("schema_version")) is not int or recipe["schema_version"] != 1:
        raise ValueError("TPU recipe requires schema_version 1")
    identifier(recipe.get("recipe_id"))
    device_index(recipe.get("device", "tpu:0"))
    result = {"device": "tpu:0", "warmup": 5, "samples": 30, "repeats": 3,
              "batch": 1, "max_relative_mad": 0.05, "max_buffer_bytes": 268435456,
              "capture_profile": False, **recipe}
    for key in ("warmup", "samples", "repeats", "batch", "max_buffer_bytes"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "warmup" else 1):
            raise ValueError(f"invalid {key}")
    if not isinstance(result["capture_profile"], bool):
        raise ValueError("capture_profile must be boolean")
    value = result["max_relative_mad"]
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid max_relative_mad")
    if not isinstance(result.get("cases"), list) or not result["cases"]:
        raise ValueError("recipe requires cases")
    from amora.probes.tpu.baseline import PROBES
    for spec in result["cases"]:
        if not isinstance(spec, dict): raise ValueError("case must be an object")
        if "measure" in spec and not isinstance(spec["measure"], bool):
            raise ValueError("measure must be boolean")
        if spec.get("kind") == "portable_core":
            if spec.get("tier", 28) not in (20, 24, 28) or spec.get("profile", "smoke") not in ("smoke", "native"):
                raise ValueError("invalid portable tier/profile")
            for key in ("start", "stop"):
                if key in spec and (isinstance(spec[key], bool) or not isinstance(spec[key], int) or spec[key] < 0):
                    raise ValueError(f"invalid portable {key}")
        elif spec.get("probe_id") not in PROBES:
            raise ValueError("unknown TPU probe")
        for key in ("shape", "tile"):
            if key in spec and (not isinstance(spec[key], list) or not spec[key] or any(
                isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in spec[key])):
                raise ValueError(f"invalid {key}")
        for key in ("iterations", "devices"):
            if key in spec and (isinstance(spec[key], bool) or not isinstance(spec[key], int) or spec[key] < 1):
                raise ValueError(f"invalid {key}")
        if "shape" in spec:
            probe = spec.get("probe_id", "")
            if probe.startswith("matrix.") and len(spec["shape"]) != 3:
                raise ValueError("matrix shape requires M, N, K")
            if probe in ("global_memory.streaming", "local_memory.transfer_sweep") and len(spec["shape"]) != 2:
                raise ValueError("Pallas streaming requires a two-dimensional shape")
        if "direction" in spec and spec["direction"] not in ("host_to_device", "device_to_host"):
            raise ValueError("invalid transfer direction")
        if "primitive" in spec and spec["primitive"] not in ("psum", "all_gather", "ppermute"):
            raise ValueError("invalid collective primitive")
        if "data_pattern" in spec and spec["data_pattern"] not in ("random","dyadic_hadamard"):
            raise ValueError("invalid matrix data pattern")
    return result


def source_identity():
    root = Path(__file__).parents[3]
    sources = {}
    for directory in (root / "amora", root / "benchmark_generators"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json"):
                sources[str(path.relative_to(root))] = file_hash(path)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest = root / "SOURCE_MANIFEST.json"
        commit = json.loads(manifest.read_text()).get("base_commit") if manifest.exists() else None
    return {"commit": commit, "source_files": sources, "source_digest": digest(sources)}


def execute_recipe(recipe, out_root="out/tpu", run_id=None, timeout=900):
    recipe = validate_recipe(recipe)
    run_id = identifier(run_id or (stamp().replace(":", "").replace("+", "-") + "-" + uuid4().hex[:8]))
    root = Path(out_root) / recipe["recipe_id"] / digest(recipe) / run_id
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "recipe.json", recipe)
    write_json(root / "source.json", source_identity())
    for name in ("runtime-lock.txt", "wheel-manifest.json"):
        source = Path(name)
        if source.is_file():
            (root/name).write_bytes(source.read_bytes())
    if os.environ.get("AMORA_TPU_INSTANCE_CONFIG"):
        (root/"instance.json").write_bytes(Path(os.environ["AMORA_TPU_INSTANCE_CONFIG"]).read_bytes())
    from .worker import run_worker
    result = run_worker("measure", {"recipe": recipe, "root": str(root.resolve())}, timeout)
    write_json(root / "worker.json", result)
    manifest = finish_manifest(root, {"backend": "tpu_jax", "recipe_digest": digest(recipe),
                                     "run_id": run_id, "completed_at": stamp(), **result})
    return root, manifest


def execute_in_process(recipe, root):
    root = Path(root)
    recipe = validate_recipe(recipe)
    from .capabilities import discover_in_process
    started = time.perf_counter_ns()
    caps = discover_in_process(recipe["device"], recipe.get("architecture_profile"))
    startup_ns = time.perf_counter_ns() - started
    write_json(root / "capabilities.json", caps)
    if caps["status"] != "ok":
        from amora.schemas.results import ProbeResult, ToolContext
        from amora.reports.json_report import render_report
        reason = ",".join(caps["unsupported_reasons"])
        report = render_report([ProbeResult.unsupported(spec.get("probe_id", "portable_core"),
            reason, backend="tpu_jax", tool_context=ToolContext(tools=caps)) for spec in recipe["cases"]],
            metadata={"backend_capabilities": caps})
        write_json(root / "report.json", report)
        return {"status": "failed", "reason": reason}
    import jax
    from .compiler import compile_program
    from .portable_core import check_outputs, prepare, program, verify_semantics
    from .probes import build_probe
    from .runner import drain, time_program
    from benchmark_generators import portable_core as portable
    semantics = verify_semantics()
    write_json(root / "semantic-verification.json", semantics)
    if not semantics["passed"]:
        raise ValueError("semantic fingerprint mismatch")
    device = jax.local_devices(backend="tpu")[device_index(recipe["device"])]
    rows, samples, compiler_artifacts = [], [], {}
    expanded = []
    for spec in recipe["cases"]:
        if spec.get("kind") == "portable_core":
            cases = portable.generate_cases(spec.get("tier", 28), spec.get("profile", "smoke"))
            start, stop = spec.get("start", 0), spec.get("stop", len(cases))
            if not 0 <= start < stop <= len(cases): raise ValueError("invalid portable shard bounds")
            expanded.extend((spec, case) for case in cases[start:stop])
        else:
            expanded.append((spec, None))
    for index, (spec, case) in enumerate(expanded):
        key = case["case_id"] if case else spec["probe_id"] + "/" + str(index)
        row = {"case_id": key, "probe_id": "portable_core" if case else spec["probe_id"],
               "status": "failed", "spec": spec}
        try:
            staged_at = time.perf_counter_ns()
            if case:
                requested = sum(math.prod(s["shape"])*4 for s in case["inputs"].values())
                if requested > recipe["max_buffer_bytes"]: raise ValueError("case exceeds buffer budget")
                args, expected, input_digest = prepare(case, device)
                function = program(case)
                types = {**case["inputs"], **{n["id"]: n for n in case["nodes"]}}
                output_dtypes = [types[name]["dtype"] for name in case.get("outputs", [case["output"]])]
                metadata = {"mode": "jax_jit", "case_sha256": case["sha256"],
                            "input_digest": input_digest,
                            "matrix": any(n["op"] == "matmul" for n in case["nodes"])}
            else:
                shape = spec.get("shape", [256, 256, 256] if "matrix." in spec["probe_id"] else [128, 128])
                requested = (sum(shape[i]*shape[(i+1)%3] for i in range(3))*4
                             if "matrix." in spec["probe_id"] else math.prod(shape)*8)
                if spec["probe_id"] == "interconnect.collectives":
                    requested *= spec.get("devices", len(jax.local_devices(backend="tpu")))
                if requested > recipe["max_buffer_bytes"]: raise ValueError("case exceeds buffer budget")
                if spec["probe_id"] == "topology.device_attributes":
                    row.update(status="passed", metadata=caps["target"])
                    rows.append(row)
                    continue
                function, args, expected, metadata = build_probe(spec["probe_id"], spec, device)
                output_dtypes = None
            metadata["staging_ns"] = time.perf_counter_ns() - staged_at
            directory = root / "compiler" / f"case-{index:04d}"
            if metadata["mode"] == "transfer":
                invoke = function
            else:
                executable, compiled = compile_program(function, args, directory)
                metadata.update(compiled)
                compiler_artifacts[key] = {kind: {**item, "path": str((directory/item["path"]).relative_to(root))}
                                           for kind, item in compiled["compiler_artifacts"].items()}
                invoke = lambda: executable(*args)
            first_at = time.perf_counter_ns()
            actual = invoke()
            drain(actual)
            metadata["first_execution_ns"] = time.perf_counter_ns() - first_at
            correctness = check_outputs(actual, expected, matrix=metadata.get("matrix", False),
                                        output_dtypes=output_dtypes)
            row.update(metadata=metadata, correctness=correctness)
            if not correctness["passed"]:
                row["reason"] = "correctness_failed"
            else:
                row["status"] = "passed"
                if spec.get("measure", True):
                    timing = time_program(invoke, **{key: recipe[key] for key in
                        ("warmup", "samples", "repeats", "batch", "max_relative_mad")})
                    samples.extend({"case_id": row["case_id"], **item} for item in timing.pop("samples"))
                    row["timing"] = timing
                    if not timing["stable"] and (case is not None or spec.get("probe_id") != "runtime.dispatch_latency"):
                        row["status"] = "inconclusive"
                    duration = timing["median_ns_per_invocation"] / 1e9
                    for quantity, field in (("logical_flops", "effective_flops_per_second"),
                                            ("requested_bytes", "effective_bytes_per_second")):
                        if quantity in metadata: row[field] = metadata[quantity] / duration
                if recipe["capture_profile"]:
                    from .profiling import capture
                    row["profiling"] = capture(invoke, root / "profile" / f"case-{index:04d}")
        except (ImportError, AttributeError, NotImplementedError) as error:
            row["status"] = "unsupported"
            row["reason"] = "feature_unavailable: " + str(error)
        except Exception as error:
            row["reason"] = type(error).__name__ + ": " + str(error)
        rows.append(row)
        # Persist each completed case before continuing a bounded campaign.
        write_json(root / "cases" / f"case-{index:04d}.json", row)
    write_json(root / "correctness.json", {"results": rows, "cases_expected": len(expanded),
                                           "passed": sum(r["status"] == "passed" for r in rows)})
    with (root / "samples.jsonl").open("x") as handle:
        for sample in samples: handle.write(json.dumps(sample, sort_keys=True) + "\n")
    failed = any(r["status"] == "failed" for r in rows)
    inconclusive = any(r["status"] in ("inconclusive", "unsupported") for r in rows)
    runtime_locked = bool(caps["runtime"].get("runtime_lock_sha256") and
                          caps["runtime"].get("runtime_lock_versions_verified"))
    inconclusive |= not runtime_locked
    from .ownership import snapshot
    final_ownership = snapshot(exclude_pids=(os.getpid(),))
    write_json(root / "ownership-final.json", final_ownership)
    if final_ownership["state"] != "exclusive":
        failed = True
    status = "failed" if failed else "inconclusive" if inconclusive else "passed"
    from .reporting import render_results
    report = render_results(rows, caps, recipe, compiler_artifacts)
    report["metadata"]["qualification"] = {"status":status,"runtime_locked":runtime_locked,
        "ownership_final":final_ownership["state"]}
    write_json(root / "report.json", report)
    return {"status": status, "runtime_locked":runtime_locked,
            "ownership_qualified": final_ownership["state"] == "exclusive",
            "cases": len(rows), "initialization_ns": startup_ns,
            "target_config_digest": caps["target"]["target_config_digest"],
            "runtime_digest": caps["runtime"]["runtime_digest"],
            "architecture": caps["architecture"]["generation"],
            "compiler_artifacts": compiler_artifacts}


def aggregate_portable_runs(paths, tier=28, profile="smoke"):
    from benchmark_generators import portable_core as core
    expected = {case["case_id"] for case in core.generate_cases(tier, profile)}
    records, identities = {}, set()
    for path in paths:
        manifest = validate_manifest(Path(path) / "manifest.json")
        identities.add((manifest.get("target_config_digest"), manifest.get("runtime_digest")))
        for row in json.loads((Path(path)/"correctness.json").read_text())["results"]:
            if row["probe_id"] == "portable_core":
                if row["case_id"] in records: raise ValueError("duplicate portable case in aggregation")
                records[row["case_id"]] = row
    complete = set(records) == expected and len(identities) == 1
    return {"tier": tier, "profile": profile, "expected": len(expected), "observed": len(records),
            "missing": sorted(expected-records.keys()), "unexpected": sorted(records.keys()-expected),
            "complete": complete, "passed": complete and all(r["status"] == "passed" for r in records.values()),
            "runs": [str(p) for p in paths], "case_statuses": {key: r["status"] for key,r in records.items()}}
