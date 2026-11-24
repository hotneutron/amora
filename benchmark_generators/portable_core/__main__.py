# SPDX-License-Identifier: Apache-2.0
"""Dedicated workload CLI; does not bypass AMORA's NVIDIA evidence pipeline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from . import _CORE20_SNAPSHOT_SCHEMA, create_backend, describe, evaluate, freeze, make_inputs, oracle, verify_snapshot


def check(folder, output, profile, backends, device):
    import numpy as np
    from .base.backends import BackendUnavailable
    from .base.runner import stamp, verify_manifest, write_new
    receipt = verify_snapshot(folder)
    manifest = verify_manifest(folder / (profile + "_manifest.json"))
    start = time.monotonic()
    rows, unavailable = [], {}
    for name in backends:
        try:
            backend = create_backend(name, device)
        except BackendUnavailable as error:
            unavailable[name] = str(error)
            continue
        for case in manifest["cases"]:
            if time.monotonic() - start > 120:
                raise RuntimeError("bounded correctness check exceeded 120 seconds")
            inputs = make_inputs(case)
            expected = oracle(case, inputs)
            actual = evaluate(case, backend, inputs)
            matrix = any(n["op"] == "matmul" for n in case["nodes"])
            tolerance = receipt["correctness_tolerances"]["matrix" if matrix else "fp32"]
            passed = len(actual) == len(expected)
            for got, want in zip(actual, expected):
                value = backend.numpy(got)
                passed &= bool(value.shape == want.shape and np.all(np.isfinite(value)) and np.allclose(value, want, **tolerance))
            rows.append({"case_id": case["case_id"], "backend": name, "passed": bool(passed)})
    result = {"created_at": stamp(), "status": "PASS" if rows and all(r["passed"] for r in rows) else "FAIL",
              "tier": manifest.get("tier", 20), "cases": len(manifest["cases"]), "checks": len(rows),
              "passed": sum(r["passed"] for r in rows), "failed": sum(not r["passed"] for r in rows),
              "unavailable": unavailable, "device": device, "profile": profile,
              "native_performance_labels_opened": 0,
              "scope": "mathematical correctness only; not native lowering, instruction or performance qualification"}
    write_new(output / "checks.json", rows)
    write_new(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    from .expanded.v2.suites import BlockedSource, PROFILES, SUITES

    listing = sub.add_parser("list")
    materialize = sub.add_parser("freeze")
    for selector in (listing, materialize):
        group = selector.add_mutually_exclusive_group()
        group.add_argument("--tier", type=int, choices=(20, 24, 28))
        group.add_argument("--suite", choices=SUITES)
    materialize.add_argument("--profile", choices=PROFILES)
    materialize.add_argument("--out", type=Path, required=True)
    checking = sub.add_parser("check")
    checking.add_argument("--frozen", type=Path, required=True)
    checking.add_argument("--out", type=Path, required=True)
    checking.add_argument("--profile", choices=("smoke", "native", "reference", "development"))
    checking.add_argument("--suite", choices=SUITES)
    checking.add_argument("--backends")
    checking.add_argument("--device", default="cpu")
    checking.add_argument("--start", type=int)
    checking.add_argument("--stop", type=int)
    checking.add_argument("--timeout", type=float)
    combine = sub.add_parser("aggregate")
    combine.add_argument("--frozen", type=Path, required=True)
    combine.add_argument("--shard", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            result = describe(args.tier or 28, suite=args.suite)
        elif args.command == "freeze":
            result = freeze(args.out, args.tier or 28, suite=args.suite, profile=args.profile)
        elif args.command == "aggregate":
            from .expanded.v2.runner import aggregate
            result = aggregate(args.frozen, args.shard)
        else:
            receipt = json.loads((args.frozen / "freeze_receipt.json").read_text())
            if receipt.get("schema") not in (None, _CORE20_SNAPSHOT_SCHEMA):
                from .expanded.v2.runner import check as check_v2
                result = check_v2(args.frozen, args.out, args.backends.split(",") if args.backends else None,
                                  args.device, start=0 if args.start is None else args.start, stop=args.stop,
                                  timeout=120 if args.timeout is None else args.timeout,
                                  profile=args.profile, suite=args.suite)
            else:
                if (args.suite is not None or args.start is not None or args.stop is not None
                        or args.timeout is not None or args.profile not in (None, "smoke", "native")):
                    raise ValueError("this option requires a v2 snapshot")
                result = check(args.frozen, args.out, args.profile or "smoke",
                               (args.backends or "numpy,torch,jax").split(","), args.device)
    except BlockedSource as error:
        result = {"status": "BLOCKED", "reason": str(error), "native_performance_qualified": False}
    except (OSError, ValueError, RuntimeError) as error:
        result = {"status": "FAIL", "reason": str(error), "native_performance_qualified": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result.get("status") in ("FAIL", "BLOCKED"))


if __name__ == "__main__":
    raise SystemExit(main())
