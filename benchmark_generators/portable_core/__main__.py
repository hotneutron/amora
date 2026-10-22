# SPDX-License-Identifier: Apache-2.0
"""Dedicated workload CLI; does not bypass AMORA's NVIDIA evidence pipeline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from . import create_backend, describe, evaluate, freeze, make_inputs, oracle, verify_snapshot


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
    listing = sub.add_parser("list")
    listing.add_argument("--tier", type=int, choices=(20, 24, 28), default=28)
    materialize = sub.add_parser("freeze")
    materialize.add_argument("--tier", type=int, choices=(20, 24, 28), default=28)
    materialize.add_argument("--out", type=Path, required=True)
    checking = sub.add_parser("check")
    checking.add_argument("--frozen", type=Path, required=True)
    checking.add_argument("--out", type=Path, required=True)
    checking.add_argument("--profile", choices=("smoke", "native"), default="smoke")
    checking.add_argument("--backends", default="numpy,torch,jax")
    checking.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.command == "list":
        result = describe(args.tier)
    elif args.command == "freeze":
        result = freeze(args.out, args.tier)
    else:
        result = check(args.frozen, args.out, args.profile, args.backends.split(","), args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result.get("status") == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
