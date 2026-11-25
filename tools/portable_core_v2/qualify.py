# SPDX-License-Identifier: Apache-2.0
"""Run all source-available v2 release checks in bounded CPU-only shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_generators.portable_core.expanded.v2 import runner
from benchmark_generators.portable_core.expanded.v2.suites import coverage

MATRIX = (("operation-gate-v2", "reference"), ("mechanism-controls", "reference"),
          ("h3-semantics", "development"), ("h3-programs", "development"),
          ("mechanism-controls", "development"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=48)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)
    if not 1 <= args.shard_size <= 256 or not 1 <= args.timeout <= 600:
        parser.error("invalid shard size or timeout")
    # Keep the required matrix independent from optional-backend detection.
    import torch
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    results = {}
    for suite, profile in MATRIX:
        key = f"{suite}-{profile}"
        root = args.out/key
        frozen = root/"frozen"
        receipt = runner.freeze(frozen, suite, profile)
        shards = []
        for start in range(0, receipt["cases"], args.shard_size):
            output = root/f"shard-{start:04d}"
            result = runner.check(frozen, output, start=start,
                                  stop=min(start+args.shard_size, receipt["cases"]), timeout=args.timeout)
            if result["status"] != "PASS":
                raise RuntimeError(f"required correctness shard failed: {output}")
            shards.append(output)
        status = runner.aggregate(frozen, shards)
        if status["status"] != "PASS":
            raise RuntimeError(f"incomplete required matrix: {key}")
        results[key] = status
        print(key, json.dumps(status, sort_keys=True), flush=True)
    result = {"status": "PASS", "scope": "available CPU correctness matrix only", "results": results,
              "exact_h3": {suite: coverage(suite) for suite in ("h3-semantics", "h3-programs")},
              "native_performance_qualified": False}
    (args.out/"release-cpu.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
