# SPDX-License-Identifier: Apache-2.0
"""Source-only consumer preflight using a separately pinned manifest digest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_generators.portable_core.expanded.v2.runner import verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args(argv)
    got = verify(args.frozen)
    if args.manifest_sha256 != got["files"]["manifest.json"]:
        parser.error("manifest does not match the independently pinned digest")
    coverage = json.loads((args.frozen/"coverage.json").read_text())
    crosswalk = coverage["source_crosswalk"]
    print(json.dumps({"status": "PASS", "scope": "source-only preflight; no scientific admission",
                      "suite": got["suite"], "profile": got["profile"], "manifest_sha256": args.manifest_sha256,
                      "case_count": got["cases"], "source_case_count": sum(bool(r["source_case_id"]) for r in crosswalk.values()),
                      "source_coverage": coverage["source_coverage"], "oracle": got["oracle"],
                      "native_performance_qualified": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
