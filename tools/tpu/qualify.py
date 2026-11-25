#!/usr/bin/env python3
"""Bounded portable-core TPU shards, immutable runs and complete coverage receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from amora.backends.tpu.artifacts import identifier, stamp, write_json
from amora.backends.tpu.measurement import aggregate_portable_runs, execute_recipe
from benchmark_generators import portable_core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=int, choices=(20,24,28), default=28)
    parser.add_argument("--profile", choices=("smoke","native"), default="smoke")
    parser.add_argument("--device", default="tpu:0")
    parser.add_argument("--out-root", type=Path, default=Path("out/tpu"))
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.shard_size < 1: parser.error("shard-size must be positive")
    campaign = args.out_root / "campaigns" / identifier(args.campaign_id)
    campaign.mkdir(parents=True, exist_ok=False)
    cases = portable_core.generate_cases(args.tier,args.profile)
    runs = []
    for index,start in enumerate(range(0,len(cases),args.shard_size)):
        recipe = {"schema_version":1,"recipe_id":"portable-core-qualification-v1","device":args.device,
                  "cases":[{"kind":"portable_core","tier":args.tier,"profile":args.profile,
                            "start":start,"stop":min(len(cases),start+args.shard_size),"measure":False}]}
        run,manifest = execute_recipe(recipe,args.out_root,run_id=f"{args.campaign_id}-{index:03d}",timeout=args.timeout)
        print(json.dumps({"shard":index,"run":str(run),"status":manifest["status"]}),flush=True)
        runs.append(run)
        if manifest["status"] != "passed":
            write_json(campaign/"result.json",{"status":"failed","completed_at":stamp(),
                "runs":[str(path) for path in runs],"failed_shard":index,"reason":manifest.get("reason")})
            return 1
    result = aggregate_portable_runs(runs,args.tier,args.profile)
    result["completed_at"] = stamp()
    write_json(campaign/"result.json",result)
    print(json.dumps(result),flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
