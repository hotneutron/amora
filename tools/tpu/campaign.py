#!/usr/bin/env python3
"""Execute prepared native TPU qualification with immutable per-stage receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import subprocess

from amora.backends.tpu.artifacts import identifier, stamp, validate_manifest, write_json
from amora.backends.tpu.measurement import execute_recipe, load_recipe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id",required=True)
    parser.add_argument("--out-root",type=Path,default=Path("out/tpu"))
    parser.add_argument("--timeout",type=float,default=900)
    parser.add_argument("--skip-portable",action="store_true")
    args=parser.parse_args();name=identifier(args.campaign_id)
    campaign=args.out_root/"campaigns"/name;campaign.mkdir(parents=True,exist_ok=False)
    stages=[]
    def execute(label,recipe):
        run,manifest=execute_recipe(recipe,args.out_root,run_id=f"{name}-{label}",timeout=args.timeout)
        validate_manifest(run/"manifest.json")
        row={"stage":label,"run":str(run),"status":manifest["status"],"run_digest":manifest["run_digest"]}
        stages.append(row);write_json(campaign/(label+".json"),row);print(json.dumps(row),flush=True)
        return manifest["status"]=="passed"
    smoke={"schema_version":1,"recipe_id":"tpu-smoke-v1","cases":[
        {"probe_id":"vector.elementwise_throughput","shape":[128,128],"measure":False},
        {"probe_id":"matrix.matmul_throughput","shape":[128,128,128],"measure":False}]}
    if not execute("smoke",smoke):return 1
    if not args.skip_portable:
        result=subprocess.run([sys.executable,"tools/tpu/qualify.py","--tier","28","--profile","smoke",
            "--campaign-id",name+"-core28","--out-root",str(args.out_root),"--timeout",str(args.timeout)])
        stages.append({"stage":"core28","status":"passed" if result.returncode==0 else "failed",
                       "receipt":str(args.out_root/"campaigns"/(name+"-core28")/"result.json")})
    for label,file in (("single-device","tpu-single-device.json"),("pallas","tpu-pallas.json")):
        execute(label,load_recipe(Path("docs/development")/file))
    collective={"schema_version":1,"recipe_id":"tpu-collective-sweep-v1","batch":16,"cases":[
        {"probe_id":"interconnect.collectives","devices":count,"shape":[payload],"primitive":primitive}
        for count in (2,4,8) for payload in (128,65536) for primitive in ("psum","all_gather","ppermute")]}
    execute("collectives",collective)
    profile={"schema_version":1,"recipe_id":"tpu-profile-v1","capture_profile":True,
             "cases":[{"probe_id":"matrix.matmul_throughput","shape":[256,256,256],"measure":False}]}
    execute("profile",profile)
    result={"completed_at":stamp(),"stages":stages,"passed":all(s["status"]=="passed" for s in stages)}
    write_json(campaign/"result.json",result);print(json.dumps(result),flush=True)
    return 0 if result["passed"] else 1


if __name__=="__main__":raise SystemExit(main())
