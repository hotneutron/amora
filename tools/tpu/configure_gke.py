#!/usr/bin/env python3
"""Record a local TPU instance configuration from its GKE node labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import socket
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("tpu-instance.json"))
    parser.add_argument("--log-dir",type=Path,default=Path("out/tpu/runtime-logs"))
    parser.add_argument("--kubectl",default="/home/kubernetes/bin/kubectl")
    parser.add_argument("--kubeconfig",default="/var/lib/kubelet/kubeconfig")
    args = parser.parse_args()
    host=socket.gethostname()
    result=subprocess.run(["sudo","-n",args.kubectl,"--kubeconfig",args.kubeconfig,
                           "get","node",host,"-o","json"],capture_output=True,text=True,timeout=20,check=True)
    node=json.loads(result.stdout);labels=node["metadata"].get("labels",{})
    topology=labels.get("cloud.google.com/gke-tpu-topology","")
    if not re.fullmatch(r"[1-9][0-9]*(?:x[1-9][0-9]*){1,2}",topology):
        parser.error("node has no recognized TPU topology")
    dims=[int(part) for part in topology.split("x")]
    chips=int(node["status"]["capacity"].get("google.com/tpu",0))
    if math.prod(dims)!=chips:
        parser.error("node topology spans multiple hosts; supply an explicit distributed configuration")
    while len(dims)<3:dims.append(1)
    args.log_dir.mkdir(parents=True,exist_ok=True)
    settings={"TPU_SKIP_MDS_QUERY":"1","TPU_HOST_BOUNDS":"1,1,1",
              "TPU_CHIPS_PER_HOST_BOUNDS":",".join(map(str,dims)),"TPU_TOPOLOGY":topology,
              "TPU_WORKER_ID":"0","TPU_WORKER_HOSTNAMES":"localhost",
              "TPU_LOG_DIR":str(args.log_dir.resolve())}
    record={"hostname":host,"recorded_at":datetime.now(timezone.utc).isoformat(),
            "source":"GKE node metadata","node_uid":node["metadata"]["uid"],
            "node_labels":{k:v for k,v in labels.items() if k in (
                "cloud.google.com/gke-tpu-accelerator","cloud.google.com/gke-tpu-topology",
                "cloud.google.com/gke-accelerator-count")},"physical_chips":chips,"environment":settings}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x") as handle:json.dump(record,handle,indent=2);handle.write("\n")
    print(json.dumps({"instance_config":str(args.out),"hostname":host,"chips":chips,"topology":topology}))
    return 0


if __name__=="__main__":raise SystemExit(main())
