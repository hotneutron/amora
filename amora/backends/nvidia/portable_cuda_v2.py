# SPDX-License-Identifier: Apache-2.0
"""Standalone, hash-bound CUDA realization of portable_core v2 programs."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import subprocess
from uuid import uuid4

from amora.benchmarking.portable_v2.corpus import OPS, REALIZATION, sha, verify_corpus, write_json
from amora.backends.flashgpu_sim.runner import capture, run_test

SOURCE = Path(__file__).with_name("portable_core_v2.cu")
FLAGS = ("-std=c++17", "-O2", "--fmad=false", "-lineinfo", "-cudart", "shared",
         "-gencode", "arch=compute_90a,code=sm_90a",
         "-gencode", "arch=compute_90a,code=compute_90a")

def build_driver(mask, *, build_root, evidence_root, cuda_root):
    cuda_root = Path(cuda_root).resolve()
    nvcc = cuda_root / "bin/nvcc"
    version = subprocess.check_output([str(nvcc), "--version"], text=True)
    contract = dict(realization=REALIZATION, source_sha256=sha(SOURCE),
                    operator_mask=int(mask), flags=list(FLAGS), compiler_version=version)
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    folder = Path(build_root).resolve() / (f"{int(mask):08x}-" + digest[:16])
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / "portable-v2"
    receipt_path = folder / "build.json"
    if receipt_path.exists() and binary.exists():
        old = json.loads(receipt_path.read_text())
        if old["binary_sha256"] == sha(binary):
            return old
    command = [str(nvcc), *FLAGS, "-DAMORA_OP_MASK=" + str(mask) + "ULL",
               str(SOURCE), "-o", str(binary)]
    receipt = capture(command, label="v2-build-" + digest[:12] + "-" + uuid4().hex[:6],
                      evidence_root=evidence_root, cwd=SOURCE.parent, timeout=300,
                      metadata=dict(kind="build", contract=contract))
    if receipt["exit_code"] != 0:
        raise RuntimeError("CUDA compilation failed: " + receipt["output"])
    result = dict(contract=contract, realization_sha256=digest, binary=str(binary),
                  binary_sha256=sha(binary), build_receipt=receipt["label"])
    receipt_path.write_text(json.dumps(result, indent=2) + "\n")
    return result

def pilot_indices(manifest):
    selected = set(manifest["pilot_indices"])
    all_ops = {n["op"] for c in manifest["cases"] for n in c["program"]["nodes"]}
    have = {n["op"] for c in manifest["cases"] if c["index"] in selected for n in c["program"]["nodes"]}
    for op in sorted(all_ops-have):
        selected.add(next(c["index"] for c in manifest["cases"] if any(n["op"] == op for n in c["program"]["nodes"])))
    for predicate in (
        lambda c: any(x["row_padding"] for x in c["program"]["inputs"]),
        lambda c: any(x["offset"] for x in c["program"]["inputs"]),
        lambda c: bool(c["program"]["state_edges"]),
        lambda c: "collide" in c["id"] and "scatter" in c["id"],
    ):
        found = next((c["index"] for c in manifest["cases"] if predicate(c)), None)
        if found is not None:
            selected.add(found)
    return sorted(selected)

def run_group(indices, *, manifest, builds, corpus_root, evidence_root, sim_root,
              cuda_root, mode, label, gpu="7", samples=0, warmup=3,
              timeout=1200, omp_threads=2, overrides=(), scratch_root=None):
    root = Path(evidence_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not indices:
        raise ValueError("empty selection")
    mask = manifest["cases"][indices[0]]["operator_mask"]
    build = builds.get(str(mask), builds.get("all"))
    if build is None:
        raise ValueError("no compiled driver for selection")
    compiled_mask = build["contract"]["operator_mask"]
    if any(manifest["cases"][i]["operator_mask"] & ~compiled_mask for i in indices):
        raise ValueError("case uses an operator outside compiled set")
    selected_file = root / (label + ".indices")
    with selected_file.open("x") as stream:
        stream.write("\n".join(map(str, indices)) + "\n")
    result_file = root / (label + ".cases.jsonl")
    args = ["--corpus=" + str(Path(corpus_root).resolve()), "--result=" + str(result_file),
            "--selection=" + str(selected_file), "--device=0",
            "--samples=" + str(samples), "--warmup=" + str(warmup)]
    functional = mode == "ptx-functional"
    preset = "SM90_H100_SASS_FRONTEND" if mode == "sass-timing" else "SM90_H100"
    receipt = run_test(binary=build["binary"], sim_root=sim_root, cuda_root=cuda_root,
                       evidence_root=root, label=label, preset=preset, mode="ptx" if functional else mode,
                       timeout=timeout, gpu=gpu, omp_threads=omp_threads,
                       overrides=overrides, scratch_root=scratch_root,
                       environment_overrides={"PTX_SIM_MODE_FUNC":"1"} if functional else None,
                       program_arguments=args)
    if receipt["metadata"]["binary_sha256"] != build["binary_sha256"]:
        raise ValueError("executed binary changed from the admitted build")
    rows = []
    if result_file.exists():
        for line in result_file.read_text().splitlines():
            row = json.loads(line)
            expected = manifest["cases"][row["index"]]
            if row["index"] not in indices or row["case_id"] != expected["id"]:
                raise ValueError("driver result outside frozen selection")
            if type(row.get("passed")) is not bool:
                raise ValueError("invalid correctness verdict")
            if row["node_ports_checked"] != sum(not x["input"] for x in expected["bindings"]):
                raise ValueError("missing node-port checks")
            if not row["passed"] and row.get("samples_ms"):
                raise ValueError("failed correctness cannot publish timing")
            if any(not math.isfinite(v) or v<0 for v in row.get("samples_ms", [])):
                raise ValueError("invalid program span")
            row.update(realization_sha256=build["realization_sha256"],
                       executable_sha256=build["binary_sha256"], mode=mode,
                       case_sha256=expected["case_sha256"], program_sha256=expected["program_sha256"])
            if functional:
                row["samples_ms"] = []
                row["timing_available"] = False
            rows.append(row)
    if len({r["index"] for r in rows}) != len(rows):
        raise ValueError("duplicate driver result")
    if mode == "native":
        log = (root / receipt["output"]).read_text(errors="replace")
        device_lines = [line for line in log.splitlines() if line.startswith("AMORA_V2_DEVICE ")]
        if not device_lines:
            if receipt["exit_code"] == 0:
                raise ValueError("missing native placement evidence")
        else:
            device = json.loads(device_lines[0].split(" ", 1)[1])
            expected_uuid = subprocess.check_output(
                ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
            if device["uuid"].lower() != expected_uuid.lower() or device["sm"] != 9:
                raise ValueError("native CUDA placement does not match selected H100")
            receipt["native_device"] = device
    seen = {r["index"] for r in rows}
    for index in indices:
        if index not in seen:
            c = manifest["cases"][index]
            rows.append(dict(index=index, case_id=c["id"], case_sha256=c["case_sha256"],
                             program_sha256=c["program_sha256"], passed=False,
                             status="not_completed", exit_code=receipt["exit_code"], mode=mode,
                             samples_ms=[], realization_sha256=build["realization_sha256"],
                             executable_sha256=build["binary_sha256"]))
    result = dict(label=label, receipt=receipt, rows=rows, selection=indices)
    write_json(root / (label + ".validation.json"), result)
    return result

