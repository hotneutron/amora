"""Build independent feature probes with source and toolchain keyed artifacts."""
from pathlib import Path
import json
import hashlib
import subprocess
from uuid import uuid4

from amora.probes.nvidia.feature_validation import ROOT, PROBES
from .runner import capture, sha256

def build_probe(probe_id, *, sim_root, cuda_root, build_root, evidence_root):
    if probe_id not in PROBES:
        raise ValueError("unknown probe " + probe_id)
    entry = PROBES[probe_id]
    sim_root, cuda_root = Path(sim_root).resolve(), Path(cuda_root).resolve()
    source = ROOT / entry["source"]
    contract = dict(probe_id=probe_id, source_sha256=sha256(source), arch=entry["arch"])
    if probe_id == "tmem-capacity":
        compiler = "g++"
        support = sim_root / "src/gpgpu-sim/flash/tcgen05/tmem.cc"
        contract["support_sha256"] = sha256(support)
        contract["support_header_sha256"] = sha256(support.with_suffix(".h"))
    else:
        compiler = str(cuda_root / "bin/nvcc")
        gtest = sim_root / "tests/third_party/gtest/googletest"
        library = sim_root / "tests/build/obj/gtest_main.a"
        if not (gtest / "include/gtest/gtest.h").is_file() or not library.is_file():
            raise FileNotFoundError("Build FlashGPU-Sim's test GoogleTest library first")
        contract["gtest_sha256"] = sha256(library)
    contract["compiler_version"] = subprocess.check_output(
        [compiler, "--version"], text=True)
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(build_root).resolve() / (probe_id + "-" + digest)
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / probe_id
    manifest = directory / "build.json"
    if manifest.is_file() and binary.is_file():
        previous = json.loads(manifest.read_text())
        if previous.get("binary_sha256") == sha256(binary):
            return previous
    if probe_id == "tmem-capacity":
        command = [compiler, "-std=c++17", "-O2", "-I" + str(sim_root / "src"),
                   str(source), str(support), "-o", str(binary)]
    else:
        arch = entry["arch"]
        compute = arch.replace("sm_", "compute_")
        command = [compiler, "-std=c++17", "-O2", "-cudart", "shared",
                   "-gencode", "arch=" + compute + ",code=" + arch,
                   "-gencode", "arch=" + compute + ",code=" + compute,
                   "-I" + str(gtest / "include"), str(source), str(library),
                   "-L" + str(cuda_root / "lib64"), "-lcuda", "-Xcompiler",
                   "-pthread", "-o", str(binary)]
    receipt = capture(command, label="build-" + probe_id + "-" + digest + "-" + uuid4().hex[:8],
                      evidence_root=evidence_root, cwd=sim_root, timeout=600,
                      metadata=dict(kind="build", contract=contract))
    if receipt["exit_code"] != 0:
        raise RuntimeError("probe build failed; see " + receipt["output"])
    result = dict(contract=contract, binary=str(binary), binary_sha256=sha256(binary))
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    return result

