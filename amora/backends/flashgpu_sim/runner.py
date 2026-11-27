"""Capture bounded command/test runs with caller-owned evidence and exact provenance."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import atexit
import time
import xml.etree.ElementTree as ET

LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_GPU_GUARD_LOCK = threading.Lock()
_GPU_GUARD_CACHE = (0.0, ())
_GPU_GUARD_INITIALIZED = False

def _gpu_context_pids():
    """Read process IDs only; this does not initialize a CUDA context."""
    global _GPU_GUARD_CACHE, _GPU_GUARD_INITIALIZED
    with _GPU_GUARD_LOCK:
        now = time.monotonic()
        if now - _GPU_GUARD_CACHE[0] < .5:
            return _GPU_GUARD_CACHE[1]
        import pynvml as nv
        if not _GPU_GUARD_INITIALIZED:
            nv.nvmlInit()
            atexit.register(nv.nvmlShutdown)
            _GPU_GUARD_INITIALIZED = True
        pids = set()
        for index in range(nv.nvmlDeviceGetCount()):
            handle = nv.nvmlDeviceGetHandleByIndex(index)
            for process in nv.nvmlDeviceGetComputeRunningProcesses(handle):
                pids.add(process.pid)
        _GPU_GUARD_CACHE = (now, tuple(sorted(pids)))
        return _GPU_GUARD_CACHE[1]

def _owned_gpu_contexts(group_pid):
    owned = []
    for pid in _gpu_context_pids():
        try:
            if pid == group_pid or os.getpgid(pid) == group_pid:
                owned.append(pid)
        except (ProcessLookupError, PermissionError):
            pass
    return owned

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def capture(command, *, label, evidence_root, cwd, timeout=600, env=None, metadata=None,
            cpus=(), forbid_gpu_contexts=False, nice_level=None):
    if not LABEL.fullmatch(label):
        raise ValueError("label must contain only letters, digits, dot, dash or underscore")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    root = Path(evidence_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = root / (label + ".txt")
    receipt_path = root / (label + ".receipt.json")
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    begin = time.monotonic()
    timed_out = False
    launch_error = None
    gpu_contexts_rejected = []
    if nice_level is not None:
        command = ["nice", "-n", str(nice_level), *command]
    if cpus:
        command = ["taskset", "-c", ",".join(map(str, cpus)), *command]
    with output.open("x") as stream:
        try:
            if forbid_gpu_contexts:
                _gpu_context_pids()  # Require the guard before any process starts.
            process = subprocess.Popen(
                list(command), cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                env=env, start_new_session=True,
            )
            if forbid_gpu_contexts:
                try:
                    while process.poll() is None:
                        gpu_contexts_rejected = _owned_gpu_contexts(process.pid)
                        if gpu_contexts_rejected or time.monotonic()-begin > timeout:
                            timed_out = not gpu_contexts_rejected
                            os.killpg(process.pid, signal.SIGKILL)
                            break
                        try: process.wait(timeout=.5)
                        except subprocess.TimeoutExpired: pass
                    process.wait()
                    code = 125 if gpu_contexts_rejected else 124 if timed_out else process.returncode
                except Exception as exc:
                    launch_error = "CPU GPU-context guard failed: " + str(exc)
                    if process.poll() is None: os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    code = 125
            else:
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    code = 124
        except OSError as exc:
            launch_error = str(exc)
            stream.write(launch_error + "\n")
            code = 127
    record = dict(label=label, started_utc=started, cwd=str(Path(cwd).resolve()),
                  argv=list(command), timeout_seconds=timeout,
                  elapsed_seconds=round(time.monotonic()-begin, 3),
                  exit_code=code, timed_out=timed_out, output=output.name,
                  sha256=sha256(output), metadata=metadata or {})
    if forbid_gpu_contexts:
        record.update(cpu_only=True, owned_gpu_contexts_rejected=gpu_contexts_rejected,
                      cpu_affinity=list(cpus), nice_level=nice_level)
    if launch_error:
        record["launch_error"] = launch_error
    receipt_path.write_text(json.dumps(record, indent=2) + "\n")
    return record

def clean_environment(sim_root: Path, cuda_root: Path, *, gpu="0", omp_threads=4):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("FLASHGPU_SASS") or key.startswith("GPGPUSIM_") or key == "PTX_SIM_MODE_FUNC":
            env.pop(key)
    libraries = [str(cuda_root / "lib64")]
    for part in env.get("LD_LIBRARY_PATH", "").split(":"):
        if part and str(sim_root / "lib") not in part and "/FlashGPU-Sim/lib/" not in part:
            libraries.append(part)
    env.update(CUDA_INSTALL_PATH=str(cuda_root), CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS=str(omp_threads),
               LD_LIBRARY_PATH=":".join(libraries),
               PATH=str(cuda_root / "bin") + ":" + env.get("PATH", ""))
    return env

def run_test(*, binary, sim_root, cuda_root, evidence_root, label,
             preset="SM90_H100", mode="ptx", gtest_filter="*", overrides=(),
             timeout=600, scratch_root=None, gpu="0", omp_threads=4,
             environment_overrides=None, program_arguments=None, cpu_only=False, cpus=()):
    if mode not in ("native", "ptx", "sass", "sass-timing"):
        raise ValueError("unknown execution mode")
    if cpu_only and mode == "native":
        raise ValueError("CPU-only simulation cannot invoke native GPU execution")
    if not LABEL.fullmatch(label):
        raise ValueError("invalid run label")
    if omp_threads <= 0:
        raise ValueError("omp_threads must be positive")
    sim_root, cuda_root, binary = map(lambda p: Path(p).resolve(),
                                     (sim_root, cuda_root, binary))
    if not binary.is_file():
        raise FileNotFoundError(binary)
    config_source = sim_root / "configs" / preset
    if not (config_source / "gpgpusim.config").is_file():
        raise FileNotFoundError(config_source / "gpgpusim.config")
    scratch = Path(scratch_root) if scratch_root else sim_root / "tests/run/feature-validation"
    scratch.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=label + "-", dir=scratch))
    shutil.copytree(config_source, run_dir, dirs_exist_ok=True)
    config = run_dir / "gpgpusim.config"
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator or not re.fullmatch(r"-?[A-Za-z0-9_]+", key) or not value or "\n" in value:
            raise ValueError("config override must have OPTION=VALUE form")
        option = "-" + key.lstrip("-")
        lines = [line for line in config.read_text().splitlines()
                 if line.split()[:1] != [option]]
        config.write_text("\n".join(lines + [option + " " + value]) + "\n")
    env = clean_environment(sim_root, cuda_root, gpu=gpu, omp_threads=omp_threads)
    env.update(environment_overrides or {})
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
        for name in ("LD_PRELOAD", "CUDA_INJECTION64_PATH", "AMORA_CUPTI_JSONL"):
            env.pop(name, None)
    if mode.startswith("sass"):
        arch = "sm90" if preset.startswith("SM90") else "sm120"
        env.update(FLASHGPU_SASS_AUTO="1", FLASHGPU_SASS_BINARY=str(binary),
                   FLASHGPU_SASS_ARCH=arch,
                   FLASHGPU_SASS_DECODE_TOOL=str(sim_root / "src/gpgpu-sim/flash/sass/tools/dump_kernel_sassir.py"),
                   FLASHGPU_SASS_PYTHON=sys.executable,
                   FLASHGPU_SASS_NVDISASM=str(cuda_root / "bin/nvdisasm"),
                   FLASHGPU_SASS_CUOBJDUMP=str(cuda_root / "bin/cuobjdump"))
        if mode == "sass-timing":
            env["FLASHGPU_SASS_TIMING"] = "1"
    root = Path(evidence_root).resolve()
    xml = root / (label + ".xml") if program_arguments is None else None
    if xml is not None and xml.exists():
        raise FileExistsError(xml)
    command = ([str(binary), "--gtest_filter=" + gtest_filter,
                "--gtest_output=xml:" + str(xml)]
               if program_arguments is None else [str(binary), *program_arguments])
    if mode != "native":
        command = ["bash", "-c", "source " + shlex.quote(str(sim_root / "setup_environment"))
                   + "; exec " + shlex.join(command)]
    revision = subprocess.check_output(["git", "-C", str(sim_root), "rev-parse", "HEAD"], text=True).strip()
    metadata = dict(kind="test" if program_arguments is None else "program",
                    sim_root=str(sim_root), sim_revision=revision,
                    binary=str(binary), binary_sha256=sha256(binary),
                    preset=preset, mode=mode, filter=gtest_filter,
                    program_arguments=program_arguments,
                    overrides=list(overrides), effective_config_sha256=sha256(config),
                    cuda_root=str(cuda_root), gpu=str(gpu), omp_threads=omp_threads,
                    environment_overrides=environment_overrides or {},
                    run_directory=str(run_dir), xml=xml.name if xml is not None else None)
    record = capture(command, label=label, evidence_root=root, cwd=run_dir,
                     timeout=timeout, env=env, metadata=metadata, cpus=cpus,
                     forbid_gpu_contexts=cpu_only, nice_level=19 if cpu_only else None)
    return record

def summarize(root):
    root = Path(root)
    records = {}
    legacy = root / "commands.jsonl"
    if legacy.exists():
        for line in legacy.read_text().splitlines():
            item = json.loads(line)
            records[item["label"]] = item
    for path in sorted(root.glob("*.receipt.json")):
        item = json.loads(path.read_text())
        records[item["label"]] = item
    results = []
    for label, record in sorted(records.items()):
        result = dict(label=label, exit_code=record["exit_code"],
                      elapsed_seconds=record["elapsed_seconds"],
                      status="passed" if record["exit_code"] == 0 else "failed")
        if record.get("timed_out"):
            result["status"] = "timeout"
        elif record["exit_code"] < 0:
            result["status"] = "crashed"
        xml = root / (label + ".xml")
        if xml.exists():
            tree = ET.parse(xml).getroot()
            cases = []
            for case in tree.findall(".//testcase"):
                failures = [f.get("message", f.text or "") for f in case.findall("failure")]
                skipped = case.find("skipped") is not None or case.get("status") == "notrun"
                cases.append(dict(name=case.get("classname", "") + "." + case.get("name", ""),
                                  status="failed" if failures else "skipped" if skipped else "passed",
                                  failure_messages=failures))
            result.update(tests=len(cases),
                          passed=sum(c["status"] == "passed" for c in cases),
                          failed=sum(c["status"] == "failed" for c in cases),
                          skipped=sum(c["status"] == "skipped" for c in cases),
                          cases=cases)
            if result["failed"]:
                result["status"] = "failed"
            elif not cases:
                result["status"] = "no_tests"
            elif result["passed"] == 0:
                result["status"] = "skipped"
        elif record.get("metadata", {}).get("kind") == "test" and record["exit_code"] == 0:
            result["status"] = "missing_test_result"
        results.append(result)
    return dict(runs=results)

