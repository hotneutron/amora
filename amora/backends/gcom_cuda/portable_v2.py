"""Strict, device-bound capture and CPU replay primitives for portable v2."""
from __future__ import annotations

from datetime import datetime, timezone
import atexit
import fcntl
import hashlib
import importlib
import gzip
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import threading
import time

_NVML_LOCK = threading.RLock()
_NVML_INITIALIZED = False


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def gpu_snapshot(only_uuid=None):
    global _NVML_INITIALIZED
    import pynvml as nv
    result = []
    with _NVML_LOCK:
        if not _NVML_INITIALIZED:
            nv.nvmlInit()
            atexit.register(nv.nvmlShutdown)
            _NVML_INITIALIZED = True
        for index in range(nv.nvmlDeviceGetCount()):
            handle = nv.nvmlDeviceGetHandleByIndex(index)
            text = lambda value: value.decode() if isinstance(value, bytes) else value
            uuid = text(nv.nvmlDeviceGetUUID(handle))
            if only_uuid is not None and uuid != only_uuid: continue
            usage = nv.nvmlDeviceGetUtilizationRates(handle)
            memory = nv.nvmlDeviceGetMemoryInfo(handle, version=nv.nvmlMemory_v2)
            compute = nv.nvmlDeviceGetComputeRunningProcesses(handle)
            graphics = nv.nvmlDeviceGetGraphicsRunningProcesses(handle)
            major, minor = nv.nvmlDeviceGetCudaComputeCapability(handle)
            mig = nv.nvmlDeviceGetMigMode(handle)[0]
            affinity = nv.nvmlDeviceGetCpuAffinity(handle, (max(os.sched_getaffinity(0))+64)//64)
            host_cpus = [i for i in os.sched_getaffinity(0) if affinity[i//64] & (1 << (i%64))]
            result.append(dict(index=index, uuid=uuid,
                               name=text(nv.nvmlDeviceGetName(handle)), compute_capability=[major, minor],
                               gpu_utilization=usage.gpu, memory_utilization=usage.memory,
                               memory_used=memory.used, memory_reserved=memory.reserved, mig=mig,
                               cpu_affinity=sorted(host_cpus),
                               sm_clock_mhz=nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM),
                               memory_clock_mhz=nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_MEM),
                               processes=sorted({x.pid for x in compute + graphics}), observed_utc=utc()))
    return result


def idle_candidate(device):
    return (device["name"] == "NVIDIA H100 80GB HBM3" and device["compute_capability"] == [9, 0]
            and device["mig"] == 0 and not device["processes"]
            and device["gpu_utilization"] == 0 and device["memory_utilization"] == 0
            and device["memory_used"] <= 64 * 1024 * 1024)


def observe_idle_pool(root, *, quiet_seconds=30, timeout=300, only_uuid=None):
    """Observe hardware availability; this does not itself acquire a lease."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    since, history, count = {}, [], 0
    begin = time.monotonic()
    started=utc()
    while time.monotonic() - begin < timeout:
        snapshot = gpu_snapshot(only_uuid)
        tick = time.monotonic()
        history.append(snapshot)
        history=history[-60:];count+=1
        with (root/"gpu-observations.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(session_started=started,snapshot=snapshot))+"\n")
        for device in snapshot:
            if idle_candidate(device): since.setdefault(device["uuid"], tick)
            else: since.pop(device["uuid"], None)
        ready = [d for d in snapshot if d["uuid"] in since and tick - since[d["uuid"]] >= quiet_seconds]
        write(root / "gpu-observation.json", dict(started_utc=started,observation_count=count,
                                                       quiet_seconds=quiet_seconds, observations=history,
                                                       ready=[d["uuid"] for d in ready]))
        if ready: return ready
        time.sleep(.5)
    return []


class GpuLease:
    """Advisory UUID lease plus continuous foreign-context rejection."""
    def __init__(self, root, device):
        self.root, self.device, self.lock = Path(root), device, None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / (self.device["uuid"] + ".lock")).open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = next(d for d in gpu_snapshot(self.device["uuid"]))
        if not idle_candidate(current):
            self.lock.close()
            raise RuntimeError("GPU is no longer idle at lease acquisition")
        write(self.root / (self.device["uuid"] + ".json"), dict(state="leased", pid=os.getpid(),
              device=current, timestamp=utc(), mechanism="advisory lease plus monitored context exclusion"))
        return self

    def __exit__(self, *args):
        write(self.root / (self.device["uuid"] + ".json"), dict(state="released", pid=os.getpid(), device=self.device, timestamp=utc()))
        self.lock.close()


def hardware_env(gpu_uuid, cuda_root, tracer=None, traces=None):
    """Set visibility in the child environment; never mutate os.environ."""
    env = dict(os.environ)
    tracer_settings=("TRACES_FOLDER","USER_DEFINED_FOLDERS","DYNAMIC_KERNEL_LIMIT_START",
                     "DYNAMIC_KERNEL_LIMIT_END","EXCLUDE_PRED_OFF","ACTIVE_FROM_START","AMORA_CUPTI_JSONL")
    for key in list(env):
        if key.startswith(("FLASHGPU_", "GPGPUSIM_")) or key in ("LD_PRELOAD", "CUDA_INJECTION64_PATH", "PTX_SIM_MODE_FUNC",*tracer_settings):
            env.pop(key, None)
    cuda_root = Path(cuda_root)
    env.update(CUDA_VISIBLE_DEVICES=gpu_uuid, CUDA_INSTALL_PATH=str(cuda_root),
               LD_LIBRARY_PATH=str(cuda_root / "lib64") + ":/lib/x86_64-linux-gnu",
               PATH=str(cuda_root / "bin") + ":/usr/bin:/bin")
    if tracer is not None:
        env.update(USER_DEFINED_FOLDERS="1", TRACES_FOLDER=str(Path(traces).resolve()),
                   CUDA_MODULE_LOADING="EAGER",
                   CUDA_INJECTION64_PATH=str(Path(tracer).resolve()),
                   ACTIVE_FROM_START="1", DYNAMIC_KERNEL_LIMIT_START="0", DYNAMIC_KERNEL_LIMIT_END="0", EXCLUDE_PRED_OFF="0")
    return env


def capture_hardware(argv, *, lease, root, label, env, timeout=1800, cpus=(), disk_reserve_bytes=8*1024**3,
                     memory_reserve_bytes=0):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    output, receipt_path = root / (label + ".txt"), root / (label + ".receipt.json")
    if output.exists() or receipt_path.exists(): raise FileExistsError(label)
    command = (["taskset", "-c", ",".join(map(str, cpus))] if cpus else []) + list(map(str, argv))
    started, begin = utc(), time.monotonic()
    observations, foreign, monitor_error, timed_out, storage_exhausted = [], [], None, False, False
    memory_exhausted=False
    with output.open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        owned_pids = {process.pid}
        while process.poll() is None:
            if memory_reserve_bytes:
                available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:')))*1024
                if available<memory_reserve_bytes:
                    memory_exhausted=True
                    os.killpg(process.pid,signal.SIGKILL)
                    break
            if shutil.disk_usage(root).free < disk_reserve_bytes:
                storage_exhausted = True
                os.killpg(process.pid, signal.SIGKILL)
                break
            if time.monotonic() - begin > timeout:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                break
            try:
                device = next(d for d in gpu_snapshot(lease.device["uuid"]))
                observations.append(device)
                for pid in device["processes"]:
                    try: owned = pid in owned_pids or os.getpgid(pid) == process.pid
                    except (ProcessLookupError, PermissionError): owned = False
                    if owned: owned_pids.add(pid)
                    if not owned: foreign.append(pid)
                if foreign:
                    if process.poll() is None: os.killpg(process.pid, signal.SIGKILL)
                    break
            except Exception as exc:
                monitor_error = str(exc)
                if process.poll() is None: os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(.2)
        code = process.wait()
    record = dict(label=label, argv=command, started_utc=started, elapsed_seconds=time.monotonic()-begin,
                  exit_code=code, timed_out=timed_out, output=output.name, output_sha256=sha(output),
                  storage_exhausted=storage_exhausted, memory_exhausted=memory_exhausted,
                  gpu_uuid=lease.device["uuid"], physical_gpu=lease.device["index"], logical_device=0,
                  foreign_processes=sorted(set(foreign)), monitor_error=monitor_error, observations=observations,
                  dedicated_interval=not foreign and monitor_error is None, cpus=list(cpus),
                  environment={key:env.get(key) for key in ("CUDA_VISIBLE_DEVICES", "CUDA_INSTALL_PATH", "LD_PRELOAD", "CUDA_INJECTION64_PATH", "TRACES_FOLDER","CUDA_MODULE_LOADING")})
    write(receipt_path, record)
    return record


def driver_rows(path, manifest, selected, *, instrumented=False):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if {row["index"] for row in rows} != set(selected) or len(rows) != len(selected): raise ValueError("driver selection mismatch")
    for row in rows:
        if type(row["index"]) is not int or not 0 <= row["index"] < len(manifest["cases"]):
            raise ValueError("invalid driver case index")
        case = manifest["cases"][row["index"]]
        if (row["case_id"] != case["id"] or type(row["passed"]) is not bool or
                row["node_ports_checked"] != sum(not b["input"] for b in case["bindings"])):
            raise ValueError("driver case or oracle-boundary mismatch")
        if (not math.isfinite(row.get("max_abs_error", math.nan)) or row["max_abs_error"] < 0 or
                (row["passed"] and (row.get("mismatches") != 0 or row.get("first_failed_value") != -1))):
            raise ValueError("driver numerical evidence contradicts pass status")
        if not row["passed"] and row.get("samples_ms"): raise ValueError("failed numerical output published timing")
        if any(not math.isfinite(x) or x < 0 or (x == 0 and not instrumented) for x in row.get("samples_ms", [])): raise ValueError("invalid event timing")
        row.update(case_sha256=case["case_sha256"], program_sha256=case["program_sha256"], timing_instrumented=instrumented)
        if instrumented:
            row["instrumented_event_samples_ms"] = row.get("samples_ms", [])
            row["samples_ms"] = []
    return rows


def verify_placement(output, gpu_uuid):
    lines = Path(output).read_text(errors="replace").splitlines()
    matches = [x for x in lines if x.startswith("AMORA_V2_DEVICE ")]
    if len(matches) != 1: raise ValueError("missing or ambiguous driver placement evidence")
    line = matches[0]
    placement = json.loads(line.split(" ", 1)[1])
    if (placement["uuid"] != gpu_uuid or placement["logical_device"] != 0 or placement["sm"] != 9 or placement["sms"] != 132
            or placement.get("operator_mask") != 2147483647):
        raise ValueError("driver placement is not the leased H100")
    return placement


def inspect_trace(trace_dir, trace_module, expected_launches, *, expected_operator_ids=None,
                  expected_kernel_names=None):
    trace_dir = Path(trace_dir)
    pb = trace_dir / "dynamic_trace.pb"
    extra = trace_dir / "extra_info/enhanced_execution_info.json"
    if not pb.is_file() or not extra.is_file(): raise ValueError("incomplete protobuf/static trace bundle")
    trace = trace_module.Trace()
    trace.ParseFromString(pb.read_bytes())
    if trace.binary_version!=90:raise ValueError("trace binary architecture is not SM90")
    if len(trace.gpu_device) != 1: raise ValueError("trace has multiple or missing GPU devices")
    device = next(iter(trace.gpu_device.values()))
    nonempty_streams = [s for s in device.streams.values() if s.kernels]
    if len(nonempty_streams) != 1: raise ValueError("portable program requires one traced kernel stream")
    kernels = [kernel for stream in device.streams.values() for kernel in stream.kernels]
    kernels.sort(key=lambda k:k.id)
    if len(kernels) != expected_launches or len({k.id for k in kernels}) != len(kernels): raise ValueError("trace launch accounting mismatch")
    # The tracer appends its documented variant delimiter and ordinal to the
    # original CUDA symbol. Compare original symbols, retaining variant names
    # in the trace evidence and preserving launch order.
    original_names = [re.sub(r"___\d+$", "", k.name) for k in kernels]
    if expected_kernel_names is not None and original_names != list(expected_kernel_names):
        raise ValueError("trace/native kernel identity or order mismatch")
    if any((expected_kernel_names is None and "node_kernel" not in k.name) or min(k.grid_dim.x,k.grid_dim.y,k.grid_dim.z,
               k.block_dim.x,k.block_dim.y,k.block_dim.z) <= 0 for k in kernels):
        raise ValueError("unexpected traced application kernel")
    if expected_operator_ids is not None:
        operators=[]
        for kernel in kernels:
            match=re.search(r"node_kernel(?:ILi(\d+)E|<(\d+)>)",kernel.name)
            if not match:raise ValueError("cannot bind traced kernel to portable node operator")
            operators.append(int(next(x for x in match.groups() if x is not None)))
        if operators!=expected_operator_ids:raise ValueError("trace node/operator launch order mismatch")
    metadata=json.loads(extra.read_text())
    static={k["unique_function_id"]:k for k in metadata.get("kernels",[])}
    if not static:raise ValueError("missing cubin static function metadata")
    instructions={}
    for fid,function in static.items():
        if function.get("architecture_version")!=90 or function.get("is_captured_from_binary") is not True:
            raise ValueError("static function lacks admitted SM90 cubin capture")
        rows=function.get("instructions",[])
        if not rows or any((len(r.get("encoded_instruction",[]))!=2 and r.get("op_code")!="NOP") or not r.get("op_code")
                or "control_bits" not in r for r in rows):raise ValueError("incomplete static instruction/control-bit metadata")
        # The shipped parser inserts padding NOPs without encoding. They can
        # remain in static metadata but cannot admit a dynamic instruction.
        instructions[fid]={r["pc_num_dec"] for r in rows if len(r.get("encoded_instruction",[]))==2}
    if any(k.function_unique_id not in static for k in kernels):raise ValueError("traced kernel lacks static function identity")
    files = [p for p in trace_dir.rglob("*") if p.is_file()]
    blocks = [p for p in (trace_dir / "threadblocks").rglob("*") if p.is_file()]
    expected_ctas = sum(k.grid_dim.x*k.grid_dim.y*k.grid_dim.z for k in kernels)
    if len(blocks) != expected_ctas: raise ValueError("trace has incomplete or extra threadblock records")
    block_module = importlib.import_module("threadblock_pb2")
    instruction_count = 0
    for stream in device.streams.values():
        for kernel in stream.kernels:
            folder = trace_dir / "threadblocks" / f"device_{device.id}" / f"stream_{stream.id}" / f"kernel_{kernel.id}"
            for x in range(kernel.grid_dim.x):
                for y in range(kernel.grid_dim.y):
                    for z in range(kernel.grid_dim.z):
                        path = folder / f"d_{device.id}_s_{stream.id}_k_{kernel.id}_{x},{y},{z}.pb"
                        if not path.is_file() or not path.stat().st_size: raise ValueError("missing selected threadblock trace: " + str(path))
                        block = block_module.threadblock()
                        block.ParseFromString(path.read_bytes())
                        if [block.block_id.x,block.block_id.y,block.block_id.z] != [x,y,z]:
                            raise ValueError("threadblock coordinate mismatch")
                        warps = (kernel.block_dim.x*kernel.block_dim.y*kernel.block_dim.z+31)//32
                        if set(block.warps) != set(range(warps)):
                            raise ValueError("threadblock warp coverage mismatch")
                        if any(not w.instructions or w.id != key for key,w in block.warps.items()):
                            raise ValueError("missing dynamic warp instructions")
                        if any(i.function_unique_id not in instructions or i.pc not in instructions[i.function_unique_id]
                               for w in block.warps.values() for i in w.instructions):
                            raise ValueError("dynamic instruction PC/function missing from cubin metadata")
                        instruction_count += sum(len(w.instructions) for w in block.warps.values())
    hashes = {str(p.relative_to(trace_dir)):sha(p) for p in sorted(files)}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return dict(trace_sha256=fingerprint, files=hashes, bytes=sum(p.stat().st_size for p in files),
                launches=len(kernels), ctas=expected_ctas, threadblock_files=len(blocks),
                dynamic_warp_instructions=instruction_count,
                cubin_functions=len(static),static_instruction_count=sum(map(len,instructions.values())),
                nvbit_version=trace.nvbit_version, binary_version=trace.binary_version,
                kernels=[dict(trace_id=k.id, name=k.name, stream=stream.id,
                              grid=[k.grid_dim.x,k.grid_dim.y,k.grid_dim.z], block=[k.block_dim.x,k.block_dim.y,k.block_dim.z])
                         for stream in device.streams.values() for k in stream.kernels],
                metadata_keys=list(metadata))


def simulate_trace(sim_bin, lib_dir, configs, trace_dir, trace_info, *, root, cpus, timeout=7200,
                   disk_reserve_bytes=0):
    """Replay exactly once for an immutable binary/config/trace fingerprint."""
    root, sim_bin, lib_dir = Path(root), Path(sim_bin), Path(lib_dir)
    if not cpus or len(cpus) != len(set(cpus)) or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("invalid replay CPU allocation")
    for relative, digest in trace_info["files"].items():
        if sha(Path(trace_dir)/relative) != digest: raise ValueError("trace changed after admission")
    overrides = ["-tma_real_base_addr_enable", "0", "-tma_operand_addr_tiling_enable", "0",
                 "-wait_block_enable", "0", "-wait_block_exact_generation_enable", "0",
                 "-wait_block_late_arrival_instrument_enable", "0"]
    # The admitted portable driver contains ordinary global-memory kernels only.
    # Static trace metadata is checked as well; a TMA opcode forbids these flags.
    static = json.loads((Path(trace_dir) / "extra_info/enhanced_execution_info.json").read_text())
    def leaves(value):
        if isinstance(value, dict):
            for v in value.values(): yield from leaves(v)
        elif isinstance(value, list):
            for v in value: yield from leaves(v)
        elif isinstance(value, str): yield value
    if any(re.match(r"^(?:UTMA|TMA|CP\.ASYNC\.BULK)", s.strip(), re.I) for s in leaves(static)):
        raise ValueError("TMA trace requires real operands; compatibility disabling refused")
    opcodes=sorted({i["op_code"] for k in static.get("kernels",[]) for i in k.get("instructions",[])})
    # CTA named barriers, including Hopper DEFER_BLOCKING, use BARRIER_OP.
    # Their native model is independent of the asynchronous MBARRIER_OP /
    # WaitBlock sidecar consumers; do not reject ordinary BAR or MEMBAR.
    if any(re.match(r"^(?:SYNCS\b|SYNC\b|MBARRIER\b|UTMA|TMA|UBLK|WGMMA|HGMMA|GMMA|CP\.ASYNC|RED\.ASYNC)",op,re.I) for op in opcodes):
        raise ValueError("sync/TMA trace requires its contract sidecars; absent-feature compatibility refused")
    contract = dict(binary_sha256=sha(sim_bin), library_sha256=sha(lib_dir / "libcudart.so"),
                    config_sha256=[sha(Path(p)) for p in configs], trace_sha256=trace_info["trace_sha256"], overrides=overrides,
                    absent_feature_proof=dict(opcodes=opcodes,disabled_consumers=["TMA address sidecars","WaitBlock exact-generation sidecars"]))
    key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    folder = root / key
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / "result.json"
    if cache.exists():
        value = json.loads(cache.read_text())
        if value["contract"] != contract or value["output_sha256"] != replay_log_sha(folder):
            raise ValueError("replay cache identity changed")
        for relative,digest in value.get("auxiliary_files",{}).items():
            if sha(folder/"auxiliary-output"/relative)!=digest:raise ValueError("replay auxiliary output changed")
        return value
    with (folder / "owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if cache.exists(): return json.loads(cache.read_text())
        if (folder / "started.json").exists(): raise RuntimeError("replay key already attempted; no identical automatic retry")
        write(folder / "started.json", dict(contract=contract, timestamp=utc(), pid=os.getpid()))
        env = dict(os.environ)
        for name in list(env):
            if name.startswith(("FLASHGPU_","GPGPUSIM_")) or name in ("LD_PRELOAD","CUDA_INJECTION64_PATH","OMP_PROC_BIND","OMP_PLACES","GOMP_CPU_AFFINITY","KMP_AFFINITY","KMP_HW_SUBSET","OMP_THREAD_LIMIT","AMORA_CUPTI_JSONL"):
                env.pop(name,None)
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(len(cpus)), OMP_DYNAMIC="FALSE",
                   LD_LIBRARY_PATH=str(lib_dir) + ":/usr/local/cuda-12.8/lib64:/lib/x86_64-linux-gnu")
        argv = [sys.executable, str(Path(__file__).with_name("portable_v2_runner.py")),
                str((folder/"resources.json").resolve()),
                "taskset", "-c", ",".join(map(str,cpus)), str(sim_bin), "-trace", str(Path(trace_dir).resolve()/"dynamic_trace.pb")]
        for config in configs: argv += ["-config", str(Path(config).resolve())]
        argv += overrides
        begin = time.monotonic()
        timed_out = False
        storage_exhausted = False
        with (folder / "stdout.txt").open("x") as stream:
            process = subprocess.Popen(argv, cwd=trace_dir, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            if disk_reserve_bytes:
                while process.poll() is None:
                    if shutil.disk_usage(folder).free<disk_reserve_bytes:
                        storage_exhausted=True
                        os.killpg(process.pid,signal.SIGKILL)
                        break
                    if time.monotonic()-begin>timeout:
                        timed_out=True
                        os.killpg(process.pid,signal.SIGKILL)
                        break
                    try:process.wait(timeout=1)
                    except subprocess.TimeoutExpired:pass
                code=process.wait()
            else:
                try: code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid,signal.SIGKILL)
                    code=process.wait()
        text = (folder / "stdout.txt").read_text(errors="replace")
        rows, parse_error = [], None
        if code == 0:
            try: rows = parse_kernel_cycles(text)
            except ValueError as exc: parse_error = str(exc)
        complete = (code==0 and "GPGPU-Sim: *** simulation thread exiting ***" in text
                    and len(rows)==trace_info["launches"] and parse_error is None
                    and [r["trace_id"] for r in rows] == [k["trace_id"] for k in trace_info["kernels"]])
        resource_file = folder/"resources.json"
        resources = json.loads(resource_file.read_text()) if resource_file.exists() else {}
        value=dict(key=key,contract=contract,exit_code=code,timed_out=timed_out,complete=complete,
                   storage_exhausted=storage_exhausted,
                   kernel_rows=rows,elapsed_seconds=time.monotonic()-begin,argv=argv,cpus=list(cpus),
                   output_sha256=sha(folder/"stdout.txt"),folder=str(folder),parse_error=parse_error)
        value.update(max_rss_kib=resources.get("max_rss_kib"), resource_usage=resources,
                     omp_threads=len(cpus), resources_sha256=sha(resource_file) if resource_file.exists() else None)
        preserve_replay_auxiliary(value,trace_dir,trace_info)
        write(cache,value)
        return value


def preserve_replay_auxiliary(replay,trace_dir,info):
    """Keep instruction statistics and other replay outputs before raw cleanup."""
    trace_dir=Path(trace_dir);destination=Path(replay["folder"])/"auxiliary-output"
    files={}
    for path in trace_dir.rglob("*"):
        if not path.is_file():continue
        relative=str(path.relative_to(trace_dir))
        if relative in info["files"]:continue
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        if sha(target)!=sha(path):raise ValueError("replay auxiliary copy changed bytes")
        files[relative]=sha(target)
    replay["auxiliary_files"]=files


def replay_log_sha(folder):
    folder=Path(folder)
    if (folder/"stdout.txt").exists():return sha(folder/"stdout.txt")
    digest=hashlib.sha256()
    with gzip.open(folder/"stdout.txt.gz","rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def compress_replay_log(replay):
    folder=Path(replay["folder"])
    raw=folder/"stdout.txt";packed=folder/"stdout.txt.gz"
    if not raw.exists():
        if replay_log_sha(folder)!=replay["output_sha256"]:raise ValueError("compressed replay log changed")
        return
    if sha(raw)!=replay["output_sha256"]:raise ValueError("replay output changed before archival")
    with raw.open("rb") as source,gzip.open(packed,"xb",compresslevel=3) as out:shutil.copyfileobj(source,out)
    digest=hashlib.sha256()
    with gzip.open(packed,"rb") as source:
        for block in iter(lambda:source.read(1024*1024),b""):digest.update(block)
    if digest.hexdigest()!=replay["output_sha256"]:raise ValueError("replay log archive is not lossless")
    raw.unlink()


def activity_samples(path, *, nodes, samples, warmup=3, repetitions=1):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return activity_records(records,nodes=nodes,samples=samples,warmup=warmup,repetitions=repetitions)


def activity_records(records, *, nodes, samples, warmup=3, repetitions=1):
    ends = [r for r in records if r["kind"] == "complete"]
    if len(ends) != 1 or ends[0]["error_code"] != 0 or ends[0]["dropped_records"] != 0:
        raise ValueError("CUPTI activity capture incomplete or dropped records")
    kernels = sorted((r for r in records if r["kind"] == "kernel"), key=lambda r:r["start_ns"])
    if len(kernels) != nodes*(1+(warmup+samples)*repetitions) or any("node_kernel" not in r["name"] for r in kernels):
        raise ValueError("CUPTI application launch accounting mismatch")
    if any(r["start_ns"] <= 0 or r["end_ns"] <= r["start_ns"] for r in kernels): raise ValueError("invalid CUPTI kernel interval")
    if len({(r["device"],r["context"],r["stream"]) for r in kernels}) != 1:
        raise ValueError("CUPTI kernel placement or stream mismatch")
    if any(a["end_ns"] > b["start_ns"] for a,b in zip(kernels,kernels[1:])):
        raise ValueError("overlapping sequential program kernels")
    selected = kernels[nodes*(1+warmup*repetitions):]
    group=nodes*repetitions
    result = [sum(r["end_ns"]-r["start_ns"] for r in selected[i:i+group])/1e6 for i in range(0,len(selected),group)]
    return dict(samples_ms=result, estimand="sum of target kernel activity durations", kernels=kernels,
                source="CUPTI without SASS instruction instrumentation")


def activity_cases(path, case_nodes, *, samples, warmup=3, repetitions=1):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    ends = [r for r in records if r["kind"] == "complete"]
    kernels = sorted((r for r in records if r["kind"] == "kernel"),key=lambda r:r["start_ns"])
    if len(kernels) != sum(case_nodes.values())*(1+(warmup+samples)*repetitions):
        raise ValueError("CUPTI case panel launch mismatch")
    offset,result=0,{}
    for index,nodes in case_nodes.items():
        count=nodes*(1+(warmup+samples)*repetitions)
        result[index]=activity_records(kernels[offset:offset+count]+ends,nodes=nodes,samples=samples,warmup=warmup,repetitions=repetitions)
        offset+=count
    return result


def parse_kernel_cycles(text):
    """Retain every completed kernel counter block, never just the last value."""
    result, current, trace_id = [], None, None
    for line in text.splitlines():
        launch = re.search(r"trace_kernel_id:\s*(\d+)", line)
        if launch: trace_id = int(launch.group(1))
        cycle = re.fullmatch(r"\s*gpu_sim_cycle\s*=\s*(\d+)\s*", line)
        if cycle: current = dict(trace_id=trace_id, cycles=int(cycle.group(1)))
        total = re.fullmatch(r"\s*gpu_tot_sim_cycle\s*=\s*(\d+)\s*", line)
        if total and current is not None:
            current["cumulative_cycles"] = int(total.group(1))
            result.append(current)
            current = None
    if any(x["cycles"] <= 0 for x in result): raise ValueError("nonpositive completed kernel cycles")
    if any(a["cumulative_cycles"] > b["cumulative_cycles"] for a,b in zip(result,result[1:])):
        raise ValueError("nonmonotonic simulator timeline")
    return result
