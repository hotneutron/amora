"""Matched H100 canonical-template capture and immutable simulator replay.

The canonical templates are timing workloads. They have no independent oracle.
This driver does not change their arithmetic, inputs, or library calls.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import fcntl
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import threading
import time

from amora.backends.flashgpu_sim import runner as flash
from amora.backends.gcom_cuda import portable_v2 as lane
from amora.backends.gcom_cuda import portable_v2_resources as resources
from amora.benchmarking.schema import BenchmarkCase
from benchmark_generators.ppp_canonical.cases import candidate_items
from benchmark_generators.ppp_canonical.generator import PPP_CANONICAL
from benchmark_generators.ppp_canonical.kernels import CANONICAL_KERNELS
from benchmark_generators.ppp_canonical.replay import contract_for_case

PROMPT = "Work on the Canonical PPP kernel results for both simulators."
RESERVE = 16 * 1024**3
ASYNC = re.compile(r"^(?:SYNCS\b|SYNC\b|MBARRIER\b|UTMA|TMA|UBLK|WGMMA|HGMMA|GMMA|CP\.ASYNC|RED\.ASYNC)", re.I)


def load(path):
    return json.loads(Path(path).read_text())


def work(case):
    s = case.shape if hasattr(case, "shape") else case["shape"]
    k = case.kernel_id if hasattr(case, "kernel_id") else case["kernel_id"]
    if "gemm" in k:
        return 2*s["M"]*s["N"]*s["K"]
    if k == "gelu":
        return s["N"]*50
    if k == "embedding":
        return s["B"]*s["S"]*s["D"]
    if k == "rmsnorm":
        return s["B"]*s["N"]*s["D"]*8
    if k == "megamoe_fp8":
        return s["B"]*s["S"]*8*min(s["D"], 1024)*8
    hidden = s.get("D", 576)
    return s["B"]*s["H"]*math.ceil(s["S"]/64)*hidden*(512 if k == "flash_attention_fwd" else 64)


def activity(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    ends = [r for r in records if r["kind"] == "complete"]
    if len(ends) != 1 or ends[0]["error_code"] or ends[0]["dropped_records"]:
        raise ValueError("CUPTI capture incomplete or lost records")
    kernels = sorted((r for r in records if r["kind"] == "kernel"), key=lambda r:r["start_ns"])
    if not kernels or any(r["start_ns"] <= 0 or r["end_ns"] <= r["start_ns"] for r in kernels):
        raise ValueError("missing or invalid native kernel intervals")
    if len({(r["device"], r["context"], r["stream"]) for r in kernels}) != 1:
        raise ValueError("native workload has multiple devices, contexts or streams")
    if any(a["end_ns"] > b["start_ns"] for a,b in zip(kernels,kernels[1:])):
        raise ValueError("native kernels overlap")
    return dict(kernels=kernels, names=[r["name"] for r in kernels],
                body_ms=sum(r["end_ns"]-r["start_ns"] for r in kernels)/1e6,
                span_ms=(kernels[-1]["end_ns"]-kernels[0]["start_ns"])/1e6)


def matched_native(runs):
    identities = lambda r:[(k["name"],k["grid"],k["block"]) for k in r["kernels"]]
    if any(identities(r) != identities(runs[0]) for r in runs):
        raise ValueError("native repeats changed kernel identities or launch dimensions")
    samples = [r["span_ms"] for r in runs]
    median = statistics.median(samples)
    cv = statistics.pstdev(samples)/statistics.mean(samples)
    return dict(native_ms=median, native_body_ms=statistics.median(r["body_ms"] for r in runs),
                samples_ms=samples, cv=cv, timing_qualified=median>=.01 and cv<=.10,
                qualification="span >=10 us; process-repeat CV <=10 percent; complete dedicated captures",
                launches=len(runs[0]["kernels"]), names=runs[0]["names"], kernels=runs[0]["kernels"])


def flash_completion(text, exit_code, launches):
    rows=lane.parse_kernel_cycles(text)
    # FlashGPU's CUDA runtime uses an atexit marker; GCoM uses a simulation
    # thread marker. Both still require successful application exit and every
    # completed launch counter block. The templates synchronize each launch.
    complete=(exit_code==0 and len(rows)==launches and
              ("GPGPU-Sim: *** exit detected ***" in text or "simulation thread exiting" in text))
    return complete, rows


class Campaign:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.project = self.root.parent.parent
        self.previous = self.root.parent/"gcom-portable-v2"
        self.cuda = Path("/usr/local/cuda-13.3")
        self.flashroot = Path("/home/cliu/wk/FlashGPU-Sim")
        self.source = self.previous/"build-source/simulator-remodeled"
        self.tracer = self.source/"util/tracer_nvbit/tracer_tool/tracer_tool.so"
        self.simulator = self.source/"gpu-simulator/bin/release/accel-sim.out"
        self.collector = self.previous/"measurement/portable-cupti.so"
        self.library = Path(load(self.previous/"execution-admission.json")["simulator_library"]["path"]).parent
        self.configs = [self.source/"gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config",
                        self.source/"gpu-simulator/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/trace.config"]
        sys.path.insert(0, str(self.previous/"schema"))
        self.schema = importlib.import_module("trace_pb2")
        self.lock = threading.RLock()

    def status(self, state, **kwargs):
        lane.write(self.root/"status.json", dict(timestamp=lane.utc(), pid=os.getpid(), state=state, **kwargs))

    def prepare(self):
        if (self.root/"manifest.json").exists():
            self.validate()
            return
        cases, generator = PPP_CANONICAL.materialize(target={"vendor":"nvidia","sku":"h100-80g"},
                                                     case_count=2500, seed=20260717)
        pilot = []
        for spec in CANONICAL_KERNELS:
            items = candidate_items(spec)
            item = min(items, key=lambda item:(work(dict(kernel_id=spec.kernel_id,shape=item["shape"])),
                                               json.dumps(item["shape"],sort_keys=True)))
            pilot.append(PPP_CANONICAL._case_from_item(spec=spec,item=item,
                         target={"vendor":"nvidia","sku":"h100-80g"},generator=generator))
        unique = {c.case_key:c for c in cases}
        unique.update({c.case_key:c for c in pilot})
        rows = [asdict(c) for c in unique.values()]
        lane.write(self.root/"manifest.json", dict(user_prompt=PROMPT, target="H100 80GB HBM3 CC9.0",
            generator=generator, primary_keys=[c.case_key for c in cases],pilot_keys=[c.case_key for c in pilot],
            cases=rows, selection="standard h100_2500 plus minimum-work source candidate per family"))
        builds = {}
        for spec in CANONICAL_KERNELS:
            case = next(c for c in pilot if c.kernel_id == spec.kernel_id)
            contract = contract_for_case(case)
            folder = self.root/"build"/spec.kernel_id
            folder.mkdir(parents=True,exist_ok=True)
            binary = folder/"canonical"
            flags = ["-std=c++17","-O2","--fmad=false","-lineinfo","-cudart","shared",
                "-DAMORA_GCOM_TRACE","-gencode","arch=compute_90,code=sm_90",
                "-gencode","arch=compute_90,code=compute_90"]
            receipt = flash.capture([str(self.cuda/"bin/nvcc"),*flags,str(contract.source),
                    "-o",str(binary),*contract.link_flags],label="compile",evidence_root=folder,
                    cwd=folder,timeout=300,env=lane.hardware_env("",self.cuda),
                    metadata=dict(source=str(contract.source),source_sha256=lane.sha(contract.source),flags=flags))
            builds[spec.kernel_id] = dict(source=str(contract.source),source_sha256=lane.sha(contract.source),
                binary=str(binary),binary_sha256=lane.sha(binary) if binary.exists() else None,
                exit_code=receipt["exit_code"],flags=flags,link_flags=contract.link_flags)
        tools = {str(p):lane.sha(p) for p in [self.simulator,self.library/"libcudart.so",self.tracer,
                                            self.collector,*self.configs,self.cuda/"bin/nvcc"]}
        old = load(self.previous/"execution-admission.json")
        for p,digest in old["tools"].items():
            if lane.sha(p) != digest: raise ValueError("previous admitted tool changed: "+p)
        lane.write(self.root/"admission.json",dict(timestamp=lane.utc(),user_prompt=PROMPT,
            manifest_sha256=lane.sha(self.root/"manifest.json"),tools=tools,builds=builds,
            flash_revision=flash.subprocess.check_output(["git","-C",str(self.flashroot),"rev-parse","HEAD"],text=True).strip(),
            flash_overrides=["gpgpu_shader_core_pipeline=2048:32","gpgpu_shader_cta=32"],
            model_parameters_changed=False,numerical_correctness="not_evaluated_no_template_oracle",
            native_policy="three cold-process CUPTI captures; same AMORA_GCOM_TRACE binary for native and simulators",
            timing_policy="native kernel span including component gaps versus simulator service timeline; warmup/embedding flush compiled out",
            bounds=dict(trace_seconds=1800,replay_seconds=7200,flash_seconds=1800,disk_reserve_bytes=RESERVE,
                        replay_threads=4,max_instances=48,expansion_case_limit=None,expansion_work_limit=None)))
        self.validate()
        self.status("prepared",primary_cases=len(cases),pilot_cases=len(pilot),unique_cases=len(rows))

    def validate(self):
        self.admission = load(self.root/"admission.json")
        if lane.sha(self.root/"manifest.json") != self.admission["manifest_sha256"]:
            raise ValueError("canonical manifest changed")
        for path,digest in self.admission["tools"].items():
            if lane.sha(path) != digest:raise ValueError("canonical tool/config changed: "+path)
        self.manifest = load(self.root/"manifest.json")
        self.cases = {c["case_key"]:c for c in self.manifest["cases"]}
        for build in self.admission["builds"].values():
            if lane.sha(build["source"]) != build["source_sha256"]:raise ValueError("canonical source changed")
            if build["binary_sha256"] and lane.sha(build["binary"]) != build["binary_sha256"]:
                raise ValueError("canonical binary changed")

    def hardware_case(self, key, lease, cpus):
        folder = self.root/"points"/key
        folder.mkdir(parents=True,exist_ok=True)
        saved = folder/"hardware.json"
        if saved.exists():return load(saved)
        if (folder/"hardware-started.json").exists():
            value = dict(case_key=key,case=self.cases[key],hardware_status="interrupted_not_retried")
            lane.write(saved,value)
            return value
        case = BenchmarkCase(**self.cases[key])
        build = self.admission["builds"][case.kernel_id]
        value = dict(case_key=key,case=self.cases[key],physical_gpu=lease.device["index"],
                     gpu_uuid=lease.device["uuid"],numerical_correctness="not_evaluated_no_template_oracle")
        lane.write(folder/"hardware-started.json",dict(timestamp=lane.utc(),pid=os.getpid(),case_key=key))
        try:
            if build["exit_code"] != 0:raise ValueError("canonical build failed")
            command = [build["binary"],*contract_for_case(case).args]
            value["command"] = command
            runs = []
            for repeat in range(3):
                label = f"native-{repeat}"
                output = folder/(label+".activity.jsonl")
                env = lane.hardware_env(lease.device["uuid"],self.cuda)
                env.update(LD_PRELOAD=str(self.collector),AMORA_CUPTI_JSONL=str(output))
                receipt = lane.capture_hardware(command,lease=lease,root=folder,label=label,env=env,
                    timeout=300,cpus=cpus,disk_reserve_bytes=RESERVE)
                if receipt["exit_code"] or not receipt["dedicated_interval"]:
                    raise ValueError("native acquisition failed or foreign GPU context detected")
                runs.append(activity(output))
            value.update(matched_native(runs),hardware_status="completed")
            lane.write(folder/"native.json",value)
            trace_dir = folder/"trace"
            env = lane.hardware_env(lease.device["uuid"],self.cuda,self.tracer,trace_dir)
            receipt = lane.capture_hardware(command,lease=lease,root=folder,label="trace",env=env,
                timeout=1800,cpus=cpus,disk_reserve_bytes=RESERVE)
            if receipt["exit_code"] or not receipt["dedicated_interval"]:
                raise ValueError("trace acquisition failed or foreign GPU context detected")
            info = lane.inspect_trace(trace_dir,self.schema,value["launches"],expected_kernel_names=value["names"])
            for native,traced in zip(value["kernels"],info["kernels"]):
                if native["grid"] != traced["grid"] or native["block"] != traced["block"]:
                    raise ValueError("native/trace launch dimensions differ")
            value.update(trace_info=info,trace_dir=str(trace_dir),trace_status="qualified")
        except Exception as exc:
            value.setdefault("hardware_status","not_completed")
            value["trace_status"] = "not_completed"
            value["reason"] = str(exc)
        lane.write(saved,value)
        return value

    def capture(self, keys, scope):
        pending = [k for k in keys if not (self.root/"points"/k/"hardware.json").exists()]
        if not pending:return [load(self.root/"points"/k/"hardware.json") for k in keys]
        devices = lane.observe_idle_pool(self.root/(scope+"-gpu-observation"),quiet_seconds=30,timeout=90)
        if not devices:
            self.status("waiting_for_dedicated_gpu",scope=scope,pending=len(pending))
            return []
        allocation = resources.cpu_admission(self.root/(scope+"-hardware-resources"),devices=devices)
        buckets = [[] for _ in devices]
        for i,key in enumerate(sorted(pending,key=lambda k:work(self.cases[k]))):buckets[i%len(devices)].append(key)
        def worker(device,cpus,selected):
            with lane.GpuLease(self.root/"resource-leases",device) as lease:
                for key in selected:
                    point = self.hardware_case(key,lease,cpus)
                    print(lane.utc(),"hardware",point["case"]["kernel_id"],point.get("trace_status"),flush=True)
                    self.report()
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            jobs = [pool.submit(worker,d,c,b) for d,c,b in zip(devices,allocation["acquisition_cpus"],buckets) if b]
            for job in as_completed(jobs):
                try:job.result()
                except Exception as exc:print(lane.utc(),"GPU admission",repr(exc),flush=True)
        return [load(self.root/"points"/k/"hardware.json") for k in keys if (self.root/"points"/k/"hardware.json").exists()]

    def flash_point(self, point, mode, slot):
        key = point["case_key"]
        folder = self.root/"points"/key
        destination = folder/("flash-"+mode+".json")
        if destination.exists():return load(destination)
        build = self.admission["builds"][point["case"]["kernel_id"]]
        label = "flash-"+mode
        started = folder/(label+"-started.json")
        if started.exists():
            value = dict(status="interrupted_not_retried")
            lane.write(destination,value)
            return value
        lane.write(started,dict(timestamp=lane.utc(),binary_sha256=build["binary_sha256"],arguments=point["command"][1:]))
        try:
            receipt = flash.run_test(binary=build["binary"],sim_root=self.flashroot,cuda_root=self.cuda,
                evidence_root=folder,label=label,preset="SM90_H100",mode=mode,
                overrides=self.admission["flash_overrides"],timeout=1800,
                scratch_root=self.root/"flash-scratch",gpu="",omp_threads=slot["omp_threads"],
                environment_overrides={"GOMP_CPU_AFFINITY":" ".join(map(str,slot["cpus"]))},
                program_arguments=point["command"][1:])
            text = (folder/(label+".txt")).read_text(errors="replace")
            complete, cycles = flash_completion(text,receipt["exit_code"],point["launches"])
            value = dict(status="completed" if complete else "timeout" if receipt["timed_out"] else "failed_or_incomplete",
                         complete=complete,receipt=str(folder/(label+".receipt.json")),kernel_rows=cycles,
                         prediction_ms=cycles[-1]["cumulative_cycles"]/1_500_000 if complete else None,
                         cpus=slot["cpus"],clock_mhz=1500)
            if not complete:value["reason"] = "\n".join(text.splitlines()[-12:])[-2400:]
        except Exception as exc:value=dict(status="adapter_error",reason=str(exc),complete=False)
        lane.write(destination,value)
        return value

    def gcom_point(self, point, slot):
        folder = self.root/"points"/point["case_key"]
        destination = folder/"gcom.json"
        if destination.exists():return load(destination)
        if point.get("trace_status") != "qualified":
            value = dict(status="trace_not_qualified",reason=point.get("reason"),complete=False)
        else:
            try:
                replay = lane.simulate_trace(self.simulator,self.library,self.configs,
                    point["trace_dir"],point["trace_info"],root=self.root/"replay-cache",cpus=slot["cpus"],timeout=7200)
                value = dict(status="completed" if replay["complete"] else "timeout" if replay["timed_out"] else "failed_or_incomplete",
                    complete=replay["complete"],replay=replay,
                    prediction_ms=replay["kernel_rows"][-1]["cumulative_cycles"]/1_800_000 if replay["complete"] else None,
                    clock_mhz=1800)
                lane.compress_replay_log(replay)
                value["trace_archive"] = resources.archive_trace(point["trace_dir"],point["trace_info"],folder/"trace.tar.gz")
            except ValueError as exc:
                value = dict(status="unsupported_sidecar_contract" if "sidecar" in str(exc) or "TMA" in str(exc) else "adapter_error",
                             reason=str(exc),complete=False)
            except Exception as exc:value=dict(status="adapter_error",reason=str(exc),complete=False)
        lane.write(destination,value)
        return value

    def replay(self, points, scope):
        eligible = [p for p in points if p.get("hardware_status") == "completed"]
        if not eligible:return
        allocation = resources.cpu_admission(self.root/(scope+"-replay-resources"),point_count=48,threads=4)
        if not allocation["slots"]:
            self.status("waiting_for_cpu",scope=scope)
            return
        pool = resources.ReplayCpuPool(allocation["slots"],fallback_threads=4)
        def worker(point,mode):
            slot = pool.acquire(4)
            try:
                result = self.gcom_point(point,slot) if mode == "gcom" else self.flash_point(point,mode,slot)
                print(lane.utc(),mode,point["case"]["kernel_id"],result["status"],flush=True)
                self.report()
            finally:pool.release(slot)
        with ThreadPoolExecutor(max_workers=min(48,len(allocation["slots"]))) as executor:
            jobs = [executor.submit(worker,p,mode) for p in eligible for mode in ("ptx","sass-timing","gcom")]
            for job in as_completed(jobs):job.result()

    def report(self, *, figures=False):
        from .report import generate
        with self.lock:generate(self.root,figures=figures)

    def recover_cublas(self):
        """Capture-only changed-tool recovery; prior model inputs never existed."""
        self.validate()
        tools=self.root/'disassembly-tools'
        tools.mkdir(exist_ok=True)
        wrapper=tools/'cuobjdump'
        if not wrapper.exists():
            wrapper.write_text('#!'+sys.executable+'\nfrom amora.backends.gcom_cuda.cubin_disassembly import main\nimport sys\nsys.exit(main())\n')
            wrapper.chmod(0o755)
        for key in self.manifest['pilot_keys']:
            folder=self.root/'points'/key
            hardware_path=folder/'hardware.json'
            if not hardware_path.exists():continue
            point=load(hardware_path)
            if 'gemm' not in point['case']['kernel_id'] or point.get('trace_status')=='qualified':continue
            if point.get('hardware_status')!='completed':continue
            if (folder/'std-elf-trace.receipt.json').exists():continue
            devices=lane.observe_idle_pool(folder/'std-elf-gpu-observation',only_uuid=point['gpu_uuid'],quiet_seconds=30,timeout=90)
            if not devices:continue
            allocation=resources.cpu_admission(folder/'std-elf-resources',devices=devices,point_count=0)
            trace=folder/'trace-std-elf'
            with lane.GpuLease(self.root/'resource-leases',devices[0]) as lease:
                env=lane.hardware_env(point['gpu_uuid'],self.cuda,self.tracer,trace)
                env.update(PATH=str(tools)+':'+env['PATH'],AMORA_DISASSEMBLY_RECEIPTS=str(folder/'std-elf-disassembly-receipts'))
                receipt=lane.capture_hardware(point['command'],lease=lease,root=folder,label='std-elf-trace',
                    env=env,timeout=1800,cpus=allocation['acquisition_cpus'][0],disk_reserve_bytes=RESERVE)
            if receipt['exit_code'] or not receipt['dedicated_interval']:
                print(lane.utc(),'Std ELF recapture failed',point['case']['kernel_id'],flush=True)
                continue
            try:
                info=lane.inspect_trace(trace,self.schema,point['launches'],expected_kernel_names=point['names'])
                if any(n['grid']!=t['grid'] or n['block']!=t['block'] for n,t in zip(point['kernels'],info['kernels'])):
                    raise ValueError('recovered native/trace launch dimensions differ')
            except Exception as exc:
                lane.write(folder/'std-elf-admission-failure.json',dict(reason=str(exc),timestamp=lane.utc()))
                continue
            lane.write(folder/'hardware-before-std-elf-recovery.json',point)
            point.update(trace_info=info,trace_dir=str(trace),trace_status='qualified',
                trace_recovery=dict(timestamp=lane.utc(),adapter_sha256=lane.sha(Path(__file__).parents[2]/'backends/gcom_cuda/cubin_disassembly.py'),
                    changed_condition='Std ELF nvdisasm rendering with every 128-bit instruction checked against original cubin',
                    model_parameters_changed=False,native_references_unchanged=True))
            point.pop('reason',None)
            lane.write(hardware_path,point)
            old=folder/'gcom.json'
            if old.exists() and load(old).get('status')=='trace_not_qualified':
                old.rename(folder/'gcom-before-std-elf-recovery.json')
            print(lane.utc(),'Std ELF qualified',point['case']['kernel_id'],info['dynamic_warp_instructions'],flush=True)

    def execute(self):
        self.prepare()
        with (self.root/"controller.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.status("family_pilot")
            keys = self.manifest["pilot_keys"]
            while not all((self.root/"points"/k/"hardware.json").exists() for k in keys):
                points = self.capture(keys,"pilot")
                self.replay(points,"pilot")
            self.replay([load(self.root/"points"/k/"hardware.json") for k in keys],"pilot")
            # Expansion is a separate reviewed, cost-bounded stage. Preserve
            # unavailable implementations instead of launching every failing shape.
            supported = set()
            for key in keys:
                folder = self.root/"points"/key
                if any((folder/f).exists() and load(folder/f).get("complete") for f in
                       ("flash-ptx.json","flash-sass-timing.json","gcom.json")):
                    supported.add(self.cases[key]["kernel_id"])
            candidates = [k for k in self.manifest["primary_keys"] if k not in keys
                          and self.cases[k]["kernel_id"] in supported and work(self.cases[k])<=200_000_000]
            # Fair round-robin family selection, then increasing source cost.
            groups = {s.kernel_id:sorted((k for k in candidates if self.cases[k]["kernel_id"]==s.kernel_id),
                                        key=lambda k:work(self.cases[k])) for s in CANONICAL_KERNELS}
            selected=[]
            while any(groups.values()) and len(selected)<128:
                for group in groups.values():
                    if group and len(selected)<128:selected.append(group.pop(0))
            lane.write(self.root/"pilot-review.json",dict(timestamp=lane.utc(),supported_families=sorted(supported),
                decision="expand only family-supported, bounded-source-work cases",selected_keys=selected,
                remaining_primary=len(self.manifest["primary_keys"])-len(selected),
                timing_model_changed=False,formal_accuracy_gate="not_evaluated",max_source_work=200_000_000))
            sizes=[load(p).get('trace_info',{}).get('bytes',0) for p in (self.root/'points').glob('*/hardware.json')]
            observed=max(sizes+[1])
            width=min(64,max(1,int((shutil.disk_usage(self.root).free-RESERVE)*.7/observed)))
            lane.write(self.root/'expansion-resource-policy.json',dict(timestamp=lane.utc(),wave_cases=width,
                replay_threads=4,max_instances=48,observed_max_trace_bytes=observed,disk_reserve_bytes=RESERVE,
                reason='Fill independent replay slots while reserving storage from observed pilot trace size; no timing change'))
            for wave in range(0,len(selected),width):
                if shutil.disk_usage(self.root).free < RESERVE+8*1024**3:
                    self.status("storage_backpressure",remaining=len(selected)-wave)
                    break
                scope=f"expansion-{wave//width:04d}"
                self.status("expansion",wave=scope,total_selected=len(selected))
                points = self.capture(selected[wave:wave+width],scope)
                self.replay(points,scope)
            self.status("bounded_campaign_finished",primary_manifest_cases=2500,expanded_selected=len(selected),
                        note="Unexecuted primary shapes remain resource-deferred; see case ledger and report.")
            self.report(figures=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","execute","execute-cpu","execute-bounded","report","recover-cublas"))
    parser.add_argument("--root",type=Path,required=True)
    args = parser.parse_args()
    campaign = Campaign(args.root)
    if args.action == "prepare":campaign.prepare()
    elif args.action == "execute":
        from .full_campaign import FullCampaign
        FullCampaign(args.root).execute_full()
    elif args.action == "execute-bounded":campaign.execute()
    elif args.action == "execute-cpu":
        from .cpu_campaign import CpuCampaign
        CpuCampaign(args.root).execute_cpu()
    elif args.action == "recover-cublas":campaign.recover_cublas()
    else:campaign.report(figures=True)


if __name__ == "__main__":main()
