"""Device-matched portable v2 GCoM capture, replay and coverage accounting."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import importlib
import gzip
import json
import os
from pathlib import Path
import statistics
import shutil
import sys

from amora.backends.gcom_cuda import portable_v2 as lane
from amora.backends.gcom_cuda import portable_v2_resources as resources
from amora.backends.nvidia.portable_cuda_v2 import pilot_indices
from .corpus import OPS,verify_corpus


def qualify(values, *, floor_ms=.01):
    if len(values)<30 or not all(x>0 for x in values):return False
    med=statistics.median(values)
    return med>=floor_ms and statistics.median(abs(x-med) for x in values)/med<=.05


def repeat_quality(runs, *, floor_ms=.01):
    medians=[statistics.median(r) for r in runs]
    median=statistics.median(medians)
    return dict(median_ms=median,repeat_medians_ms=medians,
        qualified=len(runs)==3 and all(qualify(r,floor_ms=floor_ms) for r in runs)
        and statistics.median(abs(x-median) for x in medians)/median<=.05)


def command(binary,corpus,folder,selected,label, *, samples,warmup,batch=1):
    selection=folder/(label+".indices")
    with selection.open("x") as stream:stream.write("\n".join(map(str,selected))+"\n")
    result=folder/(label+".cases.jsonl")
    argv=[binary,"--corpus="+str(corpus),"--selection="+str(selection),"--result="+str(result),
            "--device=0","--samples="+str(samples),"--warmup="+str(warmup)]
    if batch!=1:argv.append("--batch="+str(batch))
    return argv,result


def witness_indices(manifest):
    selected=set(pilot_indices(manifest))
    cases=manifest["cases"]
    selected.add(max(cases,key=lambda c:c["shape"]["M"]*c["shape"]["N"])["index"])
    for suite in ("h3-semantics","h3-programs"):
        cohort=[c for c in cases if any(a[0]==suite for a in c["aliases"])]
        for key in (lambda c:len(c["program"]["nodes"]),lambda c:len(c["program"]["state_edges"]),
                    lambda c:c["shape"]["M"]*c["shape"]["N"]):
            selected.add(max(cohort,key=key)["index"])
    return sorted(selected)


def balanced_assignments(manifest,selected,devices):
    assignments={d["uuid"]:[] for d in sorted(devices,key=lambda d:d["uuid"])}
    keys=list(assignments)
    ordered=sorted(selected,key=lambda i:(manifest["cases"][i]["aliases"],
        manifest["cases"][i]["shape"]["M"]*manifest["cases"][i]["shape"]["N"],
        manifest["cases"][i]["program_sha256"]))
    for n,index in enumerate(ordered):assignments[keys[n%len(keys)]].append(index)
    return assignments


class Campaign:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.baseline=self.root.parent/"flashgpu-portable-v2"
        self.manifest=verify_corpus(self.baseline/"corpus")
        self.admission=json.loads((self.root/"execution-admission.json").read_text())
        self.binary=self.admission["native_binary"]["binary"]
        if lane.sha(self.binary)!=self.admission["native_binary"]["binary_sha256"]:raise ValueError("native build drift")
        if lane.sha(self.baseline/"corpus/manifest.json")!=self.admission["corpus_sha256"]:raise ValueError("corpus identity drift")
        for path,digest in self.admission["tools"].items():
            if lane.sha(path)!=digest:raise ValueError("admitted tool drift: "+path)
        self.source=self.root/"build-source/simulator-remodeled"
        self.tracer=self.source/"util/tracer_nvbit/tracer_tool/tracer_tool.so"
        self.collector=self.root/"measurement/portable-cupti.so"
        self.configs=[self.source/"gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config",
            self.source/"gpu-simulator/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/trace.config"]
        expected=["82d51b95a560a33bff65631f729d049afe6b33c17de3a62c137bc14b866c2572",
            "a4d2d01079241db28ee3601c7a25adacdf64dd5a6656560fce893493ce77fbf1"]
        if [lane.sha(p) for p in self.configs]!=expected:raise ValueError("H100 config drift")
        libraries=[p.parent for p in (self.source/"gpu-simulator/gpgpu-sim/lib").rglob("libcudart.so")]
        if len(libraries)!=1:raise ValueError("ambiguous simulator library")
        self.library=libraries[0]
        sys.path.insert(0,str(self.root/"schema"))
        self.schema=importlib.import_module("trace_pb2")
        self.batch=1
        self.controller_sha256=lane.sha(Path(__file__))
        self.capture_adapter_sha256=lane.sha(Path(lane.__file__))
        self.anchor_indices=sorted({0,13,max(self.manifest["cases"],key=lambda c:c["shape"]["M"]*c["shape"]["N"])["index"],
            max(self.manifest["cases"],key=lambda c:len(c["program"]["nodes"])*c["shape"]["M"]*c["shape"]["N"])["index"]})

    def status(self,scope,state,**kwargs):
        value=dict(state=state,pid=os.getpid(),timestamp=lane.utc(),controller_sha256=self.controller_sha256,
            capture_adapter_sha256=self.capture_adapter_sha256,**kwargs)
        lane.write(self.root/(scope+"-status.json"),value)
        print(lane.utc(),scope,state,kwargs.get("message",""),flush=True)

    def resource_policy(self):
        policy=dict(replay_threads=8,large_replay_threads=32,max_instances=24,wave_cases=24)
        path=self.root/"runtime-policy.json"
        if path.exists():
            supplied=json.loads(path.read_text())
            policy.update({key:supplied[key] for key in policy if key in supplied})
        if any(type(value) is not int or value<1 for value in policy.values()):
            raise ValueError("resource policy requires positive integer limits")
        return policy

    def hardware(self,device,indices,folder,cpus, *, anchors=False):
        folder.mkdir(parents=True,exist_ok=True)
        result=[]
        with lane.GpuLease(self.root/"resource-leases",device) as lease:
            native,activity=[],[]
            env=lane.hardware_env(device["uuid"],"/usr/local/cuda-13.3")
            native_selected=sorted(set(indices+self.anchor_indices if anchors else indices))
            for repeat in range(3):
                label=f"native-r{repeat}"
                argv,output=command(self.binary,self.baseline/"corpus",folder,native_selected,label,samples=30,warmup=3,batch=self.batch)
                receipt=lane.capture_hardware(argv,lease=lease,root=folder,label=label,env=env,timeout=1800,cpus=cpus)
                self.check_receipt(receipt,folder,device)
                rows=lane.driver_rows(output,self.manifest,native_selected)
                if any(not row["passed"] or len(row["samples_ms"])!=30 for row in rows):raise RuntimeError("native numerical/sample failure")
                native.append(dict(receipt=receipt,rows=rows))
            lane.write(folder/"native.json",dict(gpu=device,native=native,activity=[],binary_sha256=lane.sha(self.binary)))
            for repeat in range(3):
                label=f"activity-r{repeat}"
                path=folder/(label+".activity.jsonl")
                activity_env=dict(env,LD_PRELOAD=str(self.collector),AMORA_CUPTI_JSONL=str(path))
                argv,output=command(self.binary,self.baseline/"corpus",folder,native_selected,label,samples=30,warmup=3,batch=self.batch)
                receipt=lane.capture_hardware(argv,lease=lease,root=folder,label=label,env=activity_env,timeout=1800,cpus=cpus)
                self.check_receipt(receipt,folder,device)
                rows=lane.driver_rows(output,self.manifest,native_selected)
                if any(not r["passed"] for r in rows):raise RuntimeError("CUPTI native numerical failure")
                body=lane.activity_cases(path,{i:len(self.manifest["cases"][i]["program"]["nodes"]) for i in native_selected},samples=30,repetitions=self.batch)
                activity.append(dict(receipt=receipt,rows=rows,bodies=body,activity_sha256=lane.sha(path)))
            lane.write(folder/"native.json",dict(gpu=device,native=native,activity=activity,binary_sha256=lane.sha(self.binary)))
            for index in indices:
                case=self.manifest["cases"][index]
                label=f"trace-{index:04d}"
                trace_folder=folder/label/"traces";trace_folder.mkdir(parents=True,exist_ok=True)
                argv,output=command(self.binary,self.baseline/"corpus",folder,[index],label,samples=1,warmup=3,batch=self.batch)
                trace_env=lane.hardware_env(device["uuid"],"/usr/local/cuda-13.3",self.tracer,trace_folder)
                receipt=lane.capture_hardware(argv,lease=lease,root=folder,label=label,env=trace_env,timeout=1800,cpus=cpus)
                self.check_receipt(receipt,folder,device)
                log=(folder/receipt["output"]).read_text(errors="replace")
                if "Error. Kernel" in log or "Warning. Failed to extract cubin" in log:
                    raise RuntimeError("tracer static cubin/register metadata qualification failed")
                rows=lane.driver_rows(output,self.manifest,[index],instrumented=True)
                if not rows[0]["passed"]:raise RuntimeError("traced numerical failure")
                nodes=len(case["program"]["nodes"])
                info=lane.inspect_trace(trace_folder,self.schema,(1+4*self.batch)*nodes,
                    expected_operator_ids=[OPS.index(n["op"]) for n in case["program"]["nodes"]]*(1+4*self.batch))
                native_runs=[next(r for r in run["rows"] if r["index"]==index)["samples_ms"] for run in native]
                body_runs=[run["bodies"][index]["samples_ms"] for run in activity]
                value=dict(state="trace_complete",case_index=index,case_id=case["id"],gpu=device,
                    native_folder=str(folder),native_quality=repeat_quality(native_runs,floor_ms=.1 if self.batch>1 else .01),
                    body_quality=repeat_quality(body_runs,floor_ms=.1 if self.batch>1 else .01),batch_repetitions=self.batch,
                    native_correct=True,trace_complete=True,replay_admitted=False,timing_qualified=False,
                    trace_receipt=receipt,trace_info=info,trace_folder=str(trace_folder),nodes=nodes,
                    case_sha256=case["case_sha256"],program_sha256=case["program_sha256"],native_binary_sha256=lane.sha(self.binary))
                lane.write(folder/(label+".hardware.json"),value)
                result.append(value)
                print(lane.utc(),"trace complete",index,"GPU",device["index"],info["bytes"],flush=True)
        return result

    @staticmethod
    def check_receipt(receipt,folder,device):
        if receipt["exit_code"]!=0 or not receipt["dedicated_interval"]:
            raise RuntimeError("hardware interval unqualified: "+receipt["label"])
        lane.verify_placement(folder/receipt["output"],device["uuid"])

    def capture(self,scope,selected, *, timeout=3600):
        self.status(scope,"waiting_for_dedicated_gpu",cases=selected)
        devices=lane.observe_idle_pool(self.root/(scope+"-gpu-observation"),quiet_seconds=30,timeout=timeout)
        if not devices:return []
        assignments=balanced_assignments(self.manifest,selected,devices)
        admission=resources.cpu_admission(self.root/(scope+"-resources"),devices=devices)
        lane.write(self.root/(scope+"-assignments.json"),dict(timestamp=lane.utc(),devices=devices,assignments=assignments,
            acquisition_cpus=admission["acquisition_cpus"],policy="balanced sorted suite/shape/program round robin; all repeats remain device-bound"))
        self.status(scope,"hardware",devices=devices,cases=selected)
        prepared,errors=[],[]
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures={pool.submit(self.hardware,d,assignments[d["uuid"]],self.root/scope/d["uuid"],cpus,
                       anchors=scope=="pilot" or scope.startswith("full/")):d for d,cpus in zip(devices,admission["acquisition_cpus"])}
            for future in as_completed(futures):
                try:prepared.extend(future.result())
                except Exception as exc:
                    errors.append(dict(gpu=futures[future],error=repr(exc)))
                    print(lane.utc(),"hardware admission error",repr(exc),flush=True)
        # Individual complete receipts survive a later device worker failure.
        by_case={p["case_index"]:p for p in prepared}
        for path in (self.root/scope).glob("GPU-*/*.hardware.json"):
            point=json.loads(path.read_text());by_case.setdefault(point["case_index"],point)
        for device in devices:
            for index in assignments[device["uuid"]]:
                if index in by_case:continue
                case=self.manifest["cases"][index]
                failure=dict(state="hardware_not_completed",case_index=index,case_id=case["id"],gpu=device,
                    native_correct=None,trace_complete=False,replay_admitted=False,timing_qualified=False,
                    case_sha256=case["case_sha256"],program_sha256=case["program_sha256"],
                    errors=[e for e in errors if e["gpu"]["uuid"]==device["uuid"]])
                lane.write(self.root/scope/(f"case-{index:04d}.result.json"),failure)
        lane.write(self.root/(scope+"-hardware-summary.json"),dict(points=list(by_case.values()),errors=errors,selected=selected))
        return list(by_case.values())

    def replay(self,scope,prepared):
        if not prepared:return []
        policy=self.resource_policy()
        allocation=resources.cpu_admission(self.root/(scope+"-replay-resources"),
            point_count=min(len(prepared),policy["max_instances"]),threads=policy["replay_threads"])
        slots=allocation["slots"]
        known_rss=[json.loads(p.read_text()).get("max_rss_kib",0) or 0 for p in (self.root/"replay-cache").glob("*/result.json")]
        rss=max(known_rss+[4*1024**2])*1024
        memory_slots=max(0,int((allocation["memory_available_bytes"]-16*1024**3)/(rss*1.5)))
        slots=slots[:memory_slots]
        if not slots:
            self.status(scope,"waiting_for_cpu_capacity",message="no sufficiently idle CPU slot")
            return []
        self.status(scope,"cpu_replay",traces=len(prepared),workers=len(slots),resource_policy=policy,
            admitted_logical_cpus=sum(slot["omp_threads"] for slot in slots))
        cpu_pool=resources.ReplayCpuPool(slots,fallback_threads=policy["replay_threads"])
        def worker(slot,points):
            results=[]
            for point in points:
                try:
                    replay=lane.simulate_trace(self.source/"gpu-simulator/bin/release/accel-sim.out",self.library,self.configs,
                        Path(point["trace_folder"]),point["trace_info"],root=self.root/"replay-cache",cpus=slot["cpus"],timeout=7200)
                except Exception as exc:
                    point.update(state="replay_exception",error=repr(exc),replay_admitted=False)
                    lane.write(self.root/scope/(f"case-{point['case_index']:04d}.result.json"),point)
                    results.append(point)
                    continue
                point["replay"]=replay
                point["replay_admitted"]=replay["complete"]
                if replay["complete"]:
                    rows=replay["kernel_rows"];n=point["nodes"]*point.get("batch_repetitions",1)
                    target=rows[-n:]
                    point["modeled_target_cycles"]=sum(r["cycles"] for r in target)
                    point["modeled_target_ms"]=point["modeled_target_cycles"]/1.8e6
                    point["timing_qualified"]=point["native_quality"]["qualified"]
                    point["timing_policy"]="sequential target service cycles including configured CTA launch delay; zero host submission gaps"
                    point["body_comparison_status"]="not_comparable: model service cycles include CTA launch delay"
                point["state"]="replay_complete" if replay["complete"] else "replay_failed"
                lane.write(self.root/scope/(f"case-{point['case_index']:04d}.result.json"),point)
                results.append(point)
                print(lane.utc(),scope,point["state"],point["case_index"],replay["elapsed_seconds"],flush=True)
            return results
        def run_point(point):
            preferred=policy["large_replay_threads"] if point["trace_info"]["bytes"]>=16*1024**2 else policy["replay_threads"]
            slot=cpu_pool.acquire(preferred)
            try:
                point["cpu_allocation"]=dict(slot,preferred_threads=preferred,resource_policy=policy,
                    policy="runtime-admitted thread limits; fill smaller remainders; NUMA-local disjoint CPU sets")
                return worker(slot,[point])
            finally:cpu_pool.release(slot)
        results=[]
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            futures=[pool.submit(run_point,point) for point in sorted(prepared,key=lambda p:p["trace_info"]["bytes"],reverse=True)]
            for f in as_completed(futures):results.extend(f.result())
        return results

    def run(self,scope, *, timeout=3600):
        selected=[0,13] if scope=="pilot" else witness_indices(self.manifest)
        saved=self.saved_points()
        reused=[p for i,p in saved.items() if i in selected and p["replay_admitted"]]
        pending=[p for i,p in saved.items() if i in selected and p.get("state")=="trace_complete"]
        failures=[p for i,p in saved.items() if i in selected and not p["replay_admitted"] and p.get("state")!="trace_complete"]
        known={p["case_index"] for p in reused+failures}
        prepared=list(pending)
        known.update(p["case_index"] for p in pending)
        for path in (self.root/scope).glob("GPU-*/*.hardware.json"):
            point=json.loads(path.read_text())
            if point["case_index"] in selected and point["case_index"] not in known:
                prepared.append(point);known.add(point["case_index"])
        remaining=[i for i in selected if i not in known]
        if remaining:prepared+=self.capture(scope,remaining,timeout=timeout)
        points=reused+failures+self.replay(scope,prepared)
        complete=len(points)==len(selected) and all(p["replay_admitted"] for p in points)
        value=dict(state="qualified" if complete else "incomplete",points=points,selected=selected,
            timestamp=lane.utc(),scope="pipeline bootstrap" if scope=="pilot" else "operator/layout/H3 witnesses",
            metric="native stream program span vs fixed-config sequential simulator service timeline")
        lane.write(self.root/(scope+"-summary.json"),value)
        self.status(scope,value["state"],completed=len(points),selected=len(selected))
        return value

    def saved_points(self):
        result={}
        for scope in ("pilot","witnesses","full"):
            for path in (self.root/scope).rglob("case-*.result.json"):
                point=json.loads(path.read_text())
                index=point["case_index"]
                if index not in result or point.get("replay_admitted"):result[index]=dict(point,result_path=str(path))
        return result

    def full(self, *, timeout=3600):
        pilot=json.loads((self.root/"witnesses-summary.json").read_text())
        if pilot["state"]!="qualified":raise RuntimeError("operator/layout/H3 witness pilot is not qualified")
        points=self.saved_points()
        attempted=set(points)
        summaries=list((self.root/"full").glob("wave-*-hardware-summary.json"))
        summaries+=[p for p in (self.root/"full").glob("wave-*-hardware-summary.json.gz") if not p.with_suffix("").exists()]
        for path in summaries:
            content=gzip.open(path,"rt").read() if path.suffix==".gz" else path.read_text()
            attempted.update(json.loads(content)["selected"])
        pending=[i for i in range(len(self.manifest["cases"])) if i not in attempted]
        max_trace=max(p["trace_info"]["bytes"] for p in points.values() if p.get("trace_info"))
        wave=0
        while pending:
            while (self.root/"full"/(f"wave-{wave:04d}-status.json")).exists():wave+=1
            disk=shutil.disk_usage(self.root).free
            storage_slots=int((disk-8*1024**3)/max(1,1.5*max_trace))
            if storage_slots<1:
                self.status("full","waiting_for_storage_capacity",remaining=len(pending),disk_free_bytes=disk)
                break
            # A bounded wave fills the measured CPU pool without allowing a
            # faster GPU producer to consume the workspace disk reserve.
            policy=self.resource_policy()
            allocation=resources.cpu_admission(self.root/"full-resources",point_count=policy["max_instances"],threads=policy["replay_threads"])
            if not allocation["slots"]:
                self.status("full","waiting_for_cpu_capacity",remaining=len(pending));break
            count=min(len(pending),storage_slots,policy["wave_cases"])
            selected,pending=pending[:count],pending[count:]
            scope=f"full/wave-{wave:04d}"
            prepared=self.capture(scope,selected,timeout=timeout)
            if not prepared:
                self.status("full","waiting_for_dedicated_gpu",remaining=len(pending)+len(selected))
                break
            outcomes=self.replay(scope,prepared)
            for point in outcomes:
                if "replay" in point:
                    lane.preserve_replay_auxiliary(point["replay"],point["trace_folder"],point["trace_info"])
                    lane.write(Path(point["replay"]["folder"])/"result.json",point["replay"])
                    lane.compress_replay_log(point["replay"])
                archive=self.root/"trace-archives"/(f"case-{point['case_index']:04d}.tar.gz")
                point["trace_archive"]=resources.archive_trace(point["trace_folder"],point["trace_info"],archive)
                point["trace_folder_removed_after_verified_archive"]=True
                lane.write(self.root/scope/(f"case-{point['case_index']:04d}.result.json"),point)
            self.status("full","wave_accounted",wave=wave,remaining=len(pending),outcomes=len(outcomes))
            from .gcom_report import report
            report(self)
            wave+=1
        points=self.saved_points()
        lane.write(self.root/"full-summary.json",dict(timestamp=lane.utc(),points=list(points.values()),
            selected=len(self.manifest["cases"]),accounted=len(points),
            state="accounted" if len(points)==len(self.manifest["cases"]) else "incomplete"))
        return points

    def short_h3(self, *, timeout=3600):
        points=self.saved_points()
        if len(points)!=len(self.manifest["cases"]):raise RuntimeError("single-invocation panel is not fully accounted")
        selected=[i for i,p in points.items() if p.get("native_correct")
            and any(a[0].startswith("h3-") for a in self.manifest["cases"][i]["aliases"])
            and (p.get("native_quality",{}).get("median_ms",1)<.01 or
                 (len(self.manifest["cases"][i]["program"]["nodes"])<=8 and
                  self.manifest["cases"][i]["shape"]["M"]*self.manifest["cases"][i]["shape"]["N"]<=16384))]
        from amora.backends.nvidia.portable_v2_batch import build_batch
        variant=build_batch(self.root/"batch-build",Path(__file__).parents[2]/"backends/nvidia/portable_core_v2.cu",
            self.admission["native_binary"]["contract"])
        self.binary=variant["binary"];self.batch=64
        lane.write(self.root/"h3-batch-selection.json",dict(selected=selected,batch=64,variant=variant,
            rule="H3 source fixtures with <=8 nodes and <=16384 shape elements, plus any H3 row below the 10us native floor; no simulator error used for selection"))
        outcomes=[];pending=list(selected);wave=0
        estimated=max([points[i].get("trace_info",{}).get("bytes",0)*257/5 for i in selected]+[1])
        while pending:
            slots=int((shutil.disk_usage(self.root).free-8*1024**3)/(estimated*1.5))
            if slots<1:
                self.status("h3-batch","waiting_for_storage_capacity",remaining=len(pending));break
            count=min(self.resource_policy()["wave_cases"],slots)
            indices,pending=pending[:count],pending[count:]
            scope=f"h3-batch/wave-{wave:04d}"
            prepared=self.capture(scope,indices,timeout=timeout)
            if not prepared:break
            results=self.replay(scope,prepared)
            for point in results:
                point["amortized_native_ms"]=point["native_quality"]["median_ms"]/64
                if point.get("modeled_target_ms"):point["amortized_prediction_ms"]=point["modeled_target_ms"]/64
                if "replay" in point:
                    lane.preserve_replay_auxiliary(point["replay"],point["trace_folder"],point["trace_info"])
                    lane.write(Path(point["replay"]["folder"])/"result.json",point["replay"])
                    lane.compress_replay_log(point["replay"])
                point["trace_archive"]=resources.archive_trace(point["trace_folder"],point["trace_info"],
                    self.root/"trace-archives"/(f"h3-batch-case-{point['case_index']:04d}.tar.gz"))
                lane.write(self.root/scope/(f"case-{point['case_index']:04d}.result.json"),point)
            outcomes.extend(results);wave+=1
            from .gcom_report import report
            report(self)
        lane.write(self.root/"h3-batch-summary.json",dict(timestamp=lane.utc(),selected=selected,points=outcomes,
            state="accounted" if len(outcomes)==len(selected) else "incomplete",batch_repetitions=64))
        self.batch=1;self.binary=self.admission["native_binary"]["binary"]
        return outcomes

    def execute(self, *, timeout=3600):
        from .gcom_report import report
        self.status("execution","source_admitted")
        try:
            for scope in ("pilot","witnesses"):
                summary=self.root/(scope+"-summary.json")
                state=json.loads(summary.read_text()) if summary.exists() else {}
                if state.get("state")!="qualified":
                    self.status("execution",scope,message="waiting for resource-admitted capture and replay")
                    state=self.run(scope,timeout=timeout)
                report(self)
                if state["state"]!="qualified":
                    self.status("execution","pending_pilot_admission",phase=scope)
                    return 75
            self.status("execution","full_panel")
            self.full(timeout=timeout);report(self)
            if len(self.saved_points())!=len(self.manifest["cases"]):
                self.status("execution","pending_full_panel")
                return 75
            self.status("execution","short_h3_batch")
            self.short_h3(timeout=timeout);report(self)
            batch=json.loads((self.root/"h3-batch-summary.json").read_text())
            complete=batch["state"]=="accounted"
            self.status("execution","accounted" if complete else "pending_h3_batch")
            return 0 if complete else 75
        except Exception as exc:
            self.status("execution","execution_error",error=repr(exc))
            report(self)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("execute","pilot","witnesses","full","short-h3","report"))
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--gpu-wait-seconds",type=int,default=3600)
    args=parser.parse_args()
    campaign=Campaign(args.root)
    if args.action=="execute":
        return campaign.execute(timeout=args.gpu_wait_seconds)
    if args.action=="report":
        from .gcom_report import report
        report(campaign)
        return
    if args.action=="full":
        campaign.full(timeout=args.gpu_wait_seconds)
        return
    if args.action=="short-h3":
        campaign.short_h3(timeout=args.gpu_wait_seconds)
        return
    if args.action=="witnesses":
        pilot=json.loads((args.root/"pilot-summary.json").read_text())
        if pilot["state"]!="qualified":raise RuntimeError("pipeline pilot is not qualified")
    campaign.run(args.action,timeout=args.gpu_wait_seconds)


if __name__=="__main__":sys.exit(main())
