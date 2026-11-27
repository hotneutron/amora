"""Resume every primary canonical case without workload-score exclusion."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from amora.backends.gcom_cuda import portable_v2 as lane
from amora.backends.gcom_cuda import portable_v2_resources as resources
from amora.benchmarking.schema import BenchmarkCase
from benchmark_generators.ppp_canonical.replay import contract_for_case
from .campaign import Campaign, activity, matched_native, load, RESERVE


def primary_keys(manifest):
    keys=list(manifest['primary_keys'])
    if len(keys)!=len(set(keys)):
        raise ValueError('primary manifest has duplicate case identities')
    return keys


def require_gpu_approval(root, gpu_uuid=None):
    path=Path(root)/'full-study/gpu-launch-approval.json'
    record=load(path) if path.exists() else {}
    if record.get('state')!='approved' or record.get('authority')!='explicit_user_approval':
        raise PermissionError('GPU jobs are paused pending explicit user approval')
    if record.get('scope')!='all_dedicated_idle_h100':
        allowed=record.get('allowed_gpu_uuids',[])
        if not allowed or gpu_uuid is not None and gpu_uuid not in allowed:
            raise PermissionError('this GPU is outside the approved launch scope')
    return record


def approved_devices(root, devices):
    approval=require_gpu_approval(root)
    if approval.get('scope')=='all_dedicated_idle_h100':return devices
    allowed=set(approval['allowed_gpu_uuids'])
    return [device for device in devices if device['uuid'] in allowed]


def warp_count(kernels):
    return sum(math.prod(k['grid'])*math.ceil(math.prod(k['block'])/32) for k in kernels)


def choose_wave(keys, estimates, free_bytes, *, reserve_bytes=RESERVE, max_cases=64):
    """Bound staging by estimated bytes; retain oversized cases in the queue."""
    budget=max(0,int((free_bytes-reserve_bytes)*.50))
    selected=[];used=0
    for key in keys:
        cost=max(1,estimates[key])
        if len(selected)>=max_cases:break
        if used+cost>budget:continue
        selected.append(key);used+=cost
    if not selected and keys and free_bytes>=reserve_bytes+8*1024**3:
        # Unknown/large cases get a single monitored capture, rather than
        # being permanently rejected by an estimate. The live disk/RAM
        # guards preserve explicit resource-failure evidence if it cannot fit.
        selected=[min(keys,key=lambda key:estimates[key])]
    return selected


class FullCampaign(Campaign):
    def __init__(self,root):
        super().__init__(root)
        self.validate()
        self.keys=primary_keys(self.manifest)
        self.phase_dir=self.root/'full-study'
        self.phase_dir.mkdir(exist_ok=True)

    def publish(self,state,**values):
        with self.lock:self.status(state,primary_total=len(self.keys),**values)

    def pending_native(self):
        return [key for key in self.keys if not (self.root/'points'/key/'hardware.json').exists()]

    @staticmethod
    def check_quiet(lease):
        current=lane.gpu_snapshot(lease.device['uuid'])[0]
        if current['processes'] or current['memory_used']>64*1024**2:
            raise RuntimeError('assigned GPU gained a foreign context before launch')

    def native_case(self,key,lease,cpus):
        folder=self.root/'points'/key
        folder.mkdir(parents=True,exist_ok=True)
        saved=folder/'hardware.json'
        if saved.exists():return load(saved)
        require_gpu_approval(self.root,lease.device['uuid'])
        self.check_quiet(lease)
        started=folder/'full-native-started.json'
        if started.exists():
            value=dict(case_key=key,case=self.cases[key],hardware_status='interrupted_not_retried',
                trace_status='native_not_completed',numerical_correctness='not_evaluated_no_template_oracle')
            lane.write(saved,value)
            return value
        case=BenchmarkCase(**self.cases[key]);build=self.admission['builds'][case.kernel_id]
        command=[build['binary'],*contract_for_case(case).args]
        value=dict(case_key=key,case=asdict(case),command=command,physical_gpu=lease.device['index'],
            gpu_uuid=lease.device['uuid'],binary_sha256=build['binary_sha256'],
            numerical_correctness='not_evaluated_no_template_oracle')
        lane.write(started,dict(timestamp=lane.utc(),pid=os.getpid(),case_key=key,binary_sha256=build['binary_sha256']))
        try:
            if build['exit_code']!=0:raise ValueError('canonical source build failed')
            runs=[]
            for repeat in range(3):
                require_gpu_approval(self.root,lease.device['uuid'])
                self.check_quiet(lease)
                label=f'full-native-{repeat}'
                output=folder/(label+'.activity.jsonl')
                env=lane.hardware_env(lease.device['uuid'],self.cuda)
                env.update(LD_PRELOAD=str(self.collector),AMORA_CUPTI_JSONL=str(output))
                receipt=lane.capture_hardware(command,lease=lease,root=folder,label=label,
                    env=env,timeout=300,cpus=cpus,disk_reserve_bytes=RESERVE)
                if receipt['exit_code'] or not receipt['dedicated_interval']:
                    raise ValueError('native acquisition failed or dedicated interval rejected')
                runs.append(activity(output))
            value.update(matched_native(runs),hardware_status='completed',trace_status='pending_full_trace')
            lane.write(folder/'native.json',value)
        except Exception as exc:
            value.update(hardware_status='not_completed',trace_status='native_not_completed',reason=str(exc))
        lane.write(saved,value)
        return value

    def native_stage(self):
        wave=0
        while self.pending_native():
            pending=self.pending_native()
            self.publish('full_native',pending_primary=len(pending),stage='native_acquisition')
            devices=lane.observe_idle_pool(self.phase_dir/'native-gpu-observation',quiet_seconds=30,timeout=60)
            devices=approved_devices(self.root,devices)
            if not devices:
                self.publish('waiting_for_dedicated_gpu',pending_primary=len(pending))
                continue
            try:allocation=resources.cpu_admission(self.phase_dir/f'native-resources-{wave:04d}',devices=devices,point_count=0)
            except ValueError as exc:
                self.publish('waiting_for_cpu',reason=str(exc),pending_primary=len(pending))
                time.sleep(30)
                continue
            buckets=[[] for _ in devices]
            for i,key in enumerate(pending[:256]):buckets[i%len(devices)].append(key)
            def worker(device,cpus,selected):
                try:
                    with lane.GpuLease(self.root/'resource-leases',device) as lease:
                        for key in selected:
                            point=self.native_case(key,lease,cpus)
                            with self.lock:
                                left=len(self.pending_native())
                                self.status('full_native',primary_total=len(self.keys),pending_primary=left,
                                    last_case=key,last_outcome=point['hardware_status'])
                            print(lane.utc(),'native',key,point['hardware_status'],flush=True)
                            if point['hardware_status']!='completed':break
                except (RuntimeError,BlockingIOError) as exc:
                    print(lane.utc(),'GPU admission wait',device['index'],str(exc),flush=True)
            with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                futures=[pool.submit(worker,d,c,b) for d,c,b in zip(devices,allocation['acquisition_cpus'],buckets) if b]
                for future in as_completed(futures):future.result()
            self.report()
            wave+=1

    def estimate_trace(self,point):
        """Estimate storage from observed trace bytes and source launch geometry."""
        target_warps=warp_count(point['kernels'])
        candidates=[]
        for key in self.manifest['pilot_keys']:
            path=self.root/'points'/key/'hardware.json'
            if not path.exists():continue
            pilot=load(path)
            if pilot['case']['kernel_id']!=point['case']['kernel_id'] or not pilot.get('trace_info'):continue
            warps=warp_count(pilot['kernels'])
            ratio=max(1,target_warps/max(1,warps))
            shape=point['case']['shape'];base=pilot['case']['shape'];family=point['case']['kernel_id']
            if 'gemm' in family:ratio*=max(1,shape['K']/base['K'])
            elif family in ('embedding','rmsnorm','flash_attention_fwd'):
                ratio*=max(1,shape['D']/base['D'])
            candidates.append(pilot['trace_info']['bytes']*ratio*2)
        return int(max(candidates+[64*1024**2]))

    def wrapper(self):
        tools=self.root/'disassembly-tools';tools.mkdir(exist_ok=True)
        path=tools/'cuobjdump'
        if not path.exists():
            path.write_text('#!'+sys.executable+'\nfrom amora.backends.gcom_cuda.cubin_disassembly import main\nimport sys\nsys.exit(main())\n')
            path.chmod(0o755)
        return tools

    def archive_raw(self,point, *, failed=False):
        trace=Path(point.get('trace_dir',''))
        if not point.get('trace_dir') or not trace.is_dir():return
        if failed:
            files={str(p.relative_to(trace)):lane.sha(p) for p in trace.rglob('*') if p.is_file()}
            if not files:return
            info=dict(files=files,trace_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())
        else:info=point['trace_info']
        destination=self.root/'points'/point['case_key']/('full-trace-failed.tar.gz' if failed else 'full-trace.tar.gz')
        point['full_trace_archive']=resources.archive_trace(trace,info,destination)
        lane.write(self.root/'points'/point['case_key']/'hardware.json',point)

    def trace_case(self,point,lease,cpus):
        key=point['case_key'];folder=self.root/'points'/key
        terminal=folder/'gcom.json'
        if terminal.exists():
            prior=load(terminal)
            if prior.get('status')!='trace_not_qualified' or prior.get('replay'):return point
        started=folder/'full-trace-started.json'
        if started.exists():
            lane.write(terminal,dict(status='trace_interrupted_not_retried',complete=False,
                full_capture_attempted=True,reason='capture attempt exists without a complete admitted trace'))
            return point
        require_gpu_approval(self.root,lease.device['uuid'])
        self.check_quiet(lease)
        if point['gpu_uuid']!=lease.device['uuid']:raise ValueError('trace must remain bound to the native reference GPU')
        if terminal.exists():terminal.rename(folder/'gcom-before-full-capture.json')
        lane.write(folder/'hardware-before-full-capture.json',point)
        lane.write(started,dict(timestamp=lane.utc(),pid=os.getpid(),key=key,
            changed_condition='verified Std ELF disassembly adapter in every capture; no previous simulator input is repeated'))
        trace=folder/'trace-full';point['trace_dir']=str(trace)
        try:
            env=lane.hardware_env(point['gpu_uuid'],self.cuda,self.tracer,trace)
            env.update(PATH=str(self.wrapper())+':'+env['PATH'],AMORA_DISASSEMBLY_RECEIPTS=str(folder/'full-disassembly-receipts'),
                INCREMENTAL_FLUSH_THRESHOLD='50000')
            receipt=lane.capture_hardware(point['command'],lease=lease,root=folder,label='full-trace',env=env,
                timeout=1800,cpus=cpus,disk_reserve_bytes=RESERVE,memory_reserve_bytes=64*1024**3)
            if receipt['exit_code'] or not receipt['dedicated_interval']:
                raise ValueError('trace capture failed, timed out, or lost its dedicated interval')
            info=lane.inspect_trace(trace,self.schema,point['launches'],expected_kernel_names=point['names'])
            if any(n['grid']!=t['grid'] or n['block']!=t['block'] for n,t in zip(point['kernels'],info['kernels'])):
                raise ValueError('native/trace launch geometry mismatch')
            point.update(trace_status='qualified',trace_info=info,
                full_capture_adapter_sha256=lane.sha(Path(__file__).parents[2]/'backends/gcom_cuda/cubin_disassembly.py'))
            point.pop('reason',None)
        except Exception as exc:
            point.update(trace_status='full_capture_failed',reason=str(exc))
            lane.write(terminal,dict(status='trace_not_qualified',complete=False,reason=str(exc),full_capture_attempted=True))
            self.archive_raw(point,failed=True)
        lane.write(folder/'hardware.json',point)
        return point

    def prepare_traces(self,keys,wave):
        ready=[];pending=[]
        for key in keys:
            point=load(self.root/'points'/key/'hardware.json')
            file=self.root/'points'/key/'gcom.json'
            if file.exists() and (load(file).get('status')!='trace_not_qualified' or load(file).get('full_capture_attempted')):
                ready.append(point);continue
            if point.get('trace_status')=='qualified' and Path(point['trace_dir']).is_dir():
                ready.append(point);continue
            if point.get('hardware_status')!='completed':
                ready.append(point);continue
            pending.append(point)
        if not pending:return ready
        requested={p['gpu_uuid'] for p in pending}
        devices=lane.observe_idle_pool(self.phase_dir/f'trace-gpu-{wave:04d}',quiet_seconds=30,timeout=60)
        devices=approved_devices(self.root,devices)
        devices=[d for d in devices if d['uuid'] in requested]
        if not devices:return ready
        allocation=resources.cpu_admission(self.phase_dir/f'trace-cpus-{wave:04d}',devices=devices,point_count=0)
        def worker(device,cpus):
            result=[]
            try:
                with lane.GpuLease(self.root/'resource-leases',device) as lease:
                    for point in pending:
                        if point['gpu_uuid']!=device['uuid']:continue
                        result.append(self.trace_case(point,lease,cpus))
                        print(lane.utc(),'trace',point['case_key'],result[-1].get('trace_status'),flush=True)
            except (RuntimeError,BlockingIOError) as exc:
                print(lane.utc(),'trace admission wait',device['index'],str(exc),flush=True)
            return result
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures=[pool.submit(worker,d,c) for d,c in zip(devices,allocation['acquisition_cpus'])]
            for future in as_completed(futures):ready.extend(future.result())
        return ready

    def replay_wave(self,points,wave):
        allocation=resources.cpu_admission(self.phase_dir/f'replay-cpus-{wave:04d}',point_count=48,threads=4)
        if not allocation['slots']:return False
        largest=max([p.get('trace_info',{}).get('bytes',0) for p in points]+[0])
        per_point=max(2*1024**3,3*largest)
        memory_workers=(allocation['memory_available_bytes']-64*1024**3)//per_point
        if memory_workers<1:
            self.publish('waiting_for_memory',pending_primary_models=len(points),required_bytes=per_point)
            return False
        max_workers=min(48,len(allocation['slots']),memory_workers)
        pool=resources.ReplayCpuPool(allocation['slots'],fallback_threads=4)
        def worker(point,mode):
            folder=self.root/'points'/point['case_key']
            file=folder/('gcom.json' if mode=='gcom' else 'flash-'+mode+'.json')
            if point.get('hardware_status')!='completed':
                if not file.exists():lane.write(file,dict(status='native_reference_failed',complete=False,reason=point.get('reason')))
                return
            slot=pool.acquire(4)
            try:
                if mode=='gcom':
                    value=self.gcom_point(point,slot)
                    if point.get('trace_status')=='qualified' and Path(point['trace_dir']).is_dir():self.archive_raw(point)
                else:value=self.flash_point(point,mode,slot)
                print(lane.utc(),mode,point['case_key'],value['status'],flush=True)
            finally:pool.release(slot)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures=[executor.submit(worker,p,m) for p in points for m in ('ptx','sass-timing','gcom')]
            for future in as_completed(futures):future.result()
        return True

    def gcom_point(self,point,slot):
        folder=self.root/'points'/point['case_key'];destination=folder/'gcom.json'
        if destination.exists():return load(destination)
        if point.get('trace_status')!='qualified':
            value=dict(status='trace_not_qualified',complete=False,full_capture_attempted=True,reason=point.get('reason'))
        else:
            try:
                replay=lane.simulate_trace(self.simulator,self.library,self.configs,point['trace_dir'],point['trace_info'],
                    root=self.root/'replay-cache',cpus=slot['cpus'],timeout=7200,disk_reserve_bytes=RESERVE)
                value=dict(status='completed' if replay['complete'] else 'storage_limit' if replay.get('storage_exhausted')
                    else 'timeout' if replay['timed_out'] else 'failed_or_incomplete',complete=replay['complete'],
                    replay=replay,prediction_ms=replay['kernel_rows'][-1]['cumulative_cycles']/1_800_000 if replay['complete'] else None,
                    clock_mhz=1800)
                lane.compress_replay_log(replay)
            except ValueError as exc:
                value=dict(status='unsupported_sidecar_contract' if 'sidecar' in str(exc) or 'TMA' in str(exc) else 'adapter_error',
                    complete=False,reason=str(exc))
            except Exception as exc:value=dict(status='adapter_error',complete=False,reason=str(exc))
        lane.write(destination,value)
        return value

    def pending_models(self):
        result=[]
        for key in self.keys:
            folder=self.root/'points'/key
            if not all((folder/file).exists() for file in ('gcom.json','flash-ptx.json','flash-sass-timing.json')):
                result.append(key);continue
            g=load(folder/'gcom.json')
            if g.get('status')=='trace_not_qualified' and not g.get('full_capture_attempted'):
                result.append(key)
        return result

    def execute_full(self):
        try:require_gpu_approval(self.root)
        except PermissionError:
            self.publish('waiting_for_user_gpu_approval',pending_primary_native=len(self.pending_native()))
            return
        with (self.root/'controller.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            lane.write(self.phase_dir/'admission.json',dict(timestamp=lane.utc(),primary_keys=self.keys,
                primary_total=len(self.keys),native_existing=len(self.keys)-len(self.pending_native()),
                fixed_case_cap=None,work_score_cutoff=None,source=self.admission,
                policy='all primary cases; native first; traces and models staged under CPU/RAM/disk admission'))
            self.native_stage()
            wave=0
            while self.pending_models():
                pending=self.pending_models()
                points={k:load(self.root/'points'/k/'hardware.json') for k in pending}
                approval=require_gpu_approval(self.root)
                if approval.get('scope')!='all_dedicated_idle_h100':
                    allowed=set(approval['allowed_gpu_uuids'])
                    pending=[k for k in pending if points[k].get('hardware_status')!='completed'
                        or points[k].get('gpu_uuid') in allowed]
                    if not pending:
                        self.publish('waiting_for_additional_gpu_approval',pending_primary_models=len(points))
                        time.sleep(30)
                        continue
                estimates={k:self.estimate_trace(p) if p.get('hardware_status')=='completed' else 1 for k,p in points.items()}
                free=shutil.disk_usage(self.root).free
                selected=choose_wave(sorted(pending,key=lambda k:estimates[k]),estimates,free)
                if not selected:
                    self.publish('waiting_for_storage',pending_primary_models=len(pending),free_bytes=free,
                        minimum_estimated_trace_bytes=min(estimates.values()),reserve_bytes=RESERVE)
                    time.sleep(30)
                    continue
                self.publish('full_simulation',wave=wave,selected=len(selected),pending_primary_models=len(pending))
                lane.write(self.phase_dir/f'wave-{wave:04d}.json',dict(keys=selected,estimated_trace_bytes=estimates,
                    free_bytes=free,reserve_bytes=RESERVE,work_cutoff=None))
                prepared=self.prepare_traces(selected,wave)
                if not self.replay_wave(prepared,wave):time.sleep(30)
                self.report()
                wave+=1
            self.publish('full_case_run_finished',primary_total=len(self.keys),primary_native_attempted=len(self.keys),
                pending_primary_models=0,scope='all primary cases have retained native and simulator outcomes; failures remain explicit')
            self.report(figures=True)
            from amora.benchmarking.portable_v2.combined_report import generate
            generate(self.project)
