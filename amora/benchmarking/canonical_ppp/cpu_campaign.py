"""CPU-only predictions for the entire canonical manifest; GPU hold stays set."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import threading
import time

from amora.backends.flashgpu_sim import runner as flash
from amora.backends.gcom_cuda import portable_v2 as lane
from amora.backends.gcom_cuda import portable_v2_resources as resources
from amora.benchmarking.schema import BenchmarkCase
from benchmark_generators.ppp_canonical.replay import contract_for_case
from .campaign import load, flash_completion
from .full_campaign import FullCampaign

PROMPT='launch the 2500 CPU continuation work.'


def logical_allocation_bytes(case):
    s=case['shape'];k=case['kernel_id']
    if 'gemm' in k:
        return 2*(s['M']*s['K']*(2 if k!='aligned_gemm_fp16' else 1)+s['K']*s['N']+s['M']*s['N'])
    if k=='gelu':return 4*s['N']
    if k=='embedding':return 2*(s['V']*s['D']+s['B']*s['S']*s['D'])+8*s['B']*s['S']+(128<<20)
    if k=='rmsnorm':return 4*s['B']*s['N']*s['D']+2*s['D']
    if k=='megamoe_fp8':return 3*s['B']*s['S']*s['D']+s['E']*s['D']
    hidden=s.get('D',576)
    return 2*s['B']*s['H']*s['S']*hidden*(4 if k=='flash_attention_fwd' else 3)


class ByteBudget:
    def __init__(self,capacity):
        self.capacity=capacity;self.free=capacity;self.condition=threading.Condition()
    def acquire(self,amount):
        if amount>self.capacity:raise MemoryError('point exceeds admitted host-memory budget')
        with self.condition:
            while self.free<amount:self.condition.wait()
            self.free-=amount
    def release(self,amount):
        with self.condition:
            self.free+=amount;self.condition.notify_all()


class CpuCampaign(FullCampaign):
    def native_stage(self):raise PermissionError('CPU continuation cannot launch native GPU jobs')
    def prepare_traces(self,*args,**kwargs):raise PermissionError('CPU continuation cannot launch GPU tracing')

    def cpu_point(self,key,mode,slot):
        folder=self.root/'points'/key;folder.mkdir(parents=True,exist_ok=True)
        destination=folder/('flash-'+mode+'.json')
        if destination.exists():return load(destination)
        case=self.cases[key];build=self.admission['builds'][case['kernel_id']]
        contract=contract_for_case(BenchmarkCase(**case))
        native=load(folder/'hardware.json') if (folder/'hardware.json').exists() else {}
        expected=native.get('launches')
        if expected is None and 'gemm' not in case['kernel_id']:expected=1
        label='flash-'+mode
        started=folder/(label+'-started.json')
        if started.exists():
            value=dict(status='interrupted_not_retried',complete=False,cpu_only=True)
            lane.write(destination,value);return value
        execution=dict(binary_sha256=build['binary_sha256'],library_sha256=self.admission['tools'].get(
            '/home/cliu/wk/FlashGPU-Sim/lib/gcc-11.3.0/cuda-13030/release/libcudart.so'),
            mode=mode,program_arguments=list(contract.args),overrides=self.admission['flash_overrides'])
        key_hash=hashlib.sha256(json.dumps(execution,sort_keys=True).encode()).hexdigest()
        lane.write(folder/'cpu-input.json',dict(case_key=key,case=case,native_reference_available=bool(native.get('native_ms')),
            gpu_capture_performed=False,source_contract='same sealed CUDA binary and arguments; timing labels pending when absent'))
        lane.write(started,dict(timestamp=lane.utc(),pid=os.getpid(),cpu_only=True,key=key_hash,contract=execution))
        try:
            receipt=flash.run_test(binary=build['binary'],sim_root=self.flashroot,cuda_root=self.cuda,
                evidence_root=folder,label=label,preset='SM90_H100',mode=mode,overrides=self.admission['flash_overrides'],
                timeout=1800,scratch_root=self.root/'flash-scratch',gpu='',omp_threads=slot['omp_threads'],
                cpus=slot['cpus'],cpu_only=True,program_arguments=list(contract.args),
                environment_overrides={'GOMP_CPU_AFFINITY':' '.join(map(str,slot['cpus']))})
            text=(folder/(label+'.txt')).read_text(errors='replace')
            cycles=lane.parse_kernel_cycles(text)
            process_complete=receipt['exit_code']==0 and bool(cycles) and (
                'GPGPU-Sim: *** exit detected ***' in text or 'simulation thread exiting' in text)
            complete=process_complete and expected is not None and len(cycles)==expected
            status='completed' if complete else 'execution_complete_unbound' if process_complete and expected is None else (
                'timeout' if receipt['timed_out'] else 'gpu_context_rejected' if receipt.get('owned_gpu_contexts_rejected') else 'failed_or_incomplete')
            value=dict(status=status,complete=complete,execution_complete=process_complete,cpu_only=True,
                key=key_hash,contract=execution,receipt=str(folder/(label+'.receipt.json')),kernel_rows=cycles,
                prediction_ms=cycles[-1]['cumulative_cycles']/1_500_000 if process_complete else None,
                native_reference_available=bool(native.get('native_ms')),native_launch_binding_available=expected is not None,
                cpus=slot['cpus'],clock_mhz=1500)
            if not complete:value['reason']='\n'.join(text.splitlines()[-10:])[-2000:]
        except Exception as exc:value=dict(status='adapter_error',complete=False,cpu_only=True,key=key_hash,reason=str(exc))
        lane.write(destination,value)
        return value

    def pending_cpu(self):
        return [(key,mode) for key in self.keys for mode in ('ptx','sass-timing')
                if not (self.root/'points'/key/('flash-'+mode+'.json')).exists()]

    def gcoM_inventory(self):
        reused=0;pending=[]
        for key in self.keys:
            folder=self.root/'points'/key;file=folder/'gcom.json'
            if file.exists() and load(file).get('status')!='trace_not_qualified':reused+=1
            else:pending.append(dict(case_key=key,reason='GPU trace capture or missing static function-address metadata required'))
        lane.write(self.phase_dir/'cpu-gcom-input-inventory.json',dict(timestamp=lane.utc(),primary_total=len(self.keys),
            verified_existing_outcomes=reused,pending_gpu_inputs=pending,
            rule='Never fabricate function addresses: GCoM uses them for global instruction addresses and I-cache pricing.'))
        return reused,len(pending)

    def execute_cpu(self):
        with (self.root/'controller.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            flash._gpu_context_pids()  # Read-only monitoring must be available before launch.
            reused_gcom,pending_gcom=self.gcoM_inventory()
            pending=self.pending_cpu()
            self.publish('cpu_continuation',cpu_jobs=len(pending),gpu_approval='pending',gpu_jobs_launched=False,
                gcoM_reused=reused_gcom,gcoM_pending_gpu_inputs=pending_gcom)
            allocation=resources.cpu_admission(self.phase_dir/'cpu-only-resources',point_count=48,threads=4)
            if not allocation['slots']:raise RuntimeError('no idle CPU allocation admitted')
            budget=ByteBudget(max(0,allocation['memory_available_bytes']-64*1024**3))
            cpu_pool=resources.ReplayCpuPool(allocation['slots'],fallback_threads=4)
            remaining=len(pending);completed=0
            lane.write(self.phase_dir/'cpu-execution-admission.json',dict(timestamp=lane.utc(),user_prompt=PROMPT,
                primary_total=len(self.keys),new_cpu_jobs=len(pending),reused_flash_jobs=2*len(self.keys)-len(pending),
                worker_limit=min(48,len(allocation['slots'])),threads_per_point=4,host_memory_budget_bytes=budget.capacity,
                cuda_visible_devices='',gpu_context_guard=True,gpu_jobs_authorized=False,
                model_parameters_changed=False,gcoM_reused=reused_gcom,gcoM_pending_gpu_inputs=pending_gcom))
            def worker(key,mode):
                nonlocal remaining,completed
                memory=(1<<30)+4*logical_allocation_bytes(self.cases[key])
                budget.acquire(memory)
                slot=cpu_pool.acquire(4)
                try:
                    value=self.cpu_point(key,mode,slot)
                    with self.lock:
                        remaining-=1;completed+=1
                        self.status('cpu_continuation',primary_total=len(self.keys),cpu_jobs_remaining=remaining,
                            cpu_jobs_completed=completed,last_case=key,last_mode=mode,last_outcome=value['status'],
                            gpu_approval='pending',gpu_jobs_launched=False,gcoM_pending_gpu_inputs=pending_gcom)
                        if completed%64==0:self.report()
                    print(lane.utc(),mode,key,value['status'],flush=True)
                finally:cpu_pool.release(slot);budget.release(memory)
            pending.sort(key=lambda task:logical_allocation_bytes(self.cases[task[0]]))
            with ThreadPoolExecutor(max_workers=min(48,len(allocation['slots']))) as executor:
                jobs=[executor.submit(worker,key,mode) for key,mode in pending]
                for job in as_completed(jobs):job.result()
            self.publish('cpu_continuation_finished_waiting_gpu_approval',cpu_jobs_completed=completed,
                gpu_jobs_launched=False,gcoM_pending_gpu_inputs=pending_gcom,pending_primary_native=len(self.pending_native()))
            self.report(figures=True)
            from amora.benchmarking.portable_v2.combined_report import generate
            generate(self.project)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    CpuCampaign(parser.parse_args().root).execute_cpu()


if __name__=='__main__':main()
