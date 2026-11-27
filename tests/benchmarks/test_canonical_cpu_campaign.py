"""CPU continuation must not require or create GPU measurements."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import pytest

from amora.benchmarking.canonical_ppp.cpu_campaign import CpuCampaign, ByteBudget, logical_allocation_bytes
from amora.backends.flashgpu_sim import runner as flash
from benchmark_generators.ppp_canonical.generator import PPP_CANONICAL


def test_cpu_only_mode_rejects_native_gpu_execution(tmp_path):
    with pytest.raises(ValueError,match='native GPU execution'):
        flash.run_test(binary=sys.executable,sim_root=tmp_path,cuda_root=tmp_path,
                       evidence_root=tmp_path,label='native',mode='native',cpu_only=True)


def test_context_guard_does_not_stop_foreign_gpu_processes(tmp_path,monkeypatch):
    monkeypatch.setattr(flash,'_gpu_context_pids',lambda:[os.getpid()])
    result=flash.capture([sys.executable,'-c','print("cpu only")'],label='cpu',
        evidence_root=tmp_path,cwd=tmp_path,timeout=5,forbid_gpu_contexts=True)
    assert result['exit_code']==0 and result['owned_gpu_contexts_rejected']==[]


def test_cpu_context_guard_stops_only_its_owned_group(tmp_path,monkeypatch):
    original=flash.subprocess.Popen;owned=[]
    def launch(*args,**kwargs):
        p=original(*args,**kwargs);owned.append(p.pid);return p
    monkeypatch.setattr(flash.subprocess,'Popen',launch)
    monkeypatch.setattr(flash,'_gpu_context_pids',lambda:owned)
    result=flash.capture([sys.executable,'-c','import time;time.sleep(2)'],label='reject',
        evidence_root=tmp_path,cwd=tmp_path,timeout=5,forbid_gpu_contexts=True)
    assert result['exit_code']==125 and result['owned_gpu_contexts_rejected']==owned


def cpu_fixture(tmp_path):
    cases,_=PPP_CANONICAL.materialize(target={'vendor':'nvidia','sku':'h100-80g'},case_count=9,seed=20260717)
    case=next(c for c in cases if c.kernel_id=='gelu')
    campaign=object.__new__(CpuCampaign);campaign.root=tmp_path;campaign.cases={case.case_key:asdict(case)}
    campaign.admission={'builds':{'gelu':{'binary':sys.executable,'binary_sha256':'source'}},
                        'tools':{},'flash_overrides':[]}
    campaign.flashroot=tmp_path;campaign.cuda=tmp_path
    return campaign,case.case_key


def test_cpu_prediction_can_run_with_native_measurements_absent(tmp_path,monkeypatch):
    campaign,key=cpu_fixture(tmp_path);called=[]
    def simulate(**kwargs):
        called.append(kwargs)
        path=Path(kwargs['evidence_root'])/(kwargs['label']+'.txt')
        path.write_text('gpu_sim_cycle = 100\ngpu_tot_sim_cycle = 100\nGPGPU-Sim: *** exit detected ***\n')
        return dict(exit_code=0,timed_out=False)
    monkeypatch.setattr(flash,'run_test',simulate)
    output=campaign.cpu_point(key,'ptx',dict(cpus=[0],omp_threads=1))
    assert output['complete'] and not output['native_reference_available']
    assert called[0]['cpu_only'] and called[0]['gpu']==''
    assert not (tmp_path/'points'/key/'hardware.json').exists()


def test_existing_cpu_results_are_reused_without_new_execution(tmp_path,monkeypatch):
    campaign,key=cpu_fixture(tmp_path);folder=tmp_path/'points'/key;folder.mkdir(parents=True)
    saved=dict(status='failed_or_incomplete',complete=False)
    (folder/'flash-ptx.json').write_text(json.dumps(saved))
    monkeypatch.setattr(flash,'run_test',lambda **kwargs:pytest.fail('identical execution repeated'))
    assert campaign.cpu_point(key,'ptx',dict(cpus=[0],omp_threads=1))==saved


def test_memory_budget_is_bytes_and_releases_capacity():
    budget=ByteBudget(100);budget.acquire(80);assert budget.free==20
    budget.release(80);assert budget.free==100
    with pytest.raises(MemoryError):budget.acquire(101)
    assert logical_allocation_bytes(dict(kernel_id='gelu',shape=dict(N=100)))==400
