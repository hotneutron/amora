"""Full-manifest scheduling and GPU approval checks; no GPU work."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from amora.benchmarking.canonical_ppp.full_campaign import (
    FullCampaign, primary_keys, choose_wave, require_gpu_approval, approved_devices,
)
from amora.backends.gcom_cuda import portable_v2 as lane


def test_full_manifest_has_no_128_case_or_work_score_cutoff():
    keys=[f'case-{i}' for i in range(2500)]
    manifest=dict(primary_keys=keys,cases=[dict(case_key=k,shape=dict(N=55_000_000)) for k in keys])
    assert primary_keys(manifest)==keys


def test_primary_case_identity_duplicates_are_rejected():
    with pytest.raises(ValueError,match='duplicate'):primary_keys(dict(primary_keys=['x','x']))


def test_native_resume_preserves_all_129_measured_primary_cases(tmp_path):
    campaign=object.__new__(FullCampaign);campaign.root=tmp_path
    campaign.keys=[f'case-{i}' for i in range(2500)]
    for key in campaign.keys[:129]:
        folder=tmp_path/'points'/key;folder.mkdir(parents=True)
        (folder/'hardware.json').write_text('{"hardware_status":"completed"}')
    assert campaign.pending_native()==campaign.keys[129:]
    assert len(campaign.pending_native())==2371


def test_existing_native_result_needs_no_new_gpu_job(tmp_path):
    campaign=object.__new__(FullCampaign);campaign.root=tmp_path
    folder=tmp_path/'points/x';folder.mkdir(parents=True)
    original=dict(hardware_status='completed',native_ms=.04)
    (folder/'hardware.json').write_text(json.dumps(original))
    assert campaign.native_case('x',None,[])==original


def test_wave_limit_is_inflight_bytes_not_total_case_count():
    keys=[str(i) for i in range(100)]
    selected=choose_wave(keys,{k:1<<30 for k in keys},12<<30,reserve_bytes=2<<30)
    assert len(selected)==5
    assert len(keys)==100


def test_large_estimated_case_is_not_permanently_rejected():
    assert choose_wave(['large'],{'large':500<<30},30<<30,reserve_bytes=16<<30)==['large']
    assert choose_wave(['large'],{'large':500<<30},20<<30,reserve_bytes=16<<30)==[]


def test_gpu_jobs_require_explicit_approval(tmp_path):
    with pytest.raises(PermissionError,match='explicit user approval'):require_gpu_approval(tmp_path)
    folder=tmp_path/'full-study';folder.mkdir()
    (folder/'gpu-launch-approval.json').write_text('{"state":"pending"}')
    with pytest.raises(PermissionError):require_gpu_approval(tmp_path)


def test_approved_gpu_scope_does_not_expand_to_other_devices(tmp_path):
    folder=tmp_path/'full-study';folder.mkdir()
    record=dict(state='approved',authority='explicit_user_approval',allowed_gpu_uuids=['GPU-7'])
    (folder/'gpu-launch-approval.json').write_text(json.dumps(record))
    devices=[dict(uuid='GPU-0'),dict(uuid='GPU-7')]
    assert approved_devices(tmp_path,devices)==[devices[1]]
    require_gpu_approval(tmp_path,'GPU-7')
    with pytest.raises(PermissionError):require_gpu_approval(tmp_path,'GPU-0')


def test_paused_execute_never_enters_native_stage(tmp_path):
    campaign=object.__new__(FullCampaign);campaign.root=tmp_path
    observed=[]
    campaign.publish=lambda state,**kwargs:observed.append(state)
    campaign.pending_native=lambda:['remaining']
    campaign.native_stage=lambda:pytest.fail('GPU stage entered while approval pending')
    campaign.execute_full()
    assert observed==['waiting_for_user_gpu_approval']


def test_ram_guard_stops_only_the_owned_capture_child(tmp_path,monkeypatch):
    monkeypatch.setattr(lane,'gpu_snapshot',lambda _:pytest.fail('GPU query should not be reached'))
    lease=SimpleNamespace(device=dict(uuid='fake',index=0))
    result=lane.capture_hardware([sys.executable,'-c','import time; time.sleep(2)'],
        lease=lease,root=tmp_path,label='memory-guard',env=dict(__import__('os').environ),
        memory_reserve_bytes=1<<63,disk_reserve_bytes=0,timeout=5)
    assert result['memory_exhausted'] and result['exit_code']!=0
    assert result['foreign_processes']==[]
