"""Kernel-notified idle controller handoff and automatic report finalization."""
import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys

from amora.backends.gcom_cuda.portable_v2 import write, utc


def load(path):return json.loads(Path(path).read_text())


def descendants(pid):
    found=set()
    def scan(parent):
        for file in Path(f'/proc/{parent}/task').glob('*/children'):
            try:children=map(int,file.read_text().split())
            except FileNotFoundError:continue
            for child in children:
                if child not in found:found.add(child);scan(child)
    scan(pid)
    return found


def wait_pid(pid):
    try:fd=os.pidfd_open(pid)
    except ProcessLookupError:return
    try:select.select([fd],[],[])
    finally:os.close(fd)


def finalize(root, pid):
    wait_pid(pid)
    recovery=root/'cublas-recovery-verified-launch.json'
    if recovery.exists():wait_pid(load(recovery)['pid'])
    from .campaign import Campaign
    from .report import generate
    from amora.benchmarking.portable_v2.combined_report import generate as combined
    campaign=Campaign(root);campaign.validate()
    pending=[load(p) for p in (root/'points').glob('*/hardware.json')
             if load(p).get('trace_status')=='qualified' and not (p.parent/'gcom.json').exists()]
    # Any newly repaired cubin input gets its first replay after controller
    # exit. Existing successful and failed immutable keys are reused.
    if pending:campaign.replay(pending,'final-repaired-inputs')
    ledger=generate(root,figures=True)
    comparison=combined(root.parent.parent)
    sources=0
    from amora.backends.gcom_cuda.portable_v2 import sha
    for source in comparison['sources'].values():
        if sha(source['path'])!=source['sha256']:raise ValueError('final source hash drift')
        sources+=1
    import re
    for report in [Path(ledger['report']),Path(load(root.parent/'flashgpu-gcom-comparison/report-target.json')['report'])]:
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',report.read_text()):
            if not (report.parent/link).exists():raise ValueError('missing report link '+link)
    target=load(root/'campaign-target.json');tracker=Path(target['tracker']);ts=datetime.now().astimezone().isoformat(timespec='seconds')
    text=tracker.read_text().replace('| Nine-family pilot | Running under dedicated GPU admission |',
        '| Nine-family pilot | Retained native and model outcomes; see final ledger |')
    text=text.replace('| Reviewed expansion | Pending pilot support/cost review |',
        '| Reviewed expansion | Bounded results retained; full-manifest deferred cases remain explicit |')
    text=text.replace('| Canonical and combined reports | Pending |','| Canonical and combined reports | Updated with verified local links, figures and source hashes |')
    final_rows={
        'Nine-family pilot':'Completed; all nine native family pilots retained with model support outcomes',
        'Reviewed expansion':f"Completed bounded stage: {ledger['native_completed']} native cases; remaining manifest shapes are deferred",
        'Canonical and combined reports':'Finalized with source hashes, figures and resolved links',
    }
    text='\n'.join(next((f'| {name} | {value} |' for name,value in final_rows.items()
                          if line.startswith('| '+name+' |')),line) for line in text.splitlines())+'\n'
    text+=f"\n- {ts}: Finalized retained canonical evidence and combined report. Native completions {ledger['native_completed']}; qualified PTX/SASS/GCoM pairs "+'/'.join(str(ledger['overall'][k]['eligible_cases']) for k in ('flash_ptx','flash_sass','gcom'))+f". Verified {sources} combined evidence hashes and report links. Full 2,500-case coverage is not claimed.\n"
    tracker.write_text(text)
    write(root/'final-report-validation.json',dict(timestamp=utc(),source_hashes_verified=sources,
        native_completed=ledger['native_completed'],metrics=ledger['overall'],report=ledger['report'],links_verified=True,
        full_manifest_completed=False,simulator_repeated=False))
    print('Canonical reports finalized',ledger['report'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--pid',type=int,required=True);args=parser.parse_args();root=args.root.resolve();pid=args.pid
    stat=Path(f'/proc/{pid}/stat')
    if not stat.exists():finalize(root,pid);return
    ticks=stat.read_text().rpartition(')')[2].split()[19]
    if b'amora.benchmarking.canonical_ppp.campaign' not in Path(f'/proc/{pid}/cmdline').read_bytes():
        raise ValueError('handoff owner is not the canonical controller')
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.inotify_init1(0)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    if libc.inotify_add_watch(fd,str(root).encode(),0x80)<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
    oldfd=os.pidfd_open(pid)
    try:
        while True:
            ready=select.select([fd,oldfd],[],[])[0]
            if oldfd in ready:finalize(root,pid);return
            events=os.read(fd,65536);names=[];offset=0
            while offset+16<=len(events):
                _,_,_,length=struct.unpack_from('iIII',events,offset)
                names.append(events[offset+16:offset+16+length].split(b'\0',1)[0])
                offset+=16+length
            if b'status.json' not in names:continue
            status=load(root/'status.json')
            if status.get('pid')!=pid or status.get('state')!='expansion':continue
            if not stat.exists() or stat.read_text().rpartition(')')[2].split()[19]!=ticks:
                raise ValueError('controller process identity changed')
            if descendants(pid):continue
            os.kill(pid,signal.SIGSTOP)
            stopped='\nState:\tT' in Path(f'/proc/{pid}/status').read_text()
            leased=any(load(p).get('state')=='leased' and load(p).get('pid')==pid
                       for p in (root/'resource-leases').glob('*.json'))
            unfinished=any(not (p.parent/'result.json').exists() for p in (root/'replay-cache').glob('*/started.json'))
            unfinished |= any(not p.with_name(p.name.replace('-started.json','.json')).exists()
                              for p in (root/'points').glob('*/flash-*-started.json'))
            if not stopped or descendants(pid) or leased or unfinished:
                os.kill(pid,signal.SIGCONT);continue
            # The expansion status is written before a fresh 30-second GPU
            # observation. No native capture or model exists at this boundary.
            os.kill(pid,signal.SIGKILL)
            command=[sys.executable,'-u','-m','amora.benchmarking.canonical_ppp.campaign','execute','--root',str(root)]
            with (root/'controller-wide-waves.log').open('x') as output:
                process=subprocess.Popen(command,cwd='/home/cliu/wk/amora',stdout=output,stderr=subprocess.STDOUT,
                    env=dict(os.environ,PYTHONPATH='/home/cliu/wk/amora',OPENBLAS_NUM_THREADS='1'),start_new_session=True)
            write(root/'wide-wave-handoff.json',dict(timestamp=utc(),old_pid=pid,old_start_ticks=ticks,new_pid=process.pid,
                observed_descendants=[],leased_gpus=[],scope='controller-only idle boundary; 64 staged cases, up to 48 distinct 4-thread replay points',
                models_interrupted=False,simulators_repeated=False))
            print('Idle controller handoff',pid,'to',process.pid,flush=True)
            finalize(root,process.pid);return
    finally:os.close(fd);os.close(oldfd)


if __name__=='__main__':main()
