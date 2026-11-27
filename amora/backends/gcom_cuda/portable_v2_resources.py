"""CPU admission and verified trace storage for portable GCoM campaigns."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import threading

from .portable_v2 import sha, utc, write


def parse_cpu_list(value):
    result=set()
    for item in value.strip().split(","):
        if not item: continue
        if "-" in item:
            lo,hi=map(int,item.split("-"));result.update(range(lo,hi+1))
        else: result.add(int(item))
    return result


def cpu_counters():
    result={}
    for line in Path("/proc/stat").read_text().splitlines():
        parts=line.split()
        if not parts or not parts[0].startswith("cpu") or not parts[0][3:].isdigit(): continue
        values=list(map(int,parts[1:9]))
        result[int(parts[0][3:])]=(sum(values),values[3]+values[4])
    return result


def cpu_admission(root, *, gpu_count=0, devices=(), point_count=24, threads=8):
    Path(root).mkdir(parents=True,exist_ok=True)
    allowed=set(os.sched_getaffinity(0))
    first=cpu_counters();time.sleep(1);second=cpu_counters()
    usage={cpu:1-(second[cpu][1]-first[cpu][1])/max(1,second[cpu][0]-first[cpu][0]) for cpu in allowed}
    groups=defaultdict(list)
    for cpu in sorted(allowed):
        folder=Path(f"/sys/devices/system/cpu/cpu{cpu}")
        core=tuple(sorted(parse_cpu_list((folder/"topology/thread_siblings_list").read_text()) & allowed))
        if groups[core]: continue
        node=next(iter(folder.glob("node*")),None)
        groups[core]=[int(node.name[4:]) if node else 0]
    # Busy foreign cores and their SMT siblings are excluded from replay.
    cores=sorted(groups,key=lambda c:(max(usage[x] for x in c),groups[c][0],c))
    reservations=[]
    for device in devices:
        preferred=set(device["cpu_affinity"])
        core=next((c for c in cores if c not in reservations and set(c)<=preferred and max(usage[x] for x in c)<.20),None)
        if core is None: raise ValueError("no NUMA-local acquisition core")
        reservations.append(core)
    if not devices: reservations=cores[:gpu_count]
    eligible=[c for c in cores if c not in reservations and max(usage[x] for x in c)<.20]
    by_node=defaultdict(list)
    for core in eligible: by_node[groups[core][0]].extend(core)
    quota=len(allowed)
    relative=Path(next(line.split(":",2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::")).lstrip("/"))
    cgroup=Path("/sys/fs/cgroup")/relative
    for folder in [cgroup,*cgroup.parents]:
        limit=folder/"cpu.max"
        if limit.exists():
            q,period=limit.read_text().split()
            if q!="max": quota=min(quota,max(1,int(q)//int(period)))
        if folder==Path("/sys/fs/cgroup"): break
    budget=max(0,quota-sum(map(len,reservations)))
    slots=[]
    for node,cpus in sorted(by_node.items()):
        while cpus and budget and len(slots)<point_count:
            size=min(threads,len(cpus),budget)
            slots.append(dict(cpus=cpus[:size],numa_node=node,omp_threads=size))
            cpus=cpus[size:];budget-=size
    mem={line.split()[0].rstrip(":"):int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()}
    value=dict(timestamp=utc(),allowed_cpus=sorted(allowed),quota_cpus=quota,cpu_usage=usage,
        acquisition_cpus=[list(c) for c in reservations],slots=slots,
        memory_available_bytes=mem["MemAvailable"],disk_free_bytes=shutil.disk_usage(root).free,
        policy="exclude cores with >=20 percent observed use and SMT siblings; disjoint NUMA-local replay slots")
    write(Path(root)/"cpu-admission.json",value)
    return value


class ReplayCpuPool:
    """Recycle disjoint NUMA-local CPU sets between distinct data points."""
    def __init__(self,slots, *, fallback_threads=8):
        if fallback_threads<1:raise ValueError("replay thread count must be positive")
        self.fallback_threads=fallback_threads
        self.order=defaultdict(list)
        for slot in slots:self.order[slot["numa_node"]].extend(slot["cpus"])
        all_cpus=[cpu for values in self.order.values() for cpu in values]
        if len(all_cpus)!=len(set(all_cpus)):raise ValueError("overlapping replay CPU pool")
        self.free={node:set(cpus) for node,cpus in self.order.items()}
        self.condition=threading.Condition()

    def acquire(self,preferred):
        with self.condition:
            while not any(self.free.values()):self.condition.wait()
            node=max(self.free,key=lambda node:len(self.free[node]))
            size=preferred if len(self.free[node])>=preferred else min(self.fallback_threads,len(self.free[node]))
            cpus=[cpu for cpu in self.order[node] if cpu in self.free[node]][:size]
            self.free[node].difference_update(cpus)
            return dict(cpus=cpus,numa_node=node,omp_threads=len(cpus))

    def release(self,slot):
        with self.condition:
            if self.free[slot["numa_node"]] & set(slot["cpus"]):raise ValueError("CPU allocation already released")
            self.free[slot["numa_node"]].update(slot["cpus"])
            self.condition.notify_all()


def archive_trace(trace_dir, info, archive):
    """Remove raw duplicates only after a complete lossless archive is verified."""
    trace_dir,archive=Path(trace_dir),Path(archive)
    archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists(): raise FileExistsError(archive)
    temporary=archive.with_suffix(archive.suffix+".tmp")
    with tarfile.open(temporary,"w:gz",compresslevel=3) as out:
        for relative in sorted(info["files"]):
            path=trace_dir/relative
            if sha(path)!=info["files"][relative]: raise ValueError("trace changed before archive")
            out.add(path,arcname=relative,recursive=False)
    verify_archive(temporary,info)
    temporary.replace(archive)
    receipt=dict(archive=str(archive),archive_sha256=sha(archive),bytes=archive.stat().st_size,
                 trace_sha256=info["trace_sha256"],verified_utc=utc())
    write(archive.with_suffix(archive.suffix+".json"),receipt)
    shutil.rmtree(trace_dir)
    return receipt


def verify_archive(archive,info):
    with tarfile.open(archive,"r:gz") as source:
        members=source.getmembers()
        if len(members)!=len(info["files"]) or {m.name for m in members}!=set(info["files"]):
            raise ValueError("trace archive file accounting mismatch")
        for member in members:
            if not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise ValueError("unsafe trace archive member")
            digest=hashlib.sha256()
            stream=source.extractfile(member)
            for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
            if digest.hexdigest()!=info["files"][member.name]:raise ValueError("trace archive changed bytes")


def restore_trace(archive,info,destination):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    verify_archive(archive,info)
    with tarfile.open(archive,"r:gz") as source:
        for member in source.getmembers():
            path=destination/member.name;path.parent.mkdir(parents=True,exist_ok=True)
            with source.extractfile(member) as stream,path.open("xb") as out:shutil.copyfileobj(stream,out)
    return destination
