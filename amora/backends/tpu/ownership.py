"""Read-only Linux device-holder evidence before libtpu initialization."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

_SCAN = r'''
import errno, glob, json, os, pathlib, sys
holders=[]; denied=0
ignored=set(map(int,sys.argv[1:]))
for path in glob.glob('/proc/[0-9]*/fd/*'):
    try:
        target=os.readlink(path)
        if target.startswith(('/dev/vfio/', '/dev/accel', '/dev/apex')):
            pid=int(path.split('/')[2])
            if pid in ignored: continue
            holders.append({'pid':pid,'name':pathlib.Path(f'/proc/{pid}/comm').read_text().strip(),
                            'device':target})
    except OSError as e:
        if e.errno in (errno.EPERM,errno.EACCES): denied+=1
# glob does not report inaccessible directories; inspect directory traversal too.
for directory in glob.glob('/proc/[0-9]*/fd'):
    try: os.listdir(directory)
    except PermissionError: denied+=1
    except FileNotFoundError: pass
print(json.dumps({'holders':holders,'inaccessible':denied,'scope':'all visible accelerator nodes'}))
'''


def snapshot(exclude_pids=()):
    if not Path("/proc").is_dir():
        return {"state": "unknown", "reason": "ownership_scan_unavailable", "holders": []}
    command = ["/usr/bin/python3", "-c", _SCAN, *(str(int(pid)) for pid in exclude_pids)]
    try:
        privileged = subprocess.run(["sudo", "-n", *command], capture_output=True,
                                    text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        privileged = None
    result = privileged if privileged and privileged.returncode == 0 else subprocess.run(
        command, capture_output=True, text=True, timeout=15)
    if result.returncode:
        return {"state": "unknown", "reason": "ownership_scan_failed", "holders": [],
                "stderr": result.stderr[-2000:]}
    record = json.loads(result.stdout)
    record["privileged"] = result is privileged
    record["state"] = "busy" if record["holders"] else (
        "unknown" if record["inaccessible"] else "exclusive")
    return record
