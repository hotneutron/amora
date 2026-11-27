"""Fetch exact NVIDIA PTX sections with version and HTML digest provenance."""
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

ids = [
    "tensor-memory", "tensor-memory-allocation",
    "tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit",
    "tcgen05-tensor-memory-ld-st-access-restrictions",
    "tcgen05-mma-instructions-mma",
    "warp-level-matrix-instructions-ldmatrix",
    "warp-level-matrix-instructions-stmatrix",
    "asynchronous-warpgroup-level-matrix-instructions-wgmma-mma",
    "parallel-synchronization-and-communication-instructions-mbarrier-phase-completion",
    "parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait",
    "data-movement-and-conversion-instructions-cp-async-bulk-tensor",
]

def extract_spec(evidence_root, *, cache=None, url="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html"):
    from bs4 import BeautifulSoup
    data = Path(cache).read_bytes() if cache else urllib.request.urlopen(url, timeout=60).read()
    soup = BeautifulSoup(data, "html.parser")
    sections = {}
    for sid in ids:
        section = soup.find(id=sid)
        if section is None:
            raise ValueError("Missing NVIDIA section " + sid)
        sections[sid] = dict(url=url + "#" + sid, text=section.get_text(" ", strip=True))
    record = dict(retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  url=url, title=soup.title.get_text(), bytes=len(data),
                  html_sha256=hashlib.sha256(data).hexdigest(), sections=sections)
    root = Path(evidence_root)
    root.mkdir(parents=True, exist_ok=True)
    output = root / "nvidia-ptx-excerpts.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(record, indent=2) + "\n")
    return {k:v for k,v in record.items() if k != "sections"}
