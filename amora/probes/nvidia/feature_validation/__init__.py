"""Independent NVIDIA feature semantics tests, separate from parameter fitting."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROBES = {
    "dsm": {
        "source": "hopper_dsm_probe.cu",
        "arch": "sm_90a",
        "contract": "Cluster remote store, atomic add, explicit fence, and remote mbarrier arrival with local owner wait.",
    },
    "tcgen05": {
        "source": "tcgen05_smoke.cu",
        "arch": "sm_100a",
        "contract": "Four-warp TMEM round trip and uniform dense f16 MMA, commit-to-mbarrier, and accumulator load. Uniform inputs do not qualify descriptor mapping.",
    },
    "tmem-capacity": {
        "source": "tmem_capacity_probe.cc",
        "arch": None,
        "contract": "Observe production TMEM capacity, address validation and reuse; exit zero reports observations rather than ISA conformance.",
    },
}

def list_probes():
    import hashlib
    return [
        dict(probe_id=name, **entry,
             source_sha256=hashlib.sha256((ROOT / entry["source"]).read_bytes()).hexdigest())
        for name, entry in PROBES.items()
    ]

