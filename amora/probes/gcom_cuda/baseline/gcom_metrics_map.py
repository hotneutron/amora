"""GCoM internal-stats -> AMORA logical metric mapping (versioned).

GCoM emits a much richer internal stat set than NCU. This module translates GCoM
stat keys into AMORA's logical metric names (see
``amora/backends/nvidia/metrics.py::MetricResolver``) so the simulator's counters
can be compared against the real NCU counters recorded by the nvidia backend.

Each entry records the required GCoM stat key(s), a derivation, the nearest NCU
metric, a fidelity tag, and an architecture scope. Reads GCoM's existing printed
stats only — no simulator changes. Grounded in the simulator source stat names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from amora.backends.gcom_cuda.config import DRAM_ATOM_BYTES

MAPPING_VERSION = "2026-07-gcom-cuda-v2"

# Fidelity tags (see plan Accuracy Model).
DIRECT = "direct"
PROPORTIONAL = "proportional"
PROXY = "proxy"


@dataclass(frozen=True)
class GcomMetric:
    logical: str
    gcom_keys: tuple[str, ...]
    ncu_metric: str | None
    fidelity: str
    architecture_scope: str
    derivation: Callable[[dict[str, float]], float | None]
    note: str = ""

    def derive(self, stats: dict[str, float]) -> float | None:
        # Only derive when all required keys are present.
        if not all(k in stats for k in self.gcom_keys):
            return None
        try:
            return self.derivation(stats)
        except (KeyError, ZeroDivisionError, ValueError):
            return None


def _identity(key: str) -> Callable[[dict[str, float]], float | None]:
    return lambda s: s.get(key)


def _times_atom(key: str) -> Callable[[dict[str, float]], float | None]:
    return lambda s: s[key] * DRAM_ATOM_BYTES


def _one_minus(key: str) -> Callable[[dict[str, float]], float | None]:
    return lambda s: 1.0 - s[key]


def _diff(a: str, b: str) -> Callable[[dict[str, float]], float | None]:
    return lambda s: s[a] - s[b]


def _stall_pct(reason: str) -> GcomMetric:
    key = f"ncu_stall_{reason}_pct"
    return GcomMetric(
        f"stall_{reason}_pct",
        (key,),
        f"smsp__average_warps_issue_stalled_{reason}_per_issue_active.ratio",
        PROPORTIONAL,
        "nvidia_hopper",
        _identity(key),
        "GCoM NCU-aligned stall reason percentage",
    )


GCOM_TO_LOGICAL: tuple[GcomMetric, ...] = (
    GcomMetric(
        "sm_active_cycles", ("gpu_sim_cycle",), "sm__cycles_active.avg",
        DIRECT, "generic_cuda", _identity("gpu_sim_cycle"),
        "sim core cycles",
    ),
    GcomMetric(
        "inst_executed", ("gpu_tot_sim_insn",), "smsp__inst_executed.sum",
        DIRECT, "generic_cuda", _identity("gpu_tot_sim_insn"),
        "thread-instruction count",
    ),
    GcomMetric(
        "global_load_requests", ("total_dl1_accesses",),
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
        PROPORTIONAL, "nvidia_generic", _identity("total_dl1_accesses"),
        "L1 access count ~ requests",
    ),
    GcomMetric(
        "dram_bytes_read", ("gpgpu_n_dram_reads",), "dram__bytes_read.sum",
        PROPORTIONAL, "nvidia_generic", _times_atom("gpgpu_n_dram_reads"),
        "reads x DRAM atom bytes",
    ),
    GcomMetric(
        "dram_bytes_write", ("gpgpu_n_dram_writes",), "dram__bytes_write.sum",
        PROPORTIONAL, "nvidia_generic", _times_atom("gpgpu_n_dram_writes"),
        "writes x DRAM atom bytes",
    ),
    GcomMetric(
        "dram_throughput", ("bwutil",),
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        PROPORTIONAL, "nvidia_generic", _identity("bwutil"),
        "BW utilization ~ pct-of-peak",
    ),
    GcomMetric(
        "l2_sector_hits", ("L2_total_cache_accesses", "L2_total_cache_misses"),
        "lts__t_sectors_lookup_hit.sum",
        PROPORTIONAL, "nvidia_generic",
        _diff("L2_total_cache_accesses", "L2_total_cache_misses"),
        "L2 accesses - misses",
    ),
    GcomMetric(
        "l1_hit_rate", ("total_dl1_miss_rate",), "l1tex__t_sector_hit_rate.pct",
        PROPORTIONAL, "nvidia_generic", _one_minus("total_dl1_miss_rate"),
        "1 - L1 miss rate",
    ),
    GcomMetric(
        "l2_hit_rate", ("L2_total_cache_miss_rate",), None,
        PROPORTIONAL, "nvidia_generic", _one_minus("L2_total_cache_miss_rate"),
        "1 - L2 miss rate",
    ),
    GcomMetric(
        "occupancy", ("gpu_occupancy",), "sm__warps_active.avg.pct_of_peak_sustained_active",
        PROPORTIONAL, "nvidia_generic", _identity("gpu_occupancy"),
        "achieved occupancy",
    ),
    GcomMetric(
        "interconnect_latency", ("avg_icnt2mem_latency",), None,
        PROXY, "gcom_simulator_only", _identity("avg_icnt2mem_latency"),
        "sim-only interconnect latency",
    ),
    _stall_pct("selected"),
    _stall_pct("not_selected"),
    _stall_pct("dispatch_stall"),
    _stall_pct("warpgroup_arrive"),
    _stall_pct("long_scoreboard"),
    _stall_pct("short_scoreboard"),
    _stall_pct("barrier"),
    _stall_pct("wait"),
    _stall_pct("mio_throttle"),
    _stall_pct("math_pipe_throttle"),
    _stall_pct("mma"),
    _stall_pct("no_instructions"),
    _stall_pct("imc_miss"),
    _stall_pct("sleeping"),
    _stall_pct("branch_resolving"),
    _stall_pct("membar"),
    _stall_pct("drain"),
    _stall_pct("lg_throttle"),
    _stall_pct("tex_throttle"),
    _stall_pct("misc"),
)


# Sanity: no duplicate logical names.
assert len({m.logical for m in GCOM_TO_LOGICAL}) == len(GCOM_TO_LOGICAL), (
    "duplicate logical metric in GCOM_TO_LOGICAL"
)
