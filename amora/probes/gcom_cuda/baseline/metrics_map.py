"""gcom_cuda per-probe simulation policy (single source of truth = nvidia registry).

This table declares only simulator-specific *policy* per probe: how to derive a
simulated value from GCoM stats, which hardware denominator field it needs, and
its fidelity / category. The probe **inventory** is derived from the NVIDIA
baseline registry (never re-listed); a load-time assertion enforces this table
covers exactly that inventory so drift fails fast.

Canonical probe metadata (concept, unit, denominator values) is NOT duplicated
here — it is read from the real NVIDIA ``ProbeResult`` at derive/compare time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from amora.probes.nvidia.baseline import PROBES as NVIDIA_PROBES

# Categories (plan §Initial gcom_h100 Policy).
COMPARABLE = "comparable"        # direct cycle/stat derivation with known denominator
APPROXIMATE = "approximate"      # sweep / differential / proxy; may need multi-launch
UNAVAILABLE = "unavailable"      # see `state` for the specific unavailable reason

# Unavailable states (plan §Unsupported, Unknown, Missing, And Proxy States).
UNSUPPORTED = "unsupported"          # simulator fundamentally does not model it
MISSING_STAT = "missing_stat"        # may model it, but stat not emitted
UNMAPPED = "unmapped"                # stat emitted but not mapped yet
PROXY_ONLY = "proxy_only"            # not equivalent, diagnostic only
NOT_APPLICABLE = "not_applicable"    # metric does not apply

# Derivation kinds (how the generic runner turns sim stats into a value).
PER_OP = "per_op"            # gpu_sim_cycle / hw_denominator
THROUGHPUT = "throughput"    # hw_denominator / gpu_sim_cycle
BANDWIDTH = "bandwidth"      # sim_bytes / (gpu_sim_cycle / core_clock)
SWEEP = "sweep"              # reduce per-variant points (knee/plateau/ratio)
DIFFERENTIAL = "differential"  # difference of two sub-kernel runs
NONE = "none"                # no value derived (unavailable rows)


@dataclass(frozen=True)
class ProbePolicy:
    category: str
    derivation: str
    # Name of the raw-value key in the HW ProbeResult used as denominator.
    hw_denominator: str | None = None
    fidelity: str = "direct"
    architecture_scope: str = "nvidia_generic"
    state: str | None = None          # set when category == UNAVAILABLE
    limitations: str = ""
    # Reducer for SWEEP probes: knee / plateau / ratio / min_max.
    reducer: str | None = None


def _u(state: str, limitations: str, scope: str = "nvidia_generic") -> ProbePolicy:
    return ProbePolicy(
        category=UNAVAILABLE, derivation=NONE, fidelity="unsupported",
        architecture_scope=scope, state=state, limitations=limitations,
    )


METRICS_MAP: dict[str, ProbePolicy] = {
    # --- Compute & Scheduling ---
    "arithmetic_latency.dependent_chain": ProbePolicy(
        COMPARABLE, PER_OP, hw_denominator="chain_length", fidelity="direct",
        architecture_scope="nvidia_generic", limitations="sim FP latency from config",
    ),
    "arithmetic_throughput.independent_chains": ProbePolicy(
        COMPARABLE, PER_OP, hw_denominator="chain_length", fidelity="direct",
        limitations="ILP-saturated FP32; per-op over the dependent chain length",
    ),
    "scheduler_policy.ready_warps": ProbePolicy(
        APPROXIMATE, SWEEP, hw_denominator="total_ops", fidelity="proportional",
        reducer="knee", limitations="saturation needs warp-count sweep",
    ),
    "scheduler_policy.mixed_issue": _u(UNSUPPORTED, "behavioral pipe-overlap class, NCU-coupled"),
    "scheduler_policy.analyze": _u(NOT_APPLICABLE, "analysis-only, no kernel"),
    "topology.device_attributes": _u(NOT_APPLICABLE, "metadata, not a sim output"),
    "topology.occupancy": _u(NOT_APPLICABLE, "planning artifact, no kernel"),
    "topology.persistent_cta": ProbePolicy(
        APPROXIMATE, SWEEP, hw_denominator=None, fidelity="proxy",
        reducer="plateau", limitations="residency stat if sim exposes it",
    ),
    # --- Register, Tensor & Sync ---
    "register_file.register_bank_sweep": ProbePolicy(
        APPROXIMATE, SWEEP, hw_denominator="total_ops", fidelity="proxy",
        architecture_scope="h100_specific", reducer="plateau",
        limitations="sim reg-bank model (config banks) bounds plateau",
    ),
    "register_file.register_latency": ProbePolicy(
        APPROXIMATE, DIFFERENTIAL, hw_denominator="chain_length", fidelity="proportional",
        limitations="needs two sub-kernels (same vs rotating)",
    ),
    "register_file.analyze": _u(NOT_APPLICABLE, "analysis-only"),
    "tensor_core.mma_latency": ProbePolicy(
        COMPARABLE, PER_OP, hw_denominator="chain", fidelity="direct",
        architecture_scope="nvidia_hopper", limitations="HMMA dep; sim tensor latency from config",
    ),
    "tensor_core.mma_throughput": ProbePolicy(
        APPROXIMATE, THROUGHPUT, hw_denominator="mma_per_cycle_per_warp", fidelity="proxy",
        architecture_scope="nvidia_hopper",
        limitations="HW reports mma/cycle directly; sim throughput is a proxy comparison",
    ),
    "synchronization.barrier_latency": ProbePolicy(
        COMPARABLE, PER_OP, hw_denominator="barriers", fidelity="direct",
        limitations="BAR latency",
    ),
    "synchronization.fence_latency": ProbePolicy(
        APPROXIMATE, DIFFERENTIAL, hw_denominator="fences", fidelity="proxy",
        limitations="fence semantics differ; needs empty-loop baseline",
    ),
    # --- On-chip Memory ---
    "shared_memory.pointer_chase": ProbePolicy(
        COMPARABLE, PER_OP, hw_denominator="chase_len", fidelity="direct",
        limitations="LDS dep latency",
    ),
    "shared_memory.bank_stride": _u(UNSUPPORTED, "bank count is a sim input (shmem_num_banks)"),
    "shared_memory.analyze": _u(NOT_APPLICABLE, "analysis-only"),
    "l1_cache.pointer_chase": ProbePolicy(
        COMPARABLE, PER_OP, hw_denominator="steps", fidelity="direct",
        limitations="L1-hit dep load",
    ),
    "l1_cache.working_set": ProbePolicy(
        APPROXIMATE, SWEEP, hw_denominator="steps", fidelity="proportional",
        reducer="knee", limitations="capacity knee needs size sweep",
    ),
    "l1_cache.conflict_sets": _u(UNSUPPORTED, "associativity knee not identifiable from one trace"),
    "l1_cache.analyze": _u(NOT_APPLICABLE, "analysis-only"),
    "l2_cache.pointer_chase": ProbePolicy(
        COMPARABLE, PER_OP, hw_denominator="steps", fidelity="direct",
        limitations="L2-hit dep load",
    ),
    "memory_pipeline.outstanding_requests": ProbePolicy(
        APPROXIMATE, SWEEP, hw_denominator=None, fidelity="proportional",
        reducer="knee", limitations="MLP knee needs in-flight sweep",
    ),
    "memory_pipeline.lane_patterns": ProbePolicy(
        UNAVAILABLE, NONE, fidelity="proxy", state=PROXY_ONLY,
        limitations="sectors/request via counter proxy only (not a probe scalar)",
    ),
    "memory_pipeline.analyze": _u(NOT_APPLICABLE, "analysis-only"),
    # --- Global Memory & DRAM ---
    "global_memory.streaming": ProbePolicy(
        COMPARABLE, BANDWIDTH, hw_denominator=None, fidelity="proportional",
        limitations="sim DRAM bytes (reads+writes x atom) / sim time at core clock",
    ),
    "global_memory.partition_sweep": _u(UNSUPPORTED, "partition camping not modeled"),
    "global_memory.row_policy_sweep": ProbePolicy(
        APPROXIMATE, SWEEP, hw_denominator=None, fidelity="proportional",
        reducer="ratio", limitations="best/worst BW across stride traces",
    ),
    "global_memory.analyze": _u(NOT_APPLICABLE, "analysis-only"),
    # --- Transfer & Interconnect ---
    "tma_copy.async_copy_latency": ProbePolicy(
        APPROXIMATE, PER_OP, hw_denominator="tiles", fidelity="proxy",
        architecture_scope="nvidia_hopper", limitations="LDGSTS modeled; native TMA not",
    ),
    "tma_copy.tma_transfer_sweep": ProbePolicy(
        APPROXIMATE, SWEEP, hw_denominator=None, fidelity="proxy",
        architecture_scope="nvidia_hopper", reducer="min_max",
        limitations="cp.async modeled; bandwidth across tile sizes",
    ),
    "tma_copy.analyze": _u(NOT_APPLICABLE, "analysis-only"),
    "interconnect.injection_rate": ProbePolicy(
        COMPARABLE, BANDWIDTH, hw_denominator=None, fidelity="proportional",
        limitations="aggregate injection: sim DRAM bytes / sim time",
    ),
    "interconnect.address_mapping": _u(UNSUPPORTED, "address mapping not comparable"),
    "interconnect.analyze": _u(NOT_APPLICABLE, "analysis-only"),
}


# --- Single source of truth: drift guard against the NVIDIA registry. ---
_NVIDIA_IDS = set(NVIDIA_PROBES)
_MAP_IDS = set(METRICS_MAP)
assert _MAP_IDS == _NVIDIA_IDS, (
    "gcom_cuda metrics_map drifted from nvidia registry: "
    f"missing={sorted(_NVIDIA_IDS - _MAP_IDS)}, extra={sorted(_MAP_IDS - _NVIDIA_IDS)}"
)


def category_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for policy in METRICS_MAP.values():
        counts[policy.category] = counts.get(policy.category, 0) + 1
    return counts
