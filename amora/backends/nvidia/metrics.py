"""Logical NVIDIA metric resolver for baseline probes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricResolution:
    logical_name: str
    selected_name: str | None
    candidates: tuple[str, ...]
    available: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_name": self.logical_name,
            "selected_name": self.selected_name,
            "candidates": list(self.candidates),
            "available": self.available,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MetricResolver:
    supported_metrics: frozenset[str] = field(default_factory=frozenset)

    CANDIDATES = {
        "sm_active_cycles": (
            "sm__cycles_active.avg",
            "smsp__cycles_active.avg",
        ),
        "inst_executed": (
            "smsp__inst_executed.sum",
            "sm__inst_executed.sum",
        ),
        "shared_transactions": (
            "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
            "l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum",
        ),
        "shared_conflicts": (
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
        ),
        # Global-memory request / sector behavior (memory_pipeline).
        "global_load_requests": (
            "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
        ),
        "global_load_sectors": (
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        ),
        # DRAM traffic (global_memory).
        "dram_bytes_read": (
            "dram__bytes_read.sum",
        ),
        "dram_bytes_write": (
            "dram__bytes_write.sum",
        ),
        "dram_throughput": (
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg",
        ),
        # L2 sector hits (l2_cache / global_memory).
        "l2_sector_hits": (
            "lts__t_sectors_lookup_hit.sum",
        ),
        # Tensor-pipe utilization (tensor_core).
        "tensor_pipe_active": (
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "sm__inst_executed_pipe_tensor.sum",
        ),
    }

    def resolve(self, logical_name: str) -> MetricResolution:
        candidates = self.CANDIDATES.get(logical_name, ())
        if not candidates:
            return MetricResolution(
                logical_name=logical_name,
                selected_name=None,
                candidates=(),
                available=False,
                reason="unknown logical metric",
            )
        for candidate in candidates:
            if self._is_supported(candidate):
                return MetricResolution(
                    logical_name=logical_name,
                    selected_name=candidate,
                    candidates=candidates,
                    available=True,
                )
        return MetricResolution(
            logical_name=logical_name,
            selected_name=None,
            candidates=candidates,
            available=False,
            reason="no candidate metric supported",
        )

    def _is_supported(self, candidate: str) -> bool:
        """Match a suffixed candidate against the supported set.

        ``ncu --query-metrics`` reports *base* metric names without the trailing
        rollup suffix (``.sum`` / ``.avg`` / ``.max`` / ...). A candidate like
        ``smsp__inst_executed.sum`` is supported when its base
        (``smsp__inst_executed``) is in the discovered set. Exact matches are
        also accepted for sets that include suffixes.
        """

        if candidate in self.supported_metrics:
            return True
        base = candidate.rsplit(".", 1)[0]
        return base in self.supported_metrics
