"""Metric semantics do not depend on the first TPU generation."""

METRICS = {
    "dispatch_to_completion_ns": {"unit": "ns", "source": "perf_counter_ns + output drain",
                                  "scope": "synchronized host interval"},
    "effective_flops_per_second": {"unit": "FLOP/s", "source": "logical work / host interval",
                                   "scope": "compiled program; not MXU utilization"},
    "effective_bytes_per_second": {"unit": "byte/s", "source": "requested bytes / host interval",
                                   "scope": "program or transfer; not physical HBM traffic"},
}
