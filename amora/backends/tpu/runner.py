"""Synchronized host timing with explicit batching and retained output leaves."""
from __future__ import annotations

import math
import time

from amora.core.statistics import summarize_samples


def drain(tree):
    if isinstance(tree, dict):
        for value in tree.values():
            drain(value)
    elif isinstance(tree, (tuple, list)):
        for value in tree:
            drain(value)
    elif hasattr(tree, "block_until_ready"):
        tree.block_until_ready()


def time_program(invoke, *, warmup=5, samples=30, repeats=3, batch=1, max_relative_mad=0.05):
    for name, value in (("warmup", warmup), ("samples", samples), ("repeats", repeats), ("batch", batch)):
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "warmup" else 1):
            raise ValueError(f"invalid {name}")
    if not math.isfinite(max_relative_mad) or max_relative_mad < 0:
        raise ValueError("invalid max_relative_mad")
    for _ in range(warmup):
        outputs = [invoke() for _ in range(batch)]
        drain(outputs)
    raw, summaries = [], []
    for repeat in range(repeats):
        series = []
        for sample in range(samples):
            started = time.perf_counter_ns()
            outputs = [invoke() for _ in range(batch)]
            drain(outputs)
            elapsed = time.perf_counter_ns() - started
            series.append(elapsed)
            raw.append({"repeat": repeat, "sample": sample,
                        "dispatch_to_completion_ns": elapsed, "invocations": batch})
        stats = summarize_samples(series)
        stats["relative_mad"] = stats["mad"] / stats["median"] if stats["median"] else float("inf")
        summaries.append(stats)
    medians = [s["median"] for s in summaries]
    stability = summarize_samples(medians)
    stable = all(s["relative_mad"] <= max_relative_mad for s in summaries)
    stable &= stability["mad"] <= max_relative_mad * stability["median"]
    return {"time_domain": "synchronized_host", "timed_unit": "single_invocation" if batch == 1
            else "batched_invocations", "warmup_iterations": warmup,
            "warmup_invocations":warmup*batch,
            "warmup_policy":"same batch, output retention and drain as measured units",
            "measured_samples_per_repeat": samples, "repeats": repeats, "batch_size": batch,
            "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
            "input_donation": False, "output_reuse": False,
            "summary": stability, "repeat_summaries": summaries, "stable": bool(stable),
            "median_ns_per_invocation": stability["median"] / batch, "samples": raw}
