"""Benchmark materialization and execution contracts."""

from amora.benchmarking.materialize import materialize_benchmark
from amora.benchmarking.registry import get_benchmark, list_benchmarks

__all__ = ["get_benchmark", "list_benchmarks", "materialize_benchmark"]
