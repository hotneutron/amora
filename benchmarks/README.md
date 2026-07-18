# Benchmark Sources

This directory contains static upstream benchmark sources. AMORA-specific
selection manifests and replay contracts must live beside an upstream
submodule, not inside it.

Initialize benchmark sources after cloning AMORA:

```bash
git submodule update --init --recursive
```

| source | path | AMORA wrapper |
| --- | --- | --- |
| MLCommons Inference | `benchmarks/MLPerf` | `benchmarks/MLPerf.amora` |
