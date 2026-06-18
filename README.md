# AMORA

AMORA is Automated Microarchitecture Observation and Reverse-engineering for
Accelerators.

The current implementation starts with NVIDIA P0 probes:

- topology and occupancy metadata
- arithmetic latency and throughput source generation
- shared-memory latency and bank-behavior source generation

The Python entrypoint can run in dry-run mode on machines without CUDA:

```bash
python -m amora --target nvidia --tier p0 --dry-run
```
