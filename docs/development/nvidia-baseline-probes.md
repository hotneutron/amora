# NVIDIA Baseline Probe Development

## Current Cutline

The current baseline implementation provides the runnable spine for NVIDIA
probes:

- package and CLI scaffold,
- layered result schemas,
- JSON report rendering,
- NVIDIA capability discovery,
- `topology.device_attributes` metadata probe,
- structured unsupported results for planned CUDA probes,
- CUDA source templates for the baseline kernel families,
- non-hardware unit tests.

The CUDA timing and profiler probes are intentionally not treated as implemented
until build orchestration, SASS validation, metric collection, and analysis are
connected.

## Commands

List NVIDIA probes:

```bash
.venv/bin/python -m amora.cli nvidia list
```

Inspect NVIDIA backend capabilities:

```bash
.venv/bin/python -m amora.cli nvidia inspect-capabilities
```

Run all baseline NVIDIA probes and write a JSON report:

```bash
.venv/bin/python -m amora.cli nvidia run --all --output out/nvidia-baseline.json
```

Run only topology metadata:

```bash
.venv/bin/python -m amora.cli nvidia run --probe topology.device_attributes
```

## Output Model

Every result preserves four layers:

- `raw_observation`
- `normalized_measurement`
- `backend_interpretation`
- `simulator_estimate`

Unsupported probes still emit all four layers. This keeps reports stable on
machines without CUDA hardware and prevents missing-tool failures from becoming
silent scalar estimates.

## Next Implementation Steps

1. Replace `nvidia-smi`-only topology identity with a CUDA API helper for SM
   count, warp size, resident limits, register limits, and shared-memory limits.
2. Add CUDA build orchestration for templates under the internal baseline probe
   package.
3. Add SASS validation using `nvdisasm` or `cuobjdump`.
4. Add NCU metric discovery and metric resolver integration.
5. Promote each planned unsupported runner to implemented status only after it
   can collect evidence and enforce rejection/downgrade rules.
