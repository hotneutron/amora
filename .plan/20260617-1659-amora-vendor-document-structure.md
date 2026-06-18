# AMORA Vendor Document Structure — 2026-06-17

## Name

**AMORA** = **Automated Microarchitecture Observation and Reverse-engineering for Accelerators**

Tagline:

```text
Amora: lovingly mapping accelerator microarchitecture.
```

The codebase will live at:

```text
~/wk/amora
```

## Recommended Folder Structure

Use `docs/vendors/` for vendor and platform knowledge. Keep raw vendor documents,
curated notes, metric mappings, ISA notes, and probe plans separate so the tree
does not become a dumping ground.

```text
~/wk/amora/
  README.md
  pyproject.toml

  amora/
    core/
    backends/
    probes/
    reports/
    schemas/

  docs/
    architecture/
      overview.md
      probing-methodology.md
      counter-taxonomy.md
      parameter-taxonomy.md

    vendors/
      nvidia/
        README.md
        sources.md
        cuda/
          docs.md
          isa.md
          runtime.md
          profiling.md
          counters.md
          metrics-map.yaml
          probe-plan.md
        nsight-compute/
          docs.md
          cli.md
          sections.md
          metrics.md
          pm-sampling.md
          source-sass-attribution.md
        cupti/
          docs.md
          range-profiling.md
          host-profiling.md
          pm-sampling.md
          pc-sampling.md
          sass-metrics.md
          api-notes.md
        nvbit/
          docs.md
          instrumentation-model.md
          examples.md
          limitations.md
        architectures/
          volta.md
          turing.md
          ampere.md
          hopper.md
          blackwell.md

      amd/
        README.md
        sources.md
        rocm/
          docs.md
          runtime.md
          profiling.md
          counters.md
          metrics-map.yaml
          probe-plan.md
        rocprof/
          docs.md
          cli.md
          counters.md
          tracing.md
        rocprofiler-sdk/
          docs.md
          api-notes.md
          counter-collection.md
        omnitrace/
          docs.md
          tracing.md
        isa/
          gcn.md
          rdna.md
          cdna.md
        architectures/
          gfx9.md
          gfx10.md
          gfx11.md
          gfx12.md
          mi200.md
          mi300.md

      intel/
        README.md
        sources.md
        level-zero/
          docs.md
          runtime.md
          metrics.md
          metrics-map.yaml
          probe-plan.md
        oneapi/
          docs.md
          sycl.md
          profiling.md
        vtune/
          docs.md
          gpu-hotspots.md
          counters.md
        architectures/
          gen.md
          xe-hpg.md
          xe-hpc.md
          gaudi.md

      apple/
        README.md
        sources.md
        metal/
          docs.md
          runtime.md
          profiling.md
          counters.md
          probe-plan.md
        xcode-instruments/
          docs.md
          gpu-counters.md
        architectures/
          apple-gpu.md
          ane.md

      google/
        README.md
        sources.md
        tpu/
          docs.md
          profiling.md
          counters.md
          probe-plan.md
        xla/
          docs.md
          hlo.md
          profiling.md
        architectures/
          tpu-v2.md
          tpu-v3.md
          tpu-v4.md
          tpu-v5.md

      qualcomm/
        README.md
        sources.md
        adreno/
          docs.md
          isa.md
          profiling.md
          counters.md
          probe-plan.md
        hexagon/
          docs.md
          profiling.md
          counters.md
        snpe-qnn/
          docs.md
          runtime.md

      arm/
        README.md
        sources.md
        mali/
          docs.md
          profiling.md
          counters.md
          probe-plan.md
        streamline/
          docs.md
          counters.md
        architectures/
          bifrost.md
          valhall.md
          immortalis.md

      imagination/
        README.md
        sources.md
        powervr/
          docs.md
          profiling.md
          counters.md
          probe-plan.md

      tenstorrent/
        README.md
        sources.md
        docs.md
        isa.md
        profiling.md
        counters.md
        probe-plan.md
        architectures/
          wormhole.md
          blackhole.md

      cerebras/
        README.md
        sources.md
        docs.md
        profiling.md
        counters.md
        probe-plan.md
        architectures/
          wse.md
          wse2.md
          wse3.md

      groq/
        README.md
        sources.md
        docs.md
        profiling.md
        counters.md
        probe-plan.md
        architectures/
          lpu.md

      samba-nova/
        README.md
        sources.md
        docs.md
        profiling.md
        counters.md
        probe-plan.md

      graphcore/
        README.md
        sources.md
        ipu/
          docs.md
          profiling.md
          counters.md
          probe-plan.md
        architectures/
          ipu-mk1.md
          bow.md

      cambricon/
        README.md
        sources.md
        mlu/
          docs.md
          profiling.md
          counters.md
          probe-plan.md

      ascend/
        README.md
        sources.md
        cann/
          docs.md
          profiling.md
          counters.md
          probe-plan.md
        architectures/
          davinci.md

      biren/
        README.md
        sources.md
        docs.md
        profiling.md
        counters.md
        probe-plan.md

      moore-threads/
        README.md
        sources.md
        musa/
          docs.md
          runtime.md
          profiling.md
          counters.md
          probe-plan.md

      generic/
        README.md
        unknown-isa-checklist.md
        counter-discovery-checklist.md
        runtime-discovery-checklist.md
        black-box-probing-checklist.md
        vendor-questionnaire.md

    papers/
      README.md
      gpu-microbenchmarking.md
      performance-counters.md
      binary-instrumentation.md
      tensor-accelerators.md

    glossary/
      counters.md
      isa.md
      memory-hierarchy.md
      synchronization.md

  data/
    vendor-doc-index/
      nvidia.yaml
      amd.yaml
      intel.yaml
      apple.yaml
      google.yaml
      qualcomm.yaml
      arm.yaml
      lesser-known.yaml

    metric-maps/
      nvidia-ncu.yaml
      nvidia-cupti.yaml
      amd-rocprof.yaml
      intel-level-zero.yaml
      apple-metal.yaml
      google-tpu.yaml
      generic.yaml

    hardware-profiles/
      README.md

  tools/
    doc_fetch/
    metric_scrape/
    report_render/
```

## Vendor Document Conventions

Each vendor directory should follow the same pattern:

| File | Purpose |
|---|---|
| `README.md` | Human-readable overview of the vendor stack and what AMORA supports. |
| `sources.md` | Canonical list of official docs, repos, papers, forum posts, and reverse-engineering references. |
| `docs.md` | Curated notes from vendor documentation. |
| `runtime.md` | Runtime APIs, launch model, memory allocation, synchronization, streams/queues. |
| `isa.md` | ISA naming, instruction formats, disassembly tools, special instructions. |
| `profiling.md` | Profiling tools and workflows. |
| `counters.md` | Available PMCs, metric names, units, collection constraints. |
| `metrics-map.yaml` | Mapping from vendor metric names to AMORA's normalized schema. |
| `probe-plan.md` | Vendor-specific probes and expected limitations. |
| `architectures/*.md` | Generation-specific notes. |

## Why This Structure

The suite needs to cover CUDA, ROCm, NPUs, TPUs, and less-known accelerators
without letting one vendor dominate the abstraction. The structure separates:

- vendor docs from AMORA code,
- raw source lists from curated interpretation,
- profiling counters from ISA instrumentation,
- runtime behavior from microarchitecture,
- mature GPU stacks from less-documented accelerator stacks.

This keeps AMORA vendor-neutral while still allowing deep vendor-specific
knowledge where it exists.

## Normalized Cross-Vendor Categories

Every vendor should eventually map its docs and counters into these AMORA
categories:

1. Topology and occupancy
2. Issue, scheduling, and execution throughput
3. Register file and operand delivery
4. Shared / local memory
5. L1 / local cache
6. L2 / shared cache
7. DRAM / HBM / external memory
8. Interconnect and address mapping
9. Tensor / matrix / systolic engines
10. DMA / async copy / TMA-like engines
11. Synchronization and barriers
12. Instruction mix and ISA behavior
13. Power, clocks, throttling, and DVFS

## Naming Notes

Use lowercase paths and hyphenated directory names:

- `samba-nova/`, not `SambaNova/`
- `moore-threads/`, not `MooreThreads/`
- `rocprofiler-sdk/`, not `rocProfilerSDK/`

Keep vendor product names inside document titles and prose. Directory names
should remain stable even if vendor branding changes.
