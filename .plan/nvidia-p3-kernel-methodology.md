# NVIDIA P3 Kernel Methodology

## Scope

This document defines probing methodology for P3 probes listed in
`/Users/bytedance/wk/amora/.plan/nvidia-probe-semantic-measurement-gap-plan.md`.
P3 probes have the largest semantic and measurement gaps. They should be
implemented only after P0-P2 baselines are stable.

P3 covers:

- TMA, DMA, and async copy behavior
- interconnect and address mapping behavior

Most P3 target parameters are simulator internal structures. The goal is not to
claim direct hardware equivalence. The goal is to produce effective behavioral
parameters and clearly identify what remains unobservable.

## Common Methodology Rules

1. Require P0-P2 baseline inputs:
   - occupancy and clocks
   - arithmetic throughput
   - shared-memory behavior
   - L1/L2/DRAM latency and bandwidth
   - scheduler and memory-pipeline coupling notes
2. Use architecture feature detection before running any P3 probe.
3. Treat all unsupported instructions, scopes, and metrics as explicit
   `unsupported` outputs.
4. Use CUPTI/NCU metrics as supporting evidence, not definitive proof of internal
   queue or router state.
5. Use PC Sampling and SASS Metrics for attribution when supported.
6. Use NVBit only in separate validation runs for dynamic instruction or memory
   stream checks.
7. Emit effective parameters, ranges, and alternatives instead of a single false
   precision value.

Risk scale:

- Medium: stable behavior with a close metric or direct instruction semantic.
- High: behavior is coupled with multiple already-measured subsystems.
- Very high: target parameter is a simulator-only internal structure.

## Probe: `tma_copy/async_copy_latency.cu`

### Goal

Measure issue-to-observable-completion latency for async-copy or TMA-like copy
instructions where the architecture supports them.

### Parameters

- `tma_unit_sm::kMaxRequestsPerCycle`
- `TMACommand`
- `TMATransferEntry`
- `TMAOpcodeFamily`
- `TMADirection`
- `TMATransferType`
- `TMAOperandForm`
- `m_command_queue`
- `m_in_flight_transfers`
- `m_outstanding_requests`
- `m_outstanding_stores_per_warp`

### Methodology

1. Detect architecture support:
   - compute capability
   - CUDA toolkit support
   - assembler support for async-copy/TMA instructions
   - required barrier/wait instructions
2. Generate minimal copy kernels:
   - one copy command
   - controlled source and destination spaces
   - required wait or barrier completion path
3. Measure:
   - issue-to-wait latency
   - wait-only baseline
   - barrier-arrive/wait overhead
   - completion latency for zero or minimal payload when legal
4. Sweep:
   - transfer size
   - alignment
   - source/destination space
   - one warp versus multiple warps
   - one CTA versus multiple CTAs
5. Use device-side timing around the copy and completion sequence.
6. Disassemble to verify the exact async/TMA instruction and wait sequence.
7. Collect NCU/CUPTI memory, stall, and async-copy-related metrics where
   available.
8. Use PC Sampling to attribute wait stalls to barrier, memory dependency, or
   dispatch categories where available.

### Reasoning

Async copy and TMA instructions expose multiple latency components: command
issue, transfer service, barrier/completion notification, and wait. A minimal
latency probe isolates the fixed components before transfer-size sweeps add
bandwidth effects.

### Risk Estimate

Risk: High to Very High.

Main risks:

- TMA support is architecture-specific and may not be accessible from a simple
  CUDA kernel on all targets.
- The simulator names are internal abstractions, not public hardware counters.
- Observed latency is coupled with shared memory, L2, DRAM, barriers, and warp
  scheduling.
- Compiler or assembler restrictions can change the generated instruction
  sequence.

Mitigation:

- Feature-gate every variant.
- Store disassembly and reject unexpected instruction forms.
- Report latency components separately.
- Mark queue and transfer-entry estimates as `coupled_inference` unless a direct
  metric exists.

## Probe: `tma_copy/tma_transfer_sweep.cu`

### Goal

Measure steady-state async/TMA transfer bandwidth, outstanding transfer limits,
and completion behavior over transfer geometry.

### Parameters

- `tma_unit_sm::kMaxRequestsPerCycle`
- `m_command_queue`
- `m_in_flight_transfers`
- `m_outstanding_requests`
- `m_outstanding_stores_per_warp`
- `TMADirection`
- `TMATransferType`
- `TMAOperandForm`

### Methodology

1. Build a family of transfer kernels with controlled producer/consumer state.
2. Sweep:
   - transfer bytes
   - rank or dimensionality when supported
   - stride
   - alignment
   - number of outstanding commands
   - warps issuing commands
   - CTAs per SM
3. Separate phases:
   - command issue phase
   - transfer-in-flight phase
   - wait/completion phase
4. Measure:
   - bytes per cycle
   - commands per cycle
   - completion latency
   - throughput plateau
   - cliff when outstanding request capacity is exceeded
5. Collect NCU/CUPTI shared, L1TEX, L2, DRAM, and stall metrics.
6. Use PM Sampling for long-running producer/consumer variants when replay would
   distort overlap behavior.

### Reasoning

Transfer sweeps reveal whether the limiting factor is command issue, in-flight
capacity, memory bandwidth, or completion synchronization. The saturation point
is the best public proxy for simulator queue-depth and request-rate fields.

### Risk Estimate

Risk: Very High.

Main risks:

- Transfer engine behavior may be undocumented and architecture-specific.
- Bandwidth plateaus can be dominated by L2/DRAM rather than TMA command limits.
- In-flight capacity may not show a clean cliff if backpressure is gradual.
- Replay profiling may perturb async overlap.

Mitigation:

- Compare transfer results against P2 global-memory and synchronization
  baselines.
- Use PM Sampling for phase behavior.
- Report multiple candidate bottleneck explanations.
- Keep TMA queue estimates low confidence unless cliffs are stable and counters
  agree.

## Probe: `tma_copy/analyze.py`

### Goal

Fit effective async/TMA command, transfer, and completion behavior.

### Methodology

1. Split measurements into fixed latency, bandwidth, and wait components.
2. Fit:
   - minimum issue-to-completion latency
   - steady-state bytes per cycle
   - commands per cycle
   - outstanding command cliff or saturation range
   - wait/completion overhead
3. Compare against P2:
   - shared-memory latency
   - L2/DRAM bandwidth
   - barrier/fence latency
4. Emit simulator mappings only as behavioral equivalents:
   - command queue capacity equivalent
   - in-flight transfer equivalent
   - request-rate equivalent
5. Mark unobservable fields explicitly.

### Reasoning

Most TMA simulator parameters describe internal state machines. Public probes can
only measure external behavior. The analyzer must keep this distinction visible.

### Risk Estimate

Risk: Very High.

Main risks:

- Many internal configurations can produce the same observable transfer curve.
- Tool metrics may not name TMA units consistently across architectures.

Mitigation:

- Emit ranges and candidate explanations.
- Use `unsupported` or `indeterminate` rather than overfitting.
- Link every TMA estimate to P2 memory and synchronization baselines.

## Probe: `interconnect/address_mapping.cu`

### Goal

Infer effective address-to-partition and address-to-cache-slice behavior.

### Parameters

- `gpgpu_mem_addr_mapping`
- `gpgpu_mem_address_mask`
- `memory_config::m_n_mem`
- `memory_config::m_n_sub_partition_per_memory_channel`

### Methodology

1. Allocate large global-memory regions.
2. Generate kernels that access controlled address sets.
3. Sweep:
   - base address
   - stride
   - selected address bits
   - buffer size
   - read versus write
   - one CTA versus many CTAs
4. Measure bandwidth and latency for each candidate mapping pattern.
5. Detect partition-camping signatures:
   - periodic bandwidth drops
   - latency spikes
   - uneven L2/DRAM metric instances if exposed
6. Use NCU/CUPTI partition, L2 slice, or memory-controller metrics when
   available.
7. Repeat across allocations to account for virtual-to-physical mapping noise.

### Reasoning

Address mapping is observable only through traffic imbalance. If certain address
bits select partitions or slices, controlled sweeps should reveal periodic
contention patterns.

### Risk Estimate

Risk: Very High.

Main risks:

- Physical address bits are usually not directly controlled.
- NVIDIA may hash address bits in undocumented ways.
- L2 slice mapping and DRAM partition mapping may not be the same.
- Page allocation, compression, and memory placement can distort patterns.

Mitigation:

- Use many base addresses and allocation sizes.
- Score multiple mapping hypotheses.
- Report candidates and confidence instead of a single mapping when evidence is
  ambiguous.
- Reuse P2 partition-sweep evidence as a prerequisite.

## Probe: `interconnect/injection_rate.cu`

### Goal

Measure latency and bandwidth under controlled traffic injection rates to fit
effective interconnect saturation behavior.

### Parameters

- `icnt_flit_size`
- `routing_delay`
- `vc_alloc_delay`
- `sw_alloc_delay`
- `credit_delay`
- `input_speedup`
- `output_speedup`
- `internal_speedup`
- `output_buffer_size`
- `use_noc_latency`

### Methodology

1. Generate traffic kernels with controlled per-SM injection rates.
2. Sweep:
   - number of active SMs
   - CTAs per SM
   - requests per warp
   - read versus write traffic
   - partition-balanced versus partition-camped addresses
3. Measure:
   - latency under load
   - throughput under load
   - saturation point
   - backpressure onset
4. Use NCU/CUPTI L2, DRAM, and fabric/interconnect metrics if available.
5. Compare balanced and imbalanced traffic to distinguish downstream DRAM limits
   from interconnect pressure.
6. Use PM Sampling for long-running saturation phases.

### Reasoning

Simulator interconnect fields are internal NoC/router parameters. Public probes
can only observe effective latency and throughput under injection pressure. The
shape of latency-versus-load and throughput-versus-load curves constrains
effective routing and buffering behavior.

### Risk Estimate

Risk: Very High.

Main risks:

- Interconnect pressure is hard to separate from L2, memory partitions, and DRAM
  scheduler behavior.
- Router-level parameters such as VC allocation and switch allocation are not
  publicly exposed.
- Fabric metrics may be unavailable or architecture-specific.

Mitigation:

- Run only after P2 L2/DRAM and memory-pipeline models exist.
- Fit effective saturation curves rather than individual router microparameters.
- Keep router/VC fields low confidence unless supported by strong counters.

## Probe: `interconnect/analyze.py`

### Goal

Fit effective address-mapping and interconnect saturation parameters from P3
traffic probes.

### Methodology

1. Score address-mapping hypotheses against partition-camping observations.
2. Fit latency-versus-injection-rate curves.
3. Fit throughput saturation points for balanced and imbalanced traffic.
4. Compare against P2:
   - L2 bandwidth
   - DRAM bandwidth
   - memory partition count candidates
   - coalescing behavior
5. Emit:
   - effective address mapping candidates
   - effective interconnect latency range
   - effective saturation throughput
   - unresolved simulator router fields
6. Use `indeterminate` for fields that cannot be separated from downstream
   memory behavior.

### Reasoning

The analyzer is a guardrail against false precision. It should say which
simulator fields are constrained by observations and which fields remain
unobservable through public CUDA/CUPTI/NVBit methods.

### Risk Estimate

Risk: Very High.

Main risks:

- Overfitting router parameters from end-to-end memory curves.
- Assuming simulator NoC structure matches NVIDIA hardware organization.
- Confusing address hashing with physical memory allocation effects.

Mitigation:

- Report effective behavioral calibration values.
- Keep raw curves and candidate models.
- Require agreement across multiple traffic shapes before raising confidence.

## P3 Output Requirements

Every P3 result should include:

- P0-P2 prerequisite result references
- architecture feature checks
- instruction support and disassembly evidence
- raw latency and bandwidth curves
- NCU/CUPTI metric availability and values
- PC Sampling or SASS Metrics attribution when used
- PM Sampling timeline when used
- NVBit validation stream paths when used
- fitted effective parameters
- unresolved alternatives
- explicit `unsupported` or `indeterminate` fields
- risk and coupling notes

## P3 Acceptance Criteria

P3 is complete when AMORA can report:

- async/TMA support status for the target GPU
- minimum async/TMA issue-to-completion latency when supported
- async/TMA transfer bandwidth curve when supported
- candidate outstanding-transfer saturation behavior or explicit indeterminate
  status
- candidate address-mapping evidence or explicit indeterminate status
- balanced versus imbalanced interconnect/memory traffic curves
- effective interconnect saturation behavior
- a list of simulator TMA and interconnect fields that remain unobservable
  through public tooling

## P3 Non-Goals

P3 should not claim:

- exact NVIDIA router topology
- exact VC allocation delay
- exact switch allocation delay
- exact TMA internal queue layout
- exact physical address mapping

Those are not public hardware facts in the available toolchain. AMORA should
represent them as effective calibration parameters or leave them unresolved.
