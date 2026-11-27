# Portable core v2 GCoM validation integration

## User request

```text
persist a plan to run the same validation/correlation experiments for portable_core v2 (both the op gates and H3 gates) with GCoM ~/wk/modern-gpu-simulator-micro-2025/ with same H100 target.
```

```text
execute the plan
```

## Layout and ownership

Workloads and the independent oracle stay in
`benchmark_generators/portable_core/expanded/v2/`. The unchanged CUDA node
realization is `amora/backends/nvidia/portable_core_v2.cu`. Capture, GPU leases,
trace admission, immutable replay and CPU/storage allocation live in
`amora/backends/gcom_cuda/portable_v2*.py`; the CUPTI collector and separate
batch64 realization support live under the NVIDIA backend. Campaign orchestration
and descriptive analysis live in `amora/benchmarking/portable_v2/gcom_*.py`.

The GPU performance project owns plans, trackers, sealed inputs, native samples,
builds, traces, replay caches and runtime state. These stay in its Git-ignored
`local_artifacts/gcom-portable-v2/`; reports belong in its `reports/` and
`reports/assets/`. Amora contains reusable code rather than a campaign database.

## Running the admitted campaign

The direct module CLI is separate from the portable package's mathematical
release checker. From the Amora checkout:

```bash
PYTHONPATH=. /home/cliu/wk/gpu-perf/local_artifacts/gcom-portable-v2/execution-env/bin/python \
  -m amora.benchmarking.portable_v2.gcom_campaign execute \
  --root /home/cliu/wk/gpu-perf/local_artifacts/gcom-portable-v2 \
  --gpu-wait-seconds 86400
```

`execute` advances through the two-case pipeline bootstrap, the frozen 39-case
operator/layout/H3 witnesses, all 1,155 unique cases, and the independent
batch64 H3 lane. It requires the admitted corpus, native binary, source build,
generated protobuf bindings and CUPTI collector. Runtime state and receipts
are authoritative; existing failed replay keys cannot be retried automatically.

To update figures and statistics from saved results without another simulation:

```bash
PYTHONPATH=. /home/cliu/wk/gpu-perf/local_artifacts/gcom-portable-v2/execution-env/bin/python \
  -m amora.benchmarking.portable_v2.gcom_campaign report \
  --root /home/cliu/wk/gpu-perf/local_artifacts/gcom-portable-v2
```

## Admission and interpretation

Each H100 needs 30 clean seconds, a UUID lease and continuous foreign-context
monitoring. Every hardware child sees its assigned physical UUID as CUDA device
0. Three native processes and 30 samples each supply device-matched references;
the historical GPU 7 labels cannot be transferred to another card.

The CUDA 13.3 binary remains unchanged. NVBit 1.8.2 needs the documented tool
thread API rename and CUkernel-to-CUfunction handle resolution before querying
metadata or instrumenting these launches. CUDA injection alone qualified this
shared-runtime executable. CUPTI collection uses a null-initialized record
iterator and requires complete kernel accounting with zero drops/errors.

The H100 80 GB preset retains its physical timing and residency values. Five
explicit compatibility overrides disable absent TMA/WaitBlock sidecar consumers.
The full static opcode inventory must prove no applicable asynchronous consumer;
ordinary named CTA barriers and memory fences keep their native model, including
Hopper `BAR.SYNC.DEFER_BLOCKING`.

The primary score compares the native stream program span with the simulator's
sequential target service timeline at 1.8 GHz, including its configured
1,500-cycle CTA launch delay and zero modeled host submission gaps. Prefix
validation and three warmup graphs execute for state and are excluded from the
scored window. CUPTI kernel-body durations are retained as a distinct diagnostic.
Batch64 has fresh native qualification and separate total/amortized results.

Native correctness, trace completeness, replay admission and timing eligibility
remain separate. Exact historical H3 references are `blocked_missing_source`;
development/smoke fixtures do not replace them. Formal gate decisions stay
`gate_not_evaluated` without approved split assignments and frozen criteria.

## Evidence integrity and capacity

The replay fingerprint binds simulator/library hashes, effective configurations,
trace/static/sidecar bytes and execution arguments. Aliases and figure changes
reuse the same result. CPU sets are disjoint and NUMA-local, with larger traces
requesting 32 OpenMP threads and smaller traces eight, within admitted CPU/RAM
limits. Hardware timing phases stay separate from CPU replay.

An 8 GiB disk reserve bounds capture. Lossless trace and stdout archives are
verified before raw duplicates are removed. Replay instruction-stat files are
retained in the immutable job folder. Case, launch and gate-scope ledgers expose
failed, missing and unqualified rows alongside successful results.

## Revision History

- 2026-11-25 19:25 -0800: Documented the source-owned integration, qualified H100 capture/replay contract, immutable evidence and explicit op/H3 scope. Full campaign and batch64 execution remain tracked in the GPU performance project.
