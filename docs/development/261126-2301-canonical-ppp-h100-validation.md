# Canonical PPP H100 capture and simulator validation

## User request

```text
Work on the Canonical PPP kernel results for both simulators.
```

## Ownership and scope

`benchmark_generators/ppp_canonical/` owns the unchanged nine CUDA templates,
shape candidates and execution contracts. Reusable campaign and reporting
code lives in `amora/benchmarking/canonical_ppp/`. The GCoM cubin text adapter
lives in `amora/backends/gcom_cuda/cubin_disassembly.py`.

The gpu-perf project owns its timestamped plan, tracker, reports and figures.
Runtime manifests, immutable model keys, GPU/CPU admission, captures, logs
and lossless archives stay in its Git-ignored
`local_artifacts/canonical-ppp-h100/`. The campaign uses the previously
validated GCoM build and collector from `local_artifacts/gcom-portable-v2/`.

## Native and simulator contracts

The standard deterministic manifest has 2,500 cases with seed 20260717. Nine
minimum-work shapes from the source generator provide family support pilots;
eight are additional unique cases. The source/build/input/argument identities
are retained. All lanes use the same CUDA 13.3 shared-runtime executable with
`AMORA_GCOM_TRACE`, removing warmup and embedding's explicit L2 flush.

Native measurement consists of three dedicated cold-process CUPTI captures.
Each capture must have no dropped records, ordered nonoverlapping launches,
one device/context/stream, and repeat-identical symbols/grids/blocks.
Scoring requires a median kernel span of at least 10 microseconds and
process-repeat CV at most 10 percent. Kernel-body sums remain separate from
spans including component gaps. Models retain their original service/launch
policies, clock domains and zero modeled host submission gaps.

The canonical templates have no independent output oracle. Completion is
timing evidence only. Uninitialized GEMM inputs, sampled attention/MLA and
unsigned-byte bounded MoE limitations are explicit in the report.

H100 residency is aligned to 132 SMs, 2,048 threads, 64 warps and 32 CTAs per
SM. No model timing parameter is fitted. Every deterministic simulator key
is executed once, including failures; later reports reparse existing output.

## Compatibility checks

GCoM appends a `___<variant>` suffix to CUDA symbols. Trace admission compares
original native symbols in order and retains complete variant names in its
evidence. Launch counts, dimensions, CTA/warp coverage, dynamic PCs and static
128-bit instruction encodings are verified. Real asynchronous consumers
cannot use the portable absent-feature sidecar overrides.

Some driver-loaded CUDA Std ELF cubins are accepted by CUDA 13.3 nvdisasm
but rejected by cuobjdump. A separately hashed command adapter renders the
legacy text framing expected by GCoM, resolves branch labels and removes
disassembler annotations. Every instruction encoding is compared against
the original ELF `.text` section. The unmodified GCoM CPU metadata parser
also accepted all 63,120 instructions in the pilot cuBLAS cubin before the
corrected capture was admitted. Failed capture attempts remain retained.

FlashGPU completion uses its CUDA runtime atexit marker, successful process
exit and a positive completed counter block for every launch. GCoM uses its
simulation-thread completion marker and exact trace-ID accounting. Their
markers are not interchangeable.

## Execution and reporting

The project creates `campaign-target.json` with its prompt and artifact paths
before preparing the campaign. Use the existing admitted environment:

```bash
PYTHONPATH=/home/cliu/wk/amora \
  /home/cliu/wk/gpu-perf/local_artifacts/gcom-portable-v2/execution-env/bin/python \
  -m amora.benchmarking.canonical_ppp.campaign execute \
  --root /home/cliu/wk/gpu-perf/local_artifacts/canonical-ppp-h100
```

Recompute reports and reference-layout figures with the same command using
`report` instead of `execute`; this does not invoke a simulator. Existing
attempted inputs are reused by the execution action.

Dedicated GPU UUID leases use continuous foreign-context rejection. CPU
allocation excludes observed busy cores and their SMT siblings, obeys quota,
and assigns disjoint NUMA sets. Default execution now resumes every primary identity through `full_campaign.py`,
first collecting native labels and then staging traces/model work by estimated
bytes and actual CPU/RAM/storage admission. The historical initial bounded
stage is available as `execute-bounded`. GPU use requires an explicit approval
receipt; idle observations and elapsed time do not approve a launch.

The handoff module uses inotify to replace a controller only at an expansion
boundary with no descendant jobs, leased GPUs or unfinished simulator keys.
The completion finalizer refreshes the canonical/combined reports, figures,
tracker, source hashes and links. It does not repeat a model execution.

## Verification

```bash
PYTHONPATH=. <execution-env>/bin/python -m pytest -q \
  tests/benchmarks/test_canonical_campaign.py \
  tests/backends/test_cubin_disassembly.py \
  tests/backends/test_gcom_portable_v2.py \
  tests/backends/test_gcom_cpu_pool.py
```

## Revision History

- 2026-11-26: Documented the canonical pilot/expansion, unchanged template and
  model contract, exact-byte Std ELF adapter, dedicated resource admission,
  immutable execution keys and automatic report finalization.

- 2026-11-27T15:14:10-08:00: Documented full-manifest continuation and the user-required GPU approval hold. CPU checks pass; GPU jobs have not been launched.
