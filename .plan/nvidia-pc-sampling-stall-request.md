# Plan: per-PC warp-issue stall attribution

To: AMORA NVIDIA backend maintainer
Status: Implemented in this working tree
Depends on: `.plan/nvidia-stall-reason-coverage-request.md` (r2) — the unit mismatch
described there applies to whatever family this request collects, and should be settled
first
Blocks: per-phase schedule extraction in the accorde project

## Actionable Plan

1. Extend NCU command modeling so AMORA can collect and re-read source reports.
   - Update `amora/backends/nvidia/ncu.py`.
   - Add support for `--section`, `--set`, `--sampling-interval`,
     `--print-source`, `--force-overwrite`, and `--import`.
   - Keep live profiling targets last; imported reports omit live-profiling
     target-process options.

2. Add an NCU source-page extraction path.
   - Update `amora/backends/nvidia/ncu_run.py`.
   - Add `run_kernel_pc_sampling`, which builds the CUDA driver, profiles with
     `--section SourceCounters`, exports a `.ncu-rep`, then imports it with
     `--page source --print-source sass --csv`.
   - Add `parse_ncu_source_pc_sampling_csv`, which parses source-page rows into
     `PcStallSample` records using PC/address columns and
     `issue_stalled_<reason>` columns.
   - Preserve raw sample counts, sampling interval, report path, binary hash,
     source hash, and the exact NCU profile/import commands.

3. Preserve SASS offsets as the join key.
   - Update `amora/backends/nvidia/sass.py`.
   - Add `SassInstruction` and `parse_sass_instructions`.
   - Keep `parse_sass_opcodes` compatible with existing tuple-unpacking callers
     while retaining `offset`, `opcode`, `family`, `registers`, and instruction
     text for new consumers.

4. Add a best-effort probe helper for per-PC attribution.
   - Update `amora/probes/nvidia/baseline/_sources.py`.
   - Add `collect_pc_stall_attribution`.
   - Join sampled PCs to parsed SASS instructions by offset.
   - Emit records shaped as `{pc_offset, function, opcode, instruction, samples,
     stalls}` plus cubin hash and provenance.
   - Return `None` cleanly when NCU source counters, CUDA, or SASS tooling are
     unavailable.

5. Validate with GPU-independent unit tests.
   - Test NCU command arguments for live source collection and imported source
     reports.
   - Test source-page CSV parsing with PC offsets, `no_instruction` aliasing,
     and thousands-separated counts.
   - Test SASS parser offset retention.

## Execution Notes

Implemented files:

- `amora/backends/nvidia/ncu.py`
- `amora/backends/nvidia/ncu_run.py`
- `amora/backends/nvidia/sass.py`
- `amora/probes/nvidia/baseline/_sources.py`
- `tests/backends/test_ncu_run.py`
- `tests/backends/test_sass.py`

Local tool confirmation:

- `ncu --list-sections | rg -i 'source|pc|sampling|stall'` shows
  `SourceCounters`.
- `ncu --help | rg -n 'page|section|import|export|sampling|source|set'`
  confirms `--section`, `--sampling-interval`, `--export`, `--import`,
  `--page source`, and `--print-source`.

Verification:

- `PYTHONPATH=. pytest -q tests/backends/test_nvidia_cuda.py tests/backends/test_ncu_run.py tests/backends/test_sass.py tests/benchmarks/test_detailed.py tests/benchmarks/test_classification.py tests/benchmarks/test_materialize.py`
  passed with `45 passed, 2 skipped`.
- Full `PYTHONPATH=. pytest -q` passed with `94 passed, 3 skipped`.

## Original Request Context

## Revision History

| Revision | Change |
|---|---|
| r1 | Created. Requests source-correlated stall attribution — the per-PC form of the aggregate vector AMORA already collects — and names the two pieces of plumbing that do not exist yet. |

## What is being asked for

AMORA collects warp-issue stall reasons **per kernel** today
(`collect_stall_attribution` in `amora/probes/nvidia/baseline/_sources.py`). This
requests the **per-PC** form: for each sampled program counter, the distribution of
stall reasons observed at that PC, joined to the SASS instruction at that address.

Nsight Compute exposes this as source-correlated counters — the
`smsp__pcsamp_warps_issue_stalled_<reason>` family, surfaced through the source page
rather than the metric summary. The exact metric names and the collection switch should
be confirmed with `ncu --query-metrics` and `ncu --list-sections` on the target
architecture rather than taken from this document; see request 3 of the coverage
document.

## Why per-PC and not per-kernel

A per-kernel vector answers "what was this kernel mostly blocked on." It cannot answer
"which part of the kernel," and the consumer's question is entirely about parts.

The accorde model describes a kernel as a phase DAG — load, matrix-multiply, softmax,
epilogue — and asserts how those phases overlap in time and which one binds. Every one of
those assertions is currently a declaration in the model, not a measurement. A per-PC
stall distribution makes them measurable, because a PC maps to a SASS region and a SASS
region maps to a phase:

- **overlap** — at a sampled instant, the number of distinct phases holding a resident
  warp. One means sequential, all means fully parallel. This is the model's
  `execution_policy` measured rather than declared.
- **initiation interval** — per-warp PC recurrence at the loop head, which replaces a
  computed interval time with an observed one.
- **binding structure per phase** — the argmax of the stall distribution restricted to
  that phase's PC range, rather than one argmax for the whole kernel.
- **the unmodelled share, per phase** — the fraction blocked on scoreboard, barrier,
  i-cache or dispatch, which the analytical model has no term for. Aggregated over a
  kernel this is a single number; per phase it says *where* the model is blind.

## Two pieces of plumbing that do not exist

### 1. `NcuCommand` cannot request a section or a source page

`NcuCommand.argv()` in `amora/backends/nvidia/ncu.py` builds
`--target-processes`, `--csv`, `--page`, `--launch-count`, `--kernel-name`, `--metrics`,
`--export`. There is no `--section` / `--set`, and no path for the source-correlated
output — which is per-instruction rows, not the single summary row the current CSV
parsing assumes.

Source-correlated data is normally collected to a report and then extracted with
`ncu --import <report> --page source --csv` (confirm the exact invocation locally). That
is a second-stage read of an exported report; `NcuCommand` currently models only a
one-shot profile-and-parse.

### 2. The SASS parser discards instruction addresses

`parse_sass_opcodes` in `amora/backends/nvidia/sass.py` returns an opcode-family
histogram and an ordered list of `(family, [registers])`. The `/*0060*/`-style address
on each disassembly line is dropped.

Without the address there is no join key between a sampled PC and an instruction, so the
per-PC stall data cannot be attributed to anything. Retaining the offset alongside the
opcode family is the minimum change; it is additive and should not disturb existing
callers, which read the histogram and the ordered list.

## Suggested output shape

One record per sampled PC, or a compact table:

| field | meaning |
|---|---|
| `pc_offset` | instruction offset within the function, the join key to SASS |
| `function` | mangled kernel symbol, since a launch may span several |
| `opcode` | SASS opcode family at that offset |
| `samples` | total samples at this PC — needed to judge statistical weight |
| `stalls` | reason to sample-count map, same reason names as the aggregate vector |

Two properties worth preserving. Keep the reason names identical to `_STALL_LOGICALS`
so the aggregate and per-PC views are directly comparable, and keep raw sample **counts**
rather than pre-normalised percentages, so a consumer can choose its own denominator and
can tell a well-sampled PC from a thinly-sampled one.

## Caveats to record with the data

- PC sampling is statistical. A per-PC distribution with few samples is noise; the
  `samples` field is what makes that visible, which is why it should not be normalised
  away.
- The sampling period is configurable and interacts with kernel duration. Whatever period
  is used should be recorded in the result, not left to the tool default.
- PCs are per-function offsets. A launch spanning multiple kernels, or a cubin rebuilt
  between runs, invalidates the join. Recording the cubin hash alongside — `sha256_file`
  in `amora/backends/nvidia/disasm.py` already exists — would make a stale join
  detectable rather than silently wrong.
- Whether the per-PC family carries the same unit problem as the aggregate one
  (see the coverage request) should be checked before the data is used, not after.

## Scope

Requested: the collection path and the SASS join key. The phase mapping —
PC range to PPP phase — belongs to the consumer, not to AMORA, and is not requested
here.

Not requested: NVBit-based instrumentation. AMORA has an `nvbit.py` hook and the
consuming project has its own sampler, but NVBit instrumentation perturbs timing in a way
NCU's PC sampling does not, so this request is deliberately the NCU path.

## Related

- `amora/backends/nvidia/ncu.py` — `NcuCommand`, `list_metrics`
- `amora/backends/nvidia/sass.py` — `parse_sass_opcodes`, `extract_kernel_section`
- `amora/backends/nvidia/disasm.py` — `run_disassembler`, `sha256_file`
- `amora/probes/nvidia/baseline/_sources.py` — the aggregate collector this parallels
- `.plan/nvidia-stall-reason-coverage-request.md` — the aggregate-vector request
- accorde repository,
  `.plan/20260803-2229-plan-extract-scheduling-from-hardware-stall-traces.md` — Layer 1
  of that plan is exactly this request

## Direction, verbatim

> also add pc-sampling request for amora.
