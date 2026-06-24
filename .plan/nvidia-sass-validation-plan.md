# NVIDIA SASS Validation Plan

## Scope

Add SASS-level validation to the NVIDIA probe pipeline so that timing-based
probes verify their kernels actually compiled to the intended instruction
stream before a measurement is promoted to a scalar. Today every kernel-bound
probe trusts that the source it timed produced the expected opcodes; this plan
closes that gap and wires the result into the existing fit-status / downgrade
rules.

This plan also lands the SASS-controlled register sweep as the first consumer
that *requires* SASS inspection (see "SASS-controlled register sweep").

## Motivation

Per the baseline and P1 methodologies, a latency/throughput scalar may only be
emitted when "opcode, counters, timing, and variance agree." Right now AMORA
checks timing and variance but not opcode. Concrete risks the validation closes:

- the compiler replaces the intended op (e.g. `FFMA` -> `FMUL`+`FADD`, or a
  dependent chain optimized away),
- unexpected memory ops appear inside a timed region,
- register pressure / spills (`STL`/`LDL`) silently change the measurement,
- vector-width or cache-operator changes in memory probes.

## Existing building blocks

- `amora/backends/nvidia/disasm.py` — `sha256_file`, `sha256_text`,
  `run_disassembler(disassembler, binary)`.
- `amora/backends/nvidia/build.py` — `CudaBuildConfig` can emit `-cubin`.
- `amora/backends/nvidia/runner.py` — compiles each probe `.cu` to a host
  executable and runs it; caches by source SHA-256.
- capability discovery already finds `nvdisasm` and `cuobjdump`.

## Key design decision: where validation runs

Two options were considered:

1. Per-probe validation inside each `.py`.
2. A shared validation step in `runner.py` that every kernel probe opts into.

**Decision: shared step in the runner, parameterized per probe.** Each probe
declares a small `SassExpectation` (required opcodes with min counts, forbidden
opcodes, dependency requirement) and the runner performs disassembly + checks
once, returning a structured `SassValidation` record alongside the timing
payload. This keeps the 12+ kernel probes DRY and guarantees uniform reporting.

## Disassembly path

The runner currently builds a *host executable* (kernel + driver) with `nvcc`.
For SASS we additionally build a `-cubin` (device-only) from the same `.cu`
using `CudaBuildConfig`, then disassemble it:

```
nvcc -arch sm_90 -cubin probe.cu -o probe.cubin
cuobjdump -sass probe.cubin        # or: nvdisasm -c probe.cubin
```

Prefer `cuobjdump -sass` (operates on the cubin/fatbin directly and is the most
portable across CUDA versions); fall back to `nvdisasm` when only an ELF cubin
is present. Cache the cubin and its disassembly text by source SHA-256 next to
the existing host-binary cache. Record `disassembly_hash` (already a field on
`ProbeIdentity`).

## SassExpectation contract

Each kernel probe provides (all optional except `kernel_symbol`):

- `kernel_symbol`: the `__global__` name to scope the SASS region (the drivers
  already use stable `extern "C"` names, e.g. `amora_baseline_fp32_dependent_chain`).
- `required_opcodes`: mapping opcode -> minimum count (e.g. `{"FFMA": 4000}` for
  the FP32 dependent chain at chain_length 4096).
- `forbidden_opcodes`: opcodes that invalidate the measurement inside the timed
  region (e.g. `{"LDL", "STL"}` to catch spills; `{"LDG", "STG"}` for the
  arithmetic probes that must stay register-resident).
- `require_dependency`: bool — whether destination-feeds-next-source must hold
  (checked heuristically by register reuse across consecutive target ops).

A minimal opcode counter is enough; full dataflow parsing is out of scope. The
dependency check is best-effort and only downgrades (never hard-rejects) when it
cannot be confirmed.

## SassValidation result + gating

The runner returns:

```text
SassValidation(
  validated: bool,
  disassembly_hash: str,
  opcode_histogram: dict[str,int],
  satisfied: list[str],        # required opcodes met
  violations: list[str],       # missing required or present forbidden
  dependency_confirmed: bool | None,
  reason: str | None,
)
```

Gating rules applied by each probe (consistent with the methodology):

- **Reject** (return `unsupported` with reason) when a `required_opcode` count
  is zero or a `forbidden_opcode` appears in the timed kernel — the measurement
  is meaningless.
- **Downgrade** (keep the value but lower fit_status one notch, e.g.
  `direct` -> `conditionally_identified`, and set `backend_interpretation.downgrade_reason`)
  when required counts are present but below the expected count, or when the
  dependency check cannot be confirmed.
- **Pass** unchanged when all required opcodes meet counts, no forbidden
  opcodes, and (if requested) dependency is confirmed.

The `SassValidation` dict is attached under
`backend_interpretation.sass_validation` (already a field on the schema) and the
opcode histogram under `raw_observation.values["sass"]`.

## Reporting

- `ProbeIdentity.disassembly_hash` populated for every kernel probe.
- The Markdown report's per-probe section gains a "SASS validation" block
  (satisfied/violations/dependency) rendered from `backend_interpretation.sass_validation`.
- A probe downgraded by SASS shows its `downgrade_reason` in the existing
  interpretation bullet, so the SUMMARY fit-status counts reflect it
  automatically.

## SASS-controlled register sweep

`register_file.register_bank_sweep` currently ships a CUDA proxy (sweeps
independent-accumulator width) and is intentionally marked `underconstrained`.
With SASS inspection available, add a variant that:

1. emits inline-PTX/`asm volatile` FFMA sequences with explicit source/dest
   register strides where the toolchain preserves them,
2. uses the disassembly to *confirm* the intended register numbers were kept
   (reject the run otherwise — this is the rejection rule from the P1 plan),
3. scores candidate bank mappings from throughput dips at each stride.

If SASS confirms register assignment, the probe may graduate from
`underconstrained` to `bounded`/`uniquely_identified`; otherwise it stays a
candidate curve. This is a follow-on to the validation infrastructure, not a
separate subsystem.

## Implementation steps

1. Add `SassExpectation` and `SassValidation` dataclasses (new
   `amora/backends/nvidia/sass.py`), plus a `parse_sass_opcodes(text)` helper
   and an opcode-histogram + region-scoping function.
2. Extend `runner.py` with `build_cubin()` and `validate_sass(source, expectation)`
   (cached by source hash), and an optional `expectation=` arg on `run_kernel`
   that returns the `SassValidation` next to the payload.
3. Add a per-probe `EXPECTATION` constant to each kernel probe and apply the
   gating rules in the probe `.py`.
4. Render the SASS block in `markdown_report.py`.
5. Add the SASS-controlled register sweep variant.

## Tests

- Unit: `parse_sass_opcodes` against captured `cuobjdump -sass` fixtures
  (committed text files) — no GPU needed.
- Unit: gating logic (reject/downgrade/pass) with synthetic `SassValidation`s.
- CUDA-gated: end-to-end validation on the FP32 dependent chain asserting
  `FFMA` present and `LDG/STG` absent.

## Risks / non-goals

- Full SASS dataflow analysis is out of scope; dependency checking is heuristic
  and only downgrades.
- SASS mnemonics differ by arch; keep `required/forbidden` opcode sets as opcode
  *prefixes* (e.g. `FFMA`, `LDS`) to stay robust across Volta..Hopper.
- Inline-PTX register control is fragile; the SASS-controlled register sweep
  must reject (not fabricate) when control is lost.
