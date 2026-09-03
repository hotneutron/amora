# Request: Measure All-Corner Gluon Warp-Stall Cycles

Status: Requested
Date: 2026-09-03 12:33 -0700
Owner: Amora NVIDIA backend
Consumer: Accorde Gluon-enhanced P³ all-corner validation

## User Prompts, Verbatim

> don't update LANDMARKS.md so frequently.
> for each kernel, we should exercise all corners of gluon exposed dimensions, and match it with gluon enhanced PPP-IR. In other words, "Gluon launches = 1" seems useless to me, correct it or justify it.
> Also, to help root-cause the stall cycle breakdown, consider employ amora to give measure stall reasons.
> Also, consider how to predict the same "stall reason (cycles)" from gluon enhanced PPP-IR.

## Revision History

| Revision | Timestamp | Change |
|---|---|---|
| r1 | 2026-09-03 12:33 -0700 | Requested same-row absolute warp-stall-cycle reduction for exhaustive Gluon corners, three-repeat whole-launch aggregation, and repeated SourceCounters localization for boundary and residual points. |

## Purpose

Accorde will generate an exhaustive finite corner registry over every declared
Gluon compile-time and runtime dimension for nine canonical and five M12
application families. Every valid corner will have:

- one or more compiled Gluon launches;
- exact binary identity, grid, launch ordinal, and runtime axes;
- an independent CUDA-event timing record; and
- a Gluon-enhanced P³ prediction of active-warp and stall-reason cycles.

Amora should provide dimensionally matched measured hardware stall evidence for
those same launches. Launch count is topology metadata, not coverage; the
campaign size comes from the generated corner registry.

## Required Aggregate Stall Unit

Use one coherent
`smsp__warp_issue_stalled_<reason>_per_warp_active.ratio` family and retain from
the same whole launch row:

- `smsp__warps_active.sum`;
- `smsp__cycles_active.sum`;
- `smsp__warps_eligible.sum`;
- `selected`;
- `not_selected`; and
- every supported structural reason.

Derive:

```text
active_warp_cycles = smsp__warps_active.sum

reason_warp_stall_cycles[r]
    = active_warp_cycles
    * warp_issue_stalled_r_per_warp_active.ratio
```

If only `.pct` is available, divide by 100 before multiplication.

Both operands must come from:

- one whole launch row;
- one NCU report;
- one coherent metric family; and
- one exact point identity.

Never multiply a replay-lane stall percentage by CUDA-event duration or by NCU
duration.

## Required Views

Emit all three:

1. full issue-state vector including `selected` and `not_selected`;
2. structural vector excluding `selected` and `not_selected`; and
3. structural fractions normalized over only structural reasons.

Preserve:

- raw ratios;
- absolute reason warp-stall cycles;
- active warp cycles;
- active SMSP cycles;
- eligible warp cycles;
- missing reasons;
- metric names and units;
- selected whole-row index;
- profiler replay/pass count;
- cache and clock controls;
- GPU UUID;
- exact runtime axes; and
- tool versions.

## Repeat And Row Selection

For every qualified compiled launch corner:

- collect three independent aggregate NCU reports;
- select one whole launch row per report using `median_total_stall`;
- do not compose reason-wise maxima;
- validate identity and runtime axes across repeats; and
- reduce the three selected rows by choosing the median-total-stall whole row,
  while retaining all three raw selected vectors.

For a multi-launch operation, keep launch ordinals separate. Operation totals
are sums of independently selected launch records; never merge different
kernel launches into one synthetic row.

## Dimensional Integrity

Qualification requires:

- all available issue-state ratios sum to `1.0 ± 0.05`, or missing-state
  coverage is explicitly quantified;
- derived reason warp-stall cycles do not exceed active warp cycles outside the
  same tolerance;
- one metric family per record;
- same-row numerator and denominator;
- `selected` and `not_selected` retained for conservation;
- `selected` and `not_selected` excluded from the structural denominator; and
- PC sample counts never relabeled as cycles.

Use the raw unit:

```text
diagnostic_profiler_replay_warp_cycle
```

This makes explicit that the values are mechanism evidence from NCU replay, not
the CUDA-event latency oracle.

## SourceCounters Localization

Collect three independent SourceCounters profiles for:

- every minimum, alignment-boundary, first-tail, first-second-tile,
  first-wave, and first-second-wave sentinel;
- every compile-time corner;
- every warm/rotating pair selected by the consumer;
- every corner that fails latency or stall-cycle gates; and
- every corner where the dominant measured reason changes.

Retain:

- raw sample counts;
- support count and repeat support;
- function-relative SASS offset;
- SASS opcode and instruction;
- exact PC-to-SASS join status;
- raw report path and SHA-256; and
- identity/runtime-axis validation.

Do not convert PC sample counts to cycles.

## Target Contract

The external target will emit:

- exact composite operation identity;
- ordered per-launch identities;
- contract and corner IDs;
- compile-time and runtime axes;
- launch ordinal and kernel name;
- cache protocol;
- clock policy;
- grid and workgroup geometry;
- compiled registers, shared memory, warps, and spills; and
- target tool versions.

Amora must require the same fields across timing, aggregate, and SourceCounters
lanes. Any drift makes the point non-qualifying.

## Output Contract

Extend the existing command-measurement compact output with:

- `aggregate_stall_cycles.csv`;
- `aggregate_stall_cycle_repeats.csv`;
- `aggregate_stall_cycle_findings.json`;
- existing `pc_samples_by_offset.csv`; and
- manifest hashes.

Required aggregate columns:

- point ID;
- contract;
- corner ID;
- cache protocol;
- launch ordinal;
- kernel name;
- metric family;
- reason;
- reason ratio;
- active warp cycles;
- reason warp-stall cycles;
- structural denominator warp cycles;
- structural fraction;
- excluded-ready-state flag;
- selected row index;
- repeat index;
- dimensional-conservation status; and
- qualification status.

## Acceptance Gates

### A0: Capability

- Live GH100 metric resolution finds one coherent `per_warp_active` reason
  family.
- `smsp__warps_active.sum`, `smsp__cycles_active.sum`, and
  `smsp__warps_eligible.sum` resolve.

### A1: Whole-Row Integrity

- Each repeat uses one whole launch row.
- No reason-wise maxima composition.
- Same-row active-warp denominator.

### A2: Repeat Integrity

- Three independent reports per qualified launch.
- Same GPU, binary, kernel, grid, cache protocol, clock policy, axes, and tools.
- Reduced record is one observed whole row.

### A3: Dimensional Integrity

- Ratio and warp-cycle fields have explicit units.
- Conservation passes or missing-state coverage is explicit.
- `selected` and `not_selected` are excluded only from the structural
  denominator.

### A4: SourceCounters Integrity

- Three independent reports per selected launch.
- Raw support counts and SASS offsets retained.
- Exact PC-to-SASS joins for quote-worthy rows.

### A5: Latency Separation

- CUDA events remain the latency oracle.
- NCU duration remains `diagnostic_profiler_perturbation_only`.
- No stall ratio is multiplied by timing-lane elapsed cycles.

## Implementation Direction

Expected touch points:

- `amora/backends/nvidia/stall_metrics.py`;
- `amora/backends/nvidia/measurement.py`;
- `amora/backends/nvidia/mechanism_measurement.py`;
- compact artifact serializers; and
- focused backend and CLI tests.

Prefer extending the existing arbitrary-command measurement bundle. Do not add a
separate profiler framework.
