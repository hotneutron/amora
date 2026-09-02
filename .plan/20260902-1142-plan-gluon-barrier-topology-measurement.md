# Plan: Measure Gluon Barrier-Topology Effects on NVIDIA Hardware

Status: Complete — Authoritative H100 Run Accepted
Date: 2026-09-02 11:42 -0700
Owner: AMORA NVIDIA backend
Consumer: external Accorde repository
Tracker:
`.plan/trackers/20260902-1142-tracker-gluon-barrier-topology-measurement.md`

## User Prompts, Verbatim

> explain 16. from gluon code to actual scheduled operations in hardware, correlate that with gluon-enhanced PPP IR execution, detail a plan to systematically make sure they matches (using performance counters and sampling if needed).

> write an actionable plan and execute it with a tracker.

> amora's plan goes to ~/wk/amora/.plan, please

> create a sub-agent to work on amora's plan

> sub agent use the same model, same reasoning effort

> Resume AM5/AM6. The final Accorde target SHA is de1ac1981e46540e35603ee47bf99de92ea56a2321220c39814ea6334f8ca1a7 and the recipe at /home/cliu/wk/accorde/reports/ppp_ir_mlp/gluon_barrier_topology_h100_validation_v1/amora_recipe.json has been regenerated with scalar axes and live NCU metrics. Revise A1 as instructed: hard invariants are compiled useful-operation metadata, request count/bytes, registers/shared/spills, executed GMMA count, TMA load bytes, and occupancy. L2 hit/miss and DRAM bytes remain collected mediation evidence and must not be one-shot preflight blockers. Do not merely raise the threshold. Update tests and timestamped plan/tracker rationale, then execute the immutable H100 run with CUDA_VISIBLE_DEVICES=0, cwd /home/cliu/wk/accorde, output root /tmp/amora-gluon-barrier-topology, and a new run id if needed. Do not modify Accorde files or commit.

## Revision History

| Revision | Timestamp | Change |
|---|---|---|
| r13 | 2026-09-02 14:30 -0700 | Revalidated the accepted final tree after plan/tracker reconciliation: 42 focused tests and the complete 151-test suite pass with one skip; scoped mypy, compileall, all 40 authoritative manifest hashes, and `git diff --check` pass. |
| r12 | 2026-09-02 14:22 -0700 | Accepted the final immutable H100 campaign produced from the stabilized Accorde target and recipe. All eight points, four pair controls, timing-quality gates, interval-order gates, and repeated-PC gates qualify. This supersedes the two retained diagnostic campaigns without rewriting them. Recorded that the delegated TraeX agent used the same GPT-5.6-Sol model and `xhigh` reasoning effort as the parent workflow. |
| r11 | 2026-09-02 13:53 -0700 | Completed the post-correction broad rerun: the final tree passes 151 tests with one skip in 250.05 seconds. This supersedes the pre-correction 150-test result as the authoritative AM6 suite result. |
| r10 | 2026-09-02 13:48 -0700 | Final audit restored the intended A1 boundary after detecting a stale hard `non_sync_sass_drift` condition: cross-topology non-sync SASS digests are diagnostics, while actual cross-lane identity drift remains blocking. Added direct regression coverage for both cases and reran validation. |
| r9 | 2026-09-02 13:44 -0700 | Recorded continuing external recipe drift: a second closeout check found recipe SHA-256 `0bb03f7a04eb6474c7f8f3915e15b9c9d945a610ec55b31f2111f0517cdc2d8b`, different from both the launch and 13:35 snapshots, while the target remains at the non-final SHA. Canonical manifest validation covered all 40 referenced files, including the frozen recipe and mediation payload. |
| r8 | 2026-09-02 13:41 -0700 | Completed AM6 validation: 41 focused tests and the full 150-test suite passed with one skip; scoped mypy, compileall, CLI help, manifest-digest validation, and `git diff --check` passed. Kept AM5 open as externally blocked rather than accepting partial hardware evidence. |
| r7 | 2026-09-02 13:35 -0700 | Recorded the second immutable H100 collection and stopped before a third run. All hard A1 counter invariants pass, but one BT-EQ calibration point has blocking cross-lane SourceCounters identity drift and one held-out BT-CAUSAL point exceeds the 2% timing-CV gate. A fresh read-only check found that the external target and recipe changed after launch; the target no longer has the user-declared final SHA, so a same-contract rerun is not authorized. |
| r6 | 2026-09-02 13:11 -0700 | Resumed AM5 with the user-confirmed final target and regenerated live-metric recipe. Revised A1 without relaxing its threshold: compiled useful-operation metadata, request count/bytes, registers/shared memory/spills, executed GMMA count, TMA load bytes, and occupancy are hard invariants; L2 hit/miss and DRAM traffic remain collected non-blocking mediation evidence because a single replay is not a stable fixture-isolation oracle. |
| r5 | 2026-09-02 13:00 -0700 | Superseded the earlier readiness snapshot after the external recipe and target changed during preflight. The current snapshot passes schema, all 40 artifact hashes, and all-point timing/aggregate identity checks, but A1 fails for three pairs; AM5 remains blocked. |
| r4 | 2026-09-02 12:57 -0700 | Revalidated the final Amora tree after the eligible-warps compatibility correction and A0-A1 preflight: all eight points passed direct A0 runtime identity, metadata, axes, and numerical checks; 39 focused tests and the complete 148-test suite passed with one skip; scoped mypy, compileall, CLI help, and diff checks passed. |
| r3 | 2026-09-02 12:49 -0700 | Reopened AM5 after the external Accorde target and frozen recipe appeared. Runtime A0 passed on representative BT-EQ and BT-CAUSAL points, all 40 frozen artifact hashes matched, and aggregate/SourceCounters profiling executed on H100. AM5 remains blocked at A1 because both BT-EQ pairs exceeded the frozen 5% L2-read-miss invariant threshold. |
| r2 | 2026-09-02 12:39 -0700 | Executed AM1-AM4 with synthetic targets: added the generic topology recipe, repeated SourceCounters pooling, pair-control validation, topology reduction, compact immutable artifacts, and validation coverage. AM5 remains blocked because the external Accorde target and frozen recipe do not exist. Replaced machine-local consumer paths outside verbatim user prompts to comply with `RULES.md`. |
| r1 | 2026-09-02 11:42 -0700 | Created the Amora-owned hardware-measurement plan and tracker for Gluon barrier-topology validation; separated raw evidence ownership from Accorde's compiler and P³ interpretation work. |

## Objective

Measure whether controlled changes to Gluon barrier topology alter:

- unprofiled kernel latency;
- TMA-to-mbarrier and WGMMA completion intervals;
- coherent aggregate stall composition;
- the SASS locations where completion becomes exposed; and
- useful-work, memory-route, occupancy, or spill controls.

Amora owns raw measurement orchestration, validation, provenance, immutable
evidence, and hardware-side reduction. Accorde owns:

- the Gluon source fixtures and target command;
- compiler-to-P³ semantic correspondence;
- P³ execution and fixed-cycle traces; and
- the final causal comparison between measured and predicted behavior.

## Hypotheses

### BT-EQ: Topology-Equivalent Join

Compare:

```text
joint:
    A.complete ----\
                    joint barrier -> WGMMA.issue
    B.complete ----/

split:
    A.complete -> A barrier --\
                               max/join -> WGMMA.issue
    B.complete -> B barrier --/
```

Both release the consumer at `max(A.complete, B.complete)`. If useful work,
resource footprint, and memory behavior are controlled, they should have the
same portable dependency schedule.

A stable measured difference localized to synchronization/control instructions
would identify missing protocol service or compiler overhead. It would not
justify a fitted fixed wait.

### BT-CAUSAL: Schedule-Visible Release Topology

Compare one joint barrier for two producer-consumer pairs against pairwise
barriers. The pairwise form permits the fast consumer to issue before the slow
producer finishes. The joint form does not.

The hardware should exhibit the same partial-order distinction predicted by
Accorde. If it does not, the evidence must distinguish:

- an invalid or confounded fixture;
- a compiler-added ordering constraint;
- shared TMA, memory, barrier, scheduler, or WGMMA service; or
- an Accorde/P³ graph error.

## External Target Contract

Accorde must provide a command target for each point. Its final non-empty
stdout line must satisfy Amora's existing timing and interval protocols.

Required target capabilities:

1. compile and retain the exact Gluon source, TTGIR, PTX, SASS, and cubin;
2. run one selected topology variant;
3. emit `cuda_event_timing` in the timing lane;
4. emit `cuda_device_intervals` in the interval lane;
5. emit subject identity and metadata in every lane;
6. expose stable kernel-name filtering for NCU;
7. accept all measurement axes from the frozen recipe; and
8. execute the same runtime point under timing, interval, aggregate, and
   SourceCounters lanes.

Required subject identity:

- kernel name;
- source SHA-256;
- TTGIR SHA-256;
- PTX SHA-256;
- SASS SHA-256;
- cubin SHA-256.

Required subject metadata:

- registers per thread;
- shared-memory bytes;
- spill count;
- total warps and threads;
- compiled queue capacity;
- producer lead;
- useful-operation fingerprint;
- synchronization-instruction fingerprint.

Required measurement axes:

- panel: `BT-EQ` or `BT-CAUSAL`;
- topology: joint, split, or pairwise;
- M, N, K, tile shape, and grid;
- pipeline depth;
- producer-asymmetry level;
- cache protocol;
- CTA concurrency;
- launch batch size; and
- clock policy.

## Measurement Lanes

### Lane 1: CUDA-Event Latency

Use `run_command_cuda_events` through
`collect_command_measurement_bundle`.

- Run at least seven fresh target processes.
- Require warmup and batched launch samples inside every process.
- Retain every process sample, pooled median, p05, p95, and process CV.
- Require exact identity, metadata, axes, GPU UUID, cache protocol, and tool
  versions across repeats.
- Mark a row unstable when process CV exceeds 2%.
- Treat CUDA-event timing as the only latency oracle.

NCU duration remains `diagnostic_profiler_perturbation_only`.

### Lane 2: Device Intervals

Collect target-emitted diagnostic intervals:

- `tma_issue_to_barrier_release`;
- `barrier_release_to_consumer_issue`;
- `wgmma_issue_to_completion`;
- `stage_buffer_release`; and
- for BT-CAUSAL, independently named producer and consumer intervals.

Require:

- monotonic timestamps;
- consistent start, end, and duration values;
- documented clock domain;
- verified SASS bracketing;
- instruction-sequence equivalence declaration; and
- instrumentation overhead no greater than 5%.

Downgrade invalid interval evidence to `diagnostic`. Never substitute it for
the uninstrumented CUDA-event latency.

### Lane 3: Aggregate NCU

Resolve available metric names from the live NCU capability inventory. Use one
coherent `per_warp_active` family for every stall vector and retain one whole
launch row.

Requested logical evidence:

- long scoreboard;
- barrier;
- wait;
- MIO throttle;
- warpgroup arrive;
- selected and not-selected;
- GMMA instruction count;
- TMA global-load bytes;
- active and eligible warps;
- shared-memory occupancy limit;
- L2 read, hit, and miss sectors; and
- DRAM read bytes.

Preserve `selected` and `not_selected` in raw output but exclude both from the
structural-stall denominator. Missing metrics remain missing and must never be
filled with zero.

### Lane 4: SourceCounters

Collect three independent SourceCounters profiles for each stable point.

Retain:

- exact cubin-relative SASS offset;
- complete instruction and opcode;
- raw not-issued samples by reason;
- source line when available;
- report SHA-256;
- parser and PC-normalization provenance; and
- exact PC-to-SASS join status.

Accorde supplies a reviewed semantic-region map with:

- `tma_issue`;
- `mbarrier_poll_fast`;
- `mbarrier_poll_retry`;
- `wgmma_wait`;
- `wgmma_issue`; and
- unrelated prologue/epilogue.

Amora applies the map without inferring portable P³ semantics from NVIDIA
opcode names.

Quote a region-level result only when:

- pooled structural support is at least 100 samples; and
- the dominant region reproduces in at least two of three profiles.

Otherwise preserve it as `diagnostic`.

## Frozen Controls

Within every topology pair, validate:

- equal M/N/K, tile geometry, numerical result, and launch grid;
- equal pipeline depth, producer lead, request count, and request bytes;
- equal normalized TMA and HGMMA useful-operation fingerprints;
- equal actor, warp, and thread count;
- zero spills;
- equal registers per thread;
- equal total shared bytes and occupancy limit;
- equal cache protocol, clock policy, CTA load, and address mapping except for
  a declared controlled axis; and
- preserve and report cross-topology SASS fingerprints and declared difference
  classes as compiler-artifact diagnostics.

Cubins are expected to differ because barrier instructions differ. Binary
identity is checked per lane within one point, not across topology variants.

Mark a pair `coupled_fixture` if any control fails. Do not publish a
barrier-topology effect from that pair.

For A1, the hard counter invariants are executed GMMA count, TMA global-load
bytes, and shared-memory occupancy limit. These supplement the hard compiled
metadata invariants above: useful-operation fingerprint, request count and
bytes, registers, shared memory, and zero spills. L2 read/hit/miss sectors and
DRAM read bytes remain required collected mediation evidence, but their
single-profile differences do not classify a pair as `coupled_fixture`. Their
variability is reported without raising the existing 5% threshold. Diagnostic
interval, PC-support, or cross-topology compiler-fingerprint status is retained
under its own gate and does not make the A1 fixture-isolation scorecard fail.

## Recipe and Execution

Add a generic barrier-topology recipe family to
`amora/backends/nvidia/mechanism_measurement.py` rather than embedding Gluon or
P³ logic in Amora.

Suggested mechanism name:

```text
barrier_topology_release
```

Required causal nodes:

```text
footprint_occupancy
  -> producer_eligibility
  -> tma_issue
  -> memory_route
  -> tma_completion
  -> barrier_release
  -> consumer_issue
  -> wgmma_completion
  -> stage_buffer_release
  -> tma_issue
```

Add an Accorde-generated recipe under its report bundle and execute it with:

```bash
PYTHONPATH=. python -m amora nvidia measure \
  --recipe <accorde-recipe.json> \
  --out-root <immutable-run-root>
```

Run randomized points while preserving explicit calibration and held-out
labels:

- BT-EQ joint and split at depths 2, 3, and 4;
- BT-CAUSAL joint and pairwise at depths 2 and 3;
- short, transition, and long K;
- calibration and held-out producer-asymmetry levels;
- one-wave and multi-wave grids when footprint controls still match.

## Output Contract

Write one immutable run directory containing:

- one `bundle.json` per point;
- raw CUDA-event process samples;
- validated device intervals;
- coherent aggregate metrics and structural histogram;
- three PC-sampling summaries plus raw report hashes;
- cross-lane identity, metadata, axes, device, and tool-version checks;
- pair-control scorecards;
- causal-node and edge mediation;
- randomized execution order;
- complete commands and environment overrides; and
- a content-digested `manifest.json`.

Also emit compact consumer tables:

- `cuda_event_timing.csv`;
- `device_intervals.csv`;
- `aggregate_stalls.csv`;
- `pc_samples_by_offset.csv`;
- `fixture_control_scorecard.csv`; and
- `hardware_findings.json`.

Do not copy raw NCU reports or cubins into Accorde by default. Accorde consumes
compact tables plus immutable paths and hashes.

## Acceptance Gates

### A0: Target Readiness

- Every point emits all required identity, metadata, and axis fields.
- Every point produces numerically correct output.
- All lane commands identify the same point and expected binary.

### A1: Pair Isolation

- Every control in **Frozen Controls** passes.
- Compiled useful-operation metadata, request count/bytes, registers/shared
  memory/spills, executed GMMA count, TMA load bytes, and occupancy are
  invariant within repeat noise.
- L2 read/hit/miss sectors and DRAM bytes are retained as non-blocking
  mediation evidence and are not judged from one replay as fixture-isolation
  failures.
- No spill or occupancy change is present.

### A2: Timing Quality

- At least seven process repeats per point.
- Process CV at most 2%.
- No NCU duration is promoted as latency.

### A3: Interval Quality

- Zero timestamp-order violations.
- Instrumentation overhead at most 5%.
- Interval status is qualifying rather than diagnostic.

### A4: Counter Integrity

- One coherent stall metric family.
- One whole launch row.
- `selected` and `not_selected` excluded from structural denominator.
- Missing metrics explicitly recorded.

### A5: PC Localization

- Exact normalized PC-to-SASS join for every nonzero row.
- Raw support counts retained.
- At least 100 pooled structural samples for quote-worthy regions.
- Dominant localization reproduced in two of three profiles.

### A6: BT-EQ Response

- Paired CUDA-event difference lies within the larger of 2% or its paired 95%
  confidence interval; or
- a larger difference is localized to synchronization/control evidence while
  every useful-work, memory-route, and footprint control remains invariant.

The second result is `missing_sync_protocol_service`, not evidence for a fixed
wait constant.

### A7: BT-CAUSAL Response

- Device intervals show early fast-consumer release only for pairwise
  topology.
- The ordering reproduces across stable points.
- Any latency direction is reported with its confidence interval and control
  status.

### A8: Held-Out Evidence

- Held-out K, depth, asymmetry, and topology labels remain frozen.
- No held-out row is used to select counters, thresholds, or fixture controls.
- The compact bundle is sufficient for Accorde to score P³ direction and
  latency without rereading profiler-specific raw files.

## Actionable Work Packets

### AM1: Add Recipe Semantics

- Add `barrier_topology_release` to the generic mechanism registry.
- Define required axes, evidence nodes, direct-evidence bindings, and causal
  edges.
- Add controlled-group validation for topology interventions and held-constant
  axes.
- Add unit tests for valid and invalid recipes.

### AM2: Support Repeated PC Sampling

- Allow a recipe point to request a PC-sampling repeat count.
- Execute each SourceCounters collection independently.
- Preserve each raw support set and emit a pooled opcode/offset summary.
- Add reproducibility status based on dominant-region agreement.

### AM3: Validate Pair Controls

- Add a generic pair-control record for metadata, useful-operation
  fingerprints, and counter invariants.
- Treat allowed synchronization fingerprints as differences, not identity
  failures.
- Mark confounded pairs before mediation or topology conclusions.

### AM4: Reduce Topology Evidence

- Emit topology-pair timing confidence intervals.
- Emit interval-order checks.
- Emit region-localized PC support without mixing sample counts with aggregate
  percentages.
- Emit causal-edge mediation with `supported`, `falsified`, `not_measured`, or
  `coupled_fixture`.

### AM5: Execute Hardware Matrix

- Consume the frozen Accorde target recipe.
- Run all four independent evidence lanes on H100.
- Re-run failed or unstable timing points once under the same frozen contract.
- Do not silently change cache, clock, launch, or sampling settings.

### AM6: Validate and Hand Off

- Run focused measurement tests and the full Amora suite.
- Run type checking, compile checks, CLI help, and `git diff --check`.
- Validate every artifact digest.
- Hand compact evidence paths and hashes to Accorde.

## External Dependency

Accorde stabilized the final target
`tools/ppp_profiling/gluon_barrier_topology_probe.py` at SHA-256
`1cc4b4e4012010ce282cf4435623c5caf71365f238fdc9d9bb14acdc5c230cb2`
and its final recipe
`reports/ppp_ir_mlp/gluon_barrier_topology_h100_validation_v1/amora_recipe.json`
at SHA-256
`0bb03f7a04eb6474c7f8f3915e15b9c9d945a610ec55b31f2111f0517cdc2d8b`.
The accepted immutable run records normalized recipe digest
`75cff261af9c9327c5acbf8c1b236b308bb2cc1702849b569512bcb2b267f052`.

The earlier target and recipe hashes remain in revisions r3-r11 as diagnostic
history. They do not qualify as the final contract and were not overwritten.

## AM5 Execution State

Relative to the requested output root, the authoritative immutable collection
is retained at
`gluon-barrier-topology-h100-v1/am5-h100-gpu0-20260902-final/`.
It contains eight qualifying point bundles, 24 independent SourceCounters
reports, six compact consumer artifacts, its frozen recipe, and its mediation
payload. Its run digest is
`6e0cbb878988dd52b0618c165c9a61799b76d93cdf956ee66a1d61894a2c92d7`.

All four groups pass cross-lane identity and topology-pair controls. The hard
counter invariants for executed GMMA count, TMA global-load bytes, and
shared-memory occupancy match within every pair. The final non-sync SASS
digests also match within every pair, satisfying the frozen run's stricter
compiler-artifact control; current reduction still treats cross-topology
non-sync SASS differences as diagnostic while preserving actual cross-lane
identity drift as blocking. L2 read/hit/miss and DRAM traffic remain
non-blocking mediation evidence.

All eight timing rows pass the 2% process-CV gate. All four interval-order
checks are `supported`, all eight repeated-PC rows are `qualifying`, and all
four hardware pair classifications are `supported`. CUDA-event timing remains
the sole latency oracle; NCU replay duration remains diagnostic.

The two earlier immutable collections remain valid diagnostic records. They are
superseded for acceptance by this stabilized final campaign; no artifact,
threshold, or failed lane was rewritten or waived.

## AM6 Validation

- Focused backend validation after the final A1 boundary audit: 42 tests passed.
- Full repository suite after the final A1 boundary correction: 151 tests
  passed and one skipped in 250.05 seconds.
- Scoped mypy passed for the two changed NVIDIA measurement modules with
  `--follow-imports=skip`; normal import traversal also reports one pre-existing
  return annotation error in unchanged `amora/backends/nvidia/runner.py`.
- Bytecode compilation of `amora/` and `tests/`, NVIDIA measurement CLI help,
  and `git diff --check` passed.
- The second run's manifest validated all 40 referenced files: eight bundles,
  24 raw SourceCounters reports, six compact consumer artifacts, the frozen
  recipe, and the mediation payload.
- The final accepted run independently validates its eight point bundles, 24
  raw SourceCounters reports, six compact artifacts, frozen recipe, and
  mediation payload under run digest
  `6e0cbb878988dd52b0618c165c9a61799b76d93cdf956ee66a1d61894a2c92d7`.
- Final closeout reran 42 focused tests and the full suite: 151 passed and one
  skipped in 234.08 seconds. Scoped mypy, compileall, all 40 authoritative
  manifest hashes, and `git diff --check` pass on the accepted tree.
- Hardware collection was initially delegated to TraeX thread
  `01a06372-e8dd-7d12-980a-540648c00dfb` using the same GPT-5.6-Sol model and
  `xhigh` reasoning effort. The authoritative final campaign was launched by
  the parent workflow after the target and recipe stabilized, avoiding an
  edit/run race.

## Completion Criteria

Amora's part is complete when:

1. the generic topology recipe and repeated-PC protocol are tested;
2. all frozen Accorde points pass cross-lane identity and pair-control checks;
3. CUDA-event, interval, aggregate, and SourceCounters evidence is collected
   without unit mixing;
4. BT-EQ and BT-CAUSAL hardware responses are classified against the gates;
5. immutable evidence and compact consumer tables are validated; and
6. no result relies on NCU duration or a fitted barrier-wait constant.
