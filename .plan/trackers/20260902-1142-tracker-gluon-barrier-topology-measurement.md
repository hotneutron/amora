# Tracker: Gluon Barrier-Topology Measurement

Status: Complete — Authoritative H100 Run Accepted
Plan: `.plan/20260902-1142-plan-gluon-barrier-topology-measurement.md`
Consumer: external Accorde repository

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
| r13 | 2026-09-02 14:30 -0700 | Completed current-tree closeout validation: 42 focused tests, 151 full-suite tests with one skip, scoped mypy, compileall, all 40 authoritative manifest hashes, and whitespace validation pass. |
| r12 | 2026-09-02 14:22 -0700 | Accepted the stabilized final H100 campaign: all eight points qualify; all four fixture, timing-quality, interval-order, and repeated-PC gates pass; and every pair is classified `supported`. Preserved the two earlier campaigns as immutable diagnostics and recorded same-model, same-`xhigh` delegation. |
| r11 | 2026-09-02 13:53 -0700 | Completed the authoritative post-correction full suite: 151 passed and one skipped in 250.05 seconds. |
| r10 | 2026-09-02 13:48 -0700 | Fixed a stale hard non-sync-SASS blocker found during final audit, preserving it as a diagnostic while keeping true cross-lane identity mismatches blocking. Added explicit regression coverage and reran focused/static validation. |
| r9 | 2026-09-02 13:44 -0700 | Recorded that the external recipe changed again during closeout, while the target remained on its non-final SHA. Canonical manifest validation passed for all 40 referenced files; no rerun was started and the immutable-input gate remains closed. |
| r8 | 2026-09-02 13:41 -0700 | Completed AM6 validation and handoff audit: 41 focused tests and 150 full-suite tests passed with one skip; scoped mypy, compileall, CLI help, digest validation, and whitespace validation passed. AM5 remains externally blocked and is not represented as accepted. |
| r7 | 2026-09-02 13:35 -0700 | Recorded and audited the second immutable H100 run. The revised hard A1 invariants pass in all four pairs, while real cross-lane identity drift blocks BT-EQ calibration and timing CV blocks held-out BT-CAUSAL. Stopped before another run because the external target and recipe changed and the target no longer matches the declared final SHA. |
| r6 | 2026-09-02 13:11 -0700 | Resumed AM5 after the user confirmed the final target and regenerated recipe. Revised A1 to use compiled useful-work/resource metadata plus executed GMMA, TMA-byte, and occupancy hard invariants; retained L2/DRAM as non-blocking mediation evidence without increasing the 5% threshold. |
| r5 | 2026-09-02 13:00 -0700 | Superseded the earlier readiness snapshot because the external recipe and target changed during preflight. Rechecked the current hashes and all eight aggregate lanes; A1 now fails on BT-EQ calibration, BT-EQ held-out, and BT-CAUSAL calibration, so AM5 remains stopped. |
| r4 | 2026-09-02 12:57 -0700 | Completed all-point A0 runtime validation and final post-preflight checks: all eight points matched frozen identity, metadata, axes, and numerical validity; 39 focused tests and 148 full-suite tests passed with one skip; scoped mypy, compileall, CLI help, and `git diff --check` passed. |
| r3 | 2026-09-02 12:49 -0700 | Reopened AM5 after the Accorde target and recipe appeared. Recorded passing A0 runtime and SourceCounters preflight, corrected Amora's eligible-warps metric-name recognition, and stopped before the full campaign because both BT-EQ pairs fail the frozen A1 L2-miss invariant. |
| r2 | 2026-09-02 12:39 -0700 | Completed AM1-AM4 against synthetic targets, recorded validation and handoff evidence, and closed the AM5 readiness gate because Accorde has no frozen topology recipe or completed protocol target. Replaced machine-local consumer paths outside verbatim user prompts to comply with `RULES.md`. |
| r1 | 2026-09-02 11:42 -0700 | Started the Amora-owned hardware-measurement workstream and recorded the Accorde target as an explicit external dependency. |

## Checklist

### Recipe

- [x] Add the generic `barrier_topology_release` mechanism.
- [x] Define required axes, causal nodes, and direct evidence.
- [x] Validate topology intervention groups and held-constant controls.
- [x] Add valid, missing-evidence, and drift-rejection tests.

### Repeated SourceCounters

- [x] Add a per-point PC-sampling repeat count.
- [x] Run independent SourceCounters collections.
- [x] Preserve every raw sample set and report hash.
- [x] Pool by exact cubin-relative SASS offset.
- [x] Emit dominant-region reproducibility status.

### Pair Controls

- [x] Compare identity, metadata, useful-operation fingerprints, and runtime
  axes within each topology pair.
- [x] Permit only declared synchronization/control binary differences.
- [x] Mark spills, occupancy changes, useful-work drift, and memory-route drift
  as `coupled_fixture`.
- [x] Limit A1 counter blockers to executed GMMA, TMA load bytes, and occupancy;
  retain L2 and DRAM differences as non-blocking mediation evidence.

### Reduction

- [x] Emit CUDA-event paired differences and confidence intervals.
- [x] Check device-interval partial ordering and instrumentation overhead.
- [x] Emit aggregate stall composition with a coherent family.
- [x] Emit PC localization with raw support counts.
- [x] Emit causal mediation without treating NCU duration as latency.

### Hardware Execution

- [x] Receive a frozen protocol-compliant target and recipe from Accorde.
- [x] Run BT-EQ points on H100.
- [x] Run BT-CAUSAL points on H100.
- [x] Re-run the complete matrix under the stabilized final contract.
- [x] Validate all cross-lane identities, metadata, axes, devices, and tools.

### Closeout

- [x] Run focused tests.
- [x] Run the full Amora suite.
- [x] Run mypy, compile checks, CLI help, and `git diff --check`.
- [x] Validate immutable synthetic artifact hashes.
- [x] Define compact evidence paths and hashes for the external Accorde consumer.
- [x] Validate the second H100 run's eight bundles, 24 NCU reports, six
  compact artifacts, frozen recipe, and mediation payload against its manifest.
- [x] Validate and accept the final H100 run digest and all compact artifacts.

## Execution Log

- 2026-09-02 11:42 -0700: Created the Amora-owned plan and tracker after the
  user clarified repository ownership.
- 2026-09-02 11:42 -0700: Confirmed the existing Amora implementation already
  separates CUDA-event timing, device intervals, aggregate NCU, and
  SourceCounters and validates cross-lane identities and runtime axes.
- 2026-09-02 11:42 -0700: Confirmed the new work is limited to a generic
  topology mechanism, repeated SourceCounters, pair-control reduction, and
  execution. Gluon compilation and P³ semantics remain outside Amora.
- 2026-09-02 12:09 -0700: Added `barrier_topology_release` with strict BT-EQ
  joint/split and BT-CAUSAL joint/pairwise groups, complete axes, seven timing
  repeats, three PC-sampling repeats, complete subject identity, frozen
  metadata, a reviewed semantic-region map, coherent per-warp-active stall
  metrics, and explicit invariant counter groups.
- 2026-09-02 12:09 -0700: Added independent SourceCounters execution with
  numbered report paths, per-repeat provenance and hashes, exact
  function-plus-cubin-offset pooling, and dominant-region qualification at 100
  structural samples with agreement in at least two profiles.
- 2026-09-02 12:09 -0700: Added pair-control scorecards and topology reduction.
  Controls reject undeclared axis, identity, metadata, SASS-class, spill,
  useful-work, memory-route, or occupancy drift as `coupled_fixture`; missing
  evidence remains `not_measured`. Timing uses paired process medians and a
  paired Student-t 95% confidence interval. Interval ordering is checked
  independently, and NCU duration remains diagnostic only.
- 2026-09-02 12:09 -0700: Added immutable compact handoff artifacts:
  `cuda_event_timing.csv`, `device_intervals.csv`, `aggregate_stalls.csv`,
  `pc_samples_by_offset.csv`, `fixture_control_scorecard.csv`, and
  `hardware_findings.json`; all are content-hashed in `manifest.json`.
- 2026-09-02 12:39 -0700: Added a CPU-only subprocess target at
  `tests/fixtures/barrier_topology_target.py` and verified both timing and
  device-interval protocol parsing for BT-CAUSAL pairwise topology. The
  end-to-end synthetic recipe test also exercises per-point PC-repeat
  overrides and validates every retained raw-report hash.
- 2026-09-02 12:09 -0700: Focused pre-change baseline passed 53 tests. Final
  focused measurement validation passed 62 tests. Full Amora validation passed
  148 tests with one skip in 288.43 seconds using
  `PATH=/usr/local/cuda-13.3/bin:$PATH`. Scoped mypy, bytecode compilation, CLI
  help, and `git diff --check` passed.
- 2026-09-02 12:09 -0700: Evaluated AM5 read-only. The external Accorde tracker
  still marks semantic recovery, controlled fixtures, protocol targets, and
  frozen-recipe handoff incomplete; repository search found no
  `barrier_topology_release` JSON recipe. AM5 was not started.
- 2026-09-02 12:49 -0700: Reopened AM5 readiness when the external target and
  frozen `gluon-barrier-topology-h100-v1` recipe appeared. The recipe has eight
  points, four topology pairs, seven timing repeats, three SourceCounters
  repeats, and complete calibration/held-out BT-EQ and BT-CAUSAL coverage.
- 2026-09-02 12:49 -0700: Fixed an Amora validation compatibility gap so the
  canonical NCU name `smsp__warps_eligible.avg.per_cycle_active` satisfies the
  eligible-warps physical-metric requirement; added a regression assertion.
- 2026-09-02 12:49 -0700: A0 runtime preflight passed on representative
  `eq-d2-k4096-joint` and `causal-gap64-pairwise` points. Both produced valid
  numerical results, all required identity/metadata/axis fields, exact frozen
  subject identities, and matching timing/device-interval identities. All 40
  frozen source, TTGIR, PTX, SASS, and cubin hashes also matched. Device
  intervals were retained as diagnostic: BT-EQ exceeded the 5% instrumentation
  overhead limit, and BT-CAUSAL reported a non-monotonic derived WGMMA interval.
- 2026-09-02 12:49 -0700: Real H100 aggregate profiling returned every requested
  metric for all eight points. BT-CAUSAL passed all seven pair invariants. Both
  BT-EQ pairs failed A1 only on `lts__t_sectors_op_read_lookup_miss.sum`: 45
  versus 51 sectors (11.76%) for calibration and 53 versus 49 sectors (7.55%)
  for held-out, above the frozen 5% threshold. GMMA count, TMA bytes, occupancy,
  total L2 reads, L2 hits, and DRAM bytes passed for those pairs.
- 2026-09-02 12:49 -0700: One real SourceCounters preflight retained
  `out/measurements/nvidia/preflight-gluon-barrier-20260902/eq-d2-k4096-joint.ncu-rep`
  with SHA-256
  `29601ea59dc793dffeb385768b35477ca4c99474fd79b0126546b47a060a9a58`.
  It contained 18 sampled PC rows, 48 raw samples, a 100% exact SASS join, and
  matching frozen identity and axes. No external Accorde file was written.
- 2026-09-02 12:49 -0700: Stopped before AM5 as required because A1 did not
  pass. The complete eight-point timing, interval, aggregate, and repeated-PC
  campaign was not launched.
- 2026-09-02 12:54 -0700: Final post-preflight validation passed: 39 focused
  backend tests; 148 full-suite tests with one skip in 234.21 seconds; scoped
  mypy on both changed NVIDIA measurement modules; `compileall`; NVIDIA measure
  CLI help; and `git diff --check`.
- 2026-09-02 12:57 -0700: Completed direct timing-protocol preflight for all
  eight frozen points. Every point exactly matched the recipe's subject
  identity, full subject metadata, and axes, and every numerical-validity flag
  was true. This confirms A0 passes across the matrix; A1 remains the sole
  blocker.
- 2026-09-02 13:00 -0700: Detected concurrent changes to the external target
  and recipe after the earlier representative interval and SourceCounters
  checks. Current SHA-256 values are
  `de1ac1981e46540e35603ee47bf99de92ea56a2321220c39814ea6334f8ca1a7`
  for the target and
  `ea59bba8431cd387454d033f72403401438936feea93894e542157be11a1553e`
  for the recipe. Revalidation found all 40 current artifact hashes valid and
  all eight timing and aggregate payloads matched current frozen identity,
  metadata, axes, and numerical-validity fields. The earlier raw PC report is
  retained but explicitly non-qualifying for the current snapshot.
- 2026-09-02 13:00 -0700: The current A1 aggregate check failed three pairs at
  the frozen 5% threshold: BT-EQ calibration DRAM reads were 8704 versus 18944
  bytes (54.05%); BT-EQ held-out L2 read misses were 63 versus 57 sectors
  (9.52%); and BT-CAUSAL calibration DRAM reads were 7680 versus 7168 bytes
  (6.67%). BT-CAUSAL held-out passed. AM5 remains stopped.
- 2026-09-02 13:11 -0700: The user confirmed target SHA-256
  `de1ac1981e46540e35603ee47bf99de92ea56a2321220c39814ea6334f8ca1a7`
  as final and supplied regenerated recipe SHA-256
  `8206eb43af0f97ad205d2694321c678fcb6bb8746763eed7b6820016e4141a3c`.
  The recipe loads with scalar axes and live NCU metrics.
- 2026-09-02 13:11 -0700: Revised A1 per the explicit user instruction. Hard
  invariants are compiled useful-operation fingerprints, request count/bytes,
  registers/shared memory/spills, executed GMMA count, TMA global-load bytes,
  and occupancy. L2 read/hit/miss and DRAM read traffic remain collected as
  non-blocking mediation evidence; the 5% hard-invariant threshold was not
  raised. Added validator and reducer regression coverage.
- 2026-09-02 13:22 -0700: The first immutable AM5 collection completed at
  `gluon-barrier-topology-h100-v1/am5-h100-gpu0-20260902-1312/`, relative to
  the requested output root,
  with run digest
  `a7743e2fc58b448b93e56ef280cc8ebccf7c8c44cb733551b9abcec9a9bd4c96`.
  Its raw data and hashes remain immutable. Its reduction exposed two remaining
  pre-revision couplings: diagnostic interval/PC status and non-sync SASS
  fingerprints still affected the A1 scorecard. Separated those from A1 while
  retaining them as diagnostics. A new run ID will produce the authoritative
  compact findings without rewriting this first run.
- 2026-09-02 13:33 -0700: The second immutable AM5 collection completed at
  `gluon-barrier-topology-h100-v1/am5-h100-gpu0-20260902-1326/`, relative to
  the requested output root.
  It used GPU 0 and the external Accorde working directory, retained eight point
  bundles and 24 independent SourceCounters reports, and emitted all six compact
  artifacts. Run digest:
  `c2dbb7a1e7b3292c6c31f64ce76112c5136b9fb1052940bbda79fd28bbe6fbb3`;
  normalized recipe digest:
  `30cd1409b9c21424e478fa33cd6d9f76d358bcef89fc8f78cd28729a87056834`.
- 2026-09-02 13:33 -0700: Every group passed the revised hard A1 counter
  invariants: executed GMMA count, TMA global-load bytes, and occupancy were
  exact within each pair. Compiled useful-work/resource metadata also passed.
  L2 read/hit/miss and DRAM-read values were collected and reported only as
  non-blocking mediation evidence.
- 2026-09-02 13:33 -0700: `eq-calibration` is `not_measured` because
  `eq-d2-k4096-split` SourceCounters repeat 2 changed `source_sha256`, and
  repeat 3 changed source, TTGIR, PTX, SASS, and cubin SHA-256 identities. This
  is a blocking A0/cross-lane failure, not a topology effect. `eq-held-out`
  passed A1 and classified `falsified`: paired CUDA-event difference
  `+1.413828773157937 us` (`+2.987096819313981%`), 95% confidence interval
  `[1.2229850901599697, 1.6046724561559045] us`.
- 2026-09-02 13:33 -0700: `causal-calibration` passed A1 and classified
  `supported`: paired CUDA-event difference `-0.4763432911464146 us`
  (`-1.5414215185593403%`), 95% confidence interval
  `[-0.8637476974017453, -0.08893888489108387] us`. `causal-held-out` passed
  A1 but is `not_measured` because `causal-gap128-joint` process CV is
  `2.2308695019926605%`, above the frozen 2% A2 limit.
- 2026-09-02 13:35 -0700: Verified every second-run manifest digest: eight
  bundles, 24 raw `.ncu-rep` files, six compact artifacts, the frozen recipe,
  and the mediation payload, 40 files total.
  The output occupies approximately 7.8 MiB and remains unchanged.
- 2026-09-02 13:35 -0700: A fresh read-only readiness check found target
  SHA-256 `1cc4b4e4012010ce282cf4435623c5caf71365f238fdc9d9bb14acdc5c230cb2`
  instead of the required
  `de1ac1981e46540e35603ee47bf99de92ea56a2321220c39814ea6334f8ca1a7`,
  and current recipe SHA-256
  `1efeb3b9d9e27def6b430c043754f1a9c56a0f70eeef8438c3d0de59f347ce2f`.
  No third run was started: the permitted rerun must use the same frozen
  contract. Amora did not write to Accorde.
- 2026-09-02 13:41 -0700: Final focused validation passed 41 backend tests.
  The full repository suite passed 150 tests with one skip in 236.27 seconds.
  Scoped mypy with `--follow-imports=skip` passed for the two changed NVIDIA
  measurement modules; direct import traversal separately exposed the known
  unchanged `amora/backends/nvidia/runner.py:78` optional-return annotation.
  Bytecode compilation of `amora/` and `tests/`, NVIDIA measurement CLI help,
  and `git diff --check` passed.
- 2026-09-02 13:48 -0700: Final source audit found and removed a stale
  `non_sync_sass_drift` A1 blocker, aligning the implementation with the r6
  policy and the second-run reduction. Cross-topology non-sync SASS fingerprints
  remain visible in `sass_diagnostics`; real per-point cross-lane identity
  mismatches still force `not_measured`. Added regression coverage for both
  behaviors. Focused backend validation now passes 42 tests, and scoped mypy,
  compileall, CLI help, and `git diff --check` pass after the correction.
- 2026-09-02 13:53 -0700: The post-correction full repository suite passed
  151 tests with one skip in 250.05 seconds. This is the authoritative final
  AM6 broad-test result.
- 2026-09-02 13:44 -0700: The closeout read-only hash check found the external
  target unchanged at its non-final SHA and the recipe changed again to
  `0bb03f7a04eb6474c7f8f3915e15b9c9d945a610ec55b31f2111f0517cdc2d8b`
  (mtime 13:42:19 -0700). The changing recipe provides an independent reason
  not to claim a same-contract rerun.
- 2026-09-02 14:22 -0700: Accepted the authoritative final campaign at
  `gluon-barrier-topology-h100-v1/am5-h100-gpu0-20260902-final/`. It used
  stabilized target SHA-256
  `1cc4b4e4012010ce282cf4435623c5caf71365f238fdc9d9bb14acdc5c230cb2`,
  recipe SHA-256
  `0bb03f7a04eb6474c7f8f3915e15b9c9d945a610ec55b31f2111f0517cdc2d8b`,
  and normalized recipe digest
  `75cff261af9c9327c5acbf8c1b236b308bb2cc1702849b569512bcb2b267f052`.
  The immutable run digest is
  `6e0cbb878988dd52b0618c165c9a61799b76d93cdf956ee66a1d61894a2c92d7`.
- 2026-09-02 14:22 -0700: All eight point bundles qualify. All four pair
  controls pass, including exact useful-operation/resource controls, hard GMMA,
  TMA-byte, and occupancy counters, and matching non-sync SASS digests. All
  timing CVs are below 2%; all interval-order checks are `supported`; all 24
  SourceCounters reports pool into qualifying exact-offset PC evidence with
  per-point support from 336 to 5229.
- 2026-09-02 14:22 -0700: Final CUDA-event pair differences are
  `-0.1323 us` and `-0.1615 us` for BT-EQ, and `+0.5447 us` and
  `+0.1705 us` for BT-CAUSAL pairwise minus joint. The hardware bundle
  establishes topology-dependent event order, not a BT-CAUSAL end-to-end
  speedup; NCU replay duration remains diagnostic only.
- 2026-09-02 14:22 -0700: The delegated TraeX work used thread
  `01a06372-e8dd-7d12-980a-540648c00dfb`, GPT-5.6-Sol, and `xhigh`
  reasoning effort, matching the parent workflow. The parent launched the
  authoritative final campaign after stabilization to avoid another edit/run
  race.
- 2026-09-02 14:30 -0700: Final current-tree validation passes 42 focused
  backend tests and 151 full-suite tests with one skip in 234.08 seconds.
  Scoped mypy, compileall, all 40 authoritative manifest hashes, and
  `git diff --check` pass.

## Current Gate

Complete. The final immutable AM5 campaign passes the frozen fixture, identity,
timing-quality, interval-order, and repeated-PC gates for all eight points. The
two earlier immutable campaigns remain diagnostic history and are superseded,
not altered. Amora has handed off the compact hardware evidence and digest;
Accorde owns the conclusion that barrier topology is valid portable execution
structure while its predicted end-to-end speedup is not confirmed.
