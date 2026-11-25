# Reconstruct `portable_core` for Accorde H3 and Operation Gate v2

**Status:** Implementation plan, 2026-11-23. No generator changes, GPU collection,
model fitting, or new gate decisions in this revision.

## Decision

Keep core20/core24/core28 as a byte-compatible legacy lane. Rebuild the **new
expansion** around one typed operation and program registry instead of adding a
long list of `Nxx` cases. A case must say not just *which operation* it contains,
but how the values flow, which axes and domains it uses, and which access contract
it exercises. Expose separately selectable suites for **H3 semantics**, **H3
programs**, and **Operation Gate v2**, plus development-only mechanism controls
and legacy regressions.

The release criterion is coverage: the specified cases can be generated,
validated, and run against an independent reference. A portable correctness
pass never transfers an H100 timing label, predictor, or admission decision.
Amora owns workload definitions, inputs, oracles, and realization interfaces.
Accorde owns P³/lowering, study splits and exposure, native timing protocols,
models, and gate decisions. Neither package imports the other's live research
scripts at runtime.

## 1. Freeze what “cover” means

| Evidence scope | What the generator must cover | What the evidence does not grant |
|---|---|---|
| **Passed Operation Gate v2** | All **10 operations × 8 motif families × 3 canonical motifs × 3 square anchors** (128/256/512): **240 motifs / 720 case specs**. Reconstruct both the admitted and refused motifs, with exact operand ports, output selection, shape bindings, and two-step recurrence unroll. | The compiled-logistic gate caught **91/91** >6% motif failures and admitted **144/240** with zero failures (95% upper bound **2.60%**) under frozen H100/S9a-O18 conditions. This is not an unconditional predictor pass, and it does not cover 1024-square, new realizations, compilers, hardware, or compound OOD. |
| **Earlier operation evidence** | Inventory the original 384 motifs and the 18 S9a-O18 extension categories as reference/comparator inputs; resolve the four mechanism-adjacent Gate-v2 development categories as executable controls. Keep the older external cohorts addressable as regression evidence if their exact inputs are released. | The old cutoff was retired. The 264-motif study lacked failure positives; the later 480-motif study had 50% recall and was reclassified as Gate-v2 development. Neither is a promoted gate or fresh confirmation for this work. |
| **H3 single-semantic routes** | Seven scoped routes: affine state scan, embedding-bag sum, weighted five-point stencil, indexed many-to-one accumulation, fixed-4 segmented reduction, SiLU gated product, and scaled round/clip. Track the four sealed shapes/domains per route, as well as released development boundaries. Treat causal masked softmax, row variance normalization, and row top-k as a separate diagnostic panel. | Seven route-specific H100 passes are not a universal H3 model. Masked softmax failed development correctness at widths 64/96, variance normalization lacked full attribution at the maximum pilot, and top-k lacked an attributable label. |
| **H3 compound and topology routes** | Preserve the five component kinds (gather, contraction, reduction, broadcast, pointwise), the **25 directed kind pairs × 2 sizes**, the **32** whole-graph supervision cases, and source mappings for the **16** three-step confirmation cases, **16** joint graph/edge compound cases (224/448), **8** new-topology cases, and **8** large-topology cases (15–20 program nodes, 18–24 observed captured nodes). Count overlaps by identity, not by summing report totals. | Joint graph/edge confirmation passed at **1.9899% MdAPE / 4.8765% p95**; novel topology and scale passed separately. None establishes transfer to arbitrary new component kinds, all mixed-semantic graphs, or another GPU/compiler. |
| **Unified H3 closeout and later value pilot** | Inventory the **42 reported target cases** across three source shapes. Reconstruct them only after a case registry and input recipes are pinned. Include an explicit fork/join vs chain control, and an embedding/scatter stress panel with independently declared value and access axes. Keep the older ordered-ownership standard/swapped and depth/tail cases as separate execution descriptors, not as substitutes for graph topology. | The last explicit program-edge model passed **5/7** semantic gates (**2.2892% MdAPE / 19.9903% p95**); embedding and scatter tails remain unresolved. October closeout was scoped to the available context. The later fixed value/alignment pilot was valid negative (72 pairs, no 5% material effect under its contract); it did not reopen confirmation or establish that values never matter. |

**Source of truth:** [Evidence lock and verbatim request](20261123-1522-portable-core-h3-operation-evidence.json).
Use the lock's repo/ref/path tuples, not the apparent latest filename. The
pinned H3 scope reports do not include runnable case registries, and the unified
archive excludes the large graph corpora. The **exact H3 reference lane is
currently `blocked_missing_source` in this evidence lock**. This does not block
v2 schema work or new, separately identified controls.

**M0 source inventory:** For every required H3 panel, record expected source
rows and freeze a crosswalk keyed by `(panel, upstream_case_id)`. Each admitted
row needs a pinned, approved source bundle with the ordered graph and operand/
return ports, numeric shapes, domain and access bindings, input recipe and
oracle revision. Record repo/ref/path plus SHA-256 for each source artifact.
Where the registry itself is unavailable, keep a panel-level blocker with the
expected count, missing artifact and Accorde owner; do not invent case IDs.
When IDs become available, record per-row `blocked_missing_source` until the
source bundle is complete. Request source-only exports, not sealed timing
labels or model state. A synthetic substitute needs a new ID and cannot close
an exact-reference row. Track overlapping panel rows by source identity, not
by summing the report totals.

**Three separate reference receipts:** `spec_parity` checks the archived graph,
ports, operation contracts, shape/domain bindings and source case crosswalk;
`portable_correctness` checks a declared portable fixture against an independent
oracle; `historical_input_byte_parity` requires per-input byte digests against
approved originals or an approved replay with a verified matching RNG/runtime,
input order, dtype and scale. The pinned Gate-v2 fixture resets seed `20260925`
and draws `torch.randn` on CUDA in FP16, multiplying by `0.025` in ordered
`executable_program.external_shapes` order. Reusing its seed on CPU does not
establish byte parity. When originals or a verified replay are unavailable,
record `historical_input_byte_parity=not_verified`; use a new, versioned CPU
fixture and case ID, retaining the upstream spec crosswalk. Do not block
portable correctness on byte parity or call a new fixture an exact replay.

## 2. Operation and topology inventory

### Required operation contracts

| Workload group | Registry entries | Reconstruction requirements |
|---|---|---|
| Gate v2: indirect grouped reads | `weighted_embedding_bag_mean`, `multi_hash_embedding_sum`, `indexed_group_min`, `indexed_group_l2` | Exact index formula and auxiliary tensor roles (weights vs second table), fan-in 4, FP32 working arithmetic, active leading row region and explicit zero-padded output. Include L2 epsilon and nonzero weight denominator. Keep all four as hard/refused controls, not optional omissions. |
| Gate v2: row or pair operators | `rotary_pair_transform`, `row_logsumexp`, `cumulative_max_scan` | Paired column/angle indexing with even-width guard, stable logsumexp **broadcast back** to row shape, ordered inclusive prefix max. Do not substitute generic RoPE, scalar reduction output, or prefix sum. |
| Gate v2: local/threshold/quantization | `bilinear_resample`, `grouped_threshold_count`, `dequantize_bias_gelu` | Preserve the fixed four weights and clamped neighbors; fixed groups of four with strict `>0` and zero tail; exact scale/clamp/half-up rounding, bias placement, and tanh-GELU. |
| Gate v2: development controls | `weighted_embedding_bag_sum`, `embedding_bag_mean`, `dual_hash_gather_sum`, `indexed_group_max` | Source-pinned 96-motif/288-case panel as development-only reference. Preserve one-table dual hash vs two-table multi-hash and weighted sum vs normalized weighted mean. |
| H3: scans, reductions, indexing | `affine_state_scan`, `embedding_bag_sum`, fixed-4 `segment_sum`, H3 `scatter_add` | Affine carry composition (`s[i] = a[i] * s[i-1] + x[i]`), index/duplicate semantics and active-region padding, contiguous column groups, and the original fixed modular many-to-one map. Explicit-index or atomic scatter is a separate realization/contract, not an alias. |
| H3: neighborhood and transforms | `depthwise_weighted_stencil`, `silu_gated_product`, `scaled_round_clip` | Five-point weights and clamped boundaries; explicit second input to `silu(x)*aux`; scale, saturation and rounding order. Use the scoped semantic fixtures as authority for each port, not an assumed name match. |
| H3: diagnostic-only | `causal_masked_softmax`, `row_variance_normalize`, `row_topk_select` | Bind mask semantics for rectangular shapes, variance/epsilon and dtype, and duplicates/ties and output layout for the original selection contract. Preserve known disagreement cases; if clarifying a contract changes outputs, mint a new revision and keep the old native route blocked. |

The existing N22 SwiGLU, N25 sum-scan, N26 offset-based segment sum, N27
explicit-index scatter, and N28 signed zero-boundary 3×3 stencil are useful
controls, **not** exact aliases for the corresponding H3 operators. Keep their
formulas, shapes, seeds, and case hashes. Preserve N24 group ordering and empty
outputs. Reuse legacy primitives only after shape, dtype, reduction, and layout
parity is shown. In particular, source FP16 boundary conversions and FP32
arithmetic are not interchangeable with legacy BF16-rounded/FP32 math.

### Programs and panel selection

- **Operation-only reference:** dependent chain, independent merge, shared
  fork/join, residual bypass, nested diamond, multi-step branches, recurrence,
  and nested shared join. Adapt the frozen `executable_program` rows, not only
  their canonical motif labels. Round-trip every edge's operand slot (including
  auxiliary inputs), return/output, conditional path if present, and recurrence
  initial state and feedback. Do not gratuitously change the shape-preserving
  contraction, broadcast, reduction, gather, or pointwise variants around an op.
- **H3 directed-edge and compound panels:** reproduce the ordered pair panel
  at sizes 160 and 320; `A→B` and `B→A` are separate cases. Include matched
  operation-only, shape-only, baseline and compound cells;
  use equal-element wide/tall controls and check natural tails separately.
  Include cross ladder, asymmetric hourglass tail, dual residual weave and
  tri-fork cascade, then the independent large-topology families (deep residual
  spine, dual-trunk cross weave, nested diamond cascade, wide four-branch
  pyramid). Preserve original topology and size pairs where released; newly
  generated shapes/topologies are neither old matches nor gate passes.
- **H3 context controls:** start with embedding/grouped gather and scatter,
  then cover all seven semantics with disjoint donors. At fixed numeric shape,
  keep arithmetic and dtype constant while varying alignment/offset, physical row
  stride, index order/reuse, active unique rows and table working set, colliding destinations,
  or declared value amplitudes/patterns. Use matched changes and no-change
  controls; freeze distribution, corpus roles and direction of the comparisons.
  Keep working-set **bytes** in the spec; L2-relative ratios require a named
  architecture in Accorde. A logical access descriptor is not measured DRAM
  traffic or hidden dynamic cache state. Treat source value/alignment replicas of
  the November fixed pilot as diagnostic; a new context study needs new source
  contracts and authorization, not a fifth pilot attempt.

## 3. Target package and compatibility

- Leave all Python files directly under `base/` and `expanded/` unchanged:
  v1 snapshot verification hashes each directory's complete `*.py` set. Add
  `expanded/v2/spec.py` (validation and canonical encoding), `expanded/v2/ops/`
  (typed rules, execution and independent oracles), `expanded/v2/suites/` (pure
  generators and coverage contracts) and `expanded/v2/runner.py`. Wire them
  through the package-level API/CLI. Gate source adapters are offline and
  explicit; vendor only reviewed, source-only bindings or reproducible recipe
  seeds. No raw traces, model checkpoints, sealed labels or research state.
- **Separate identities:** `program_sha256` over canonical semantic graph
  (ordered operand roles, dtype/casts, operation attributes, ordered output ports,
  and state edges); `case_sha256` additionally binds numeric shape, declared
  input/access contract and input recipe revision; `realization_sha256` binds
  implementation, compiler/device/layout and execution protocol in a separate
  receipt. Record Accorde's original case/motif hash separately, never replace it
  with Amora's. Rename internal node IDs and reorder independent branches without
  changing logical identity; swapping noncommutative inputs or changing a
  reduction axis, output slot, seed, or access contract must change the appropriate
  identity. Graph identity ignores suite and role names so the same case cannot
  escape split checks by changing labels.
- **Type and execution contracts:** tensors include shape/dtype, axis and input
  domain, scalar constants, exact accumulation/rounding, pad/tail policy and
  named output shapes. Program DAG edges specify producer/consumer ports and
  data/state roles; recurrent edges use a bounded, explicit unroll into a valid
  execution DAG. Keep alias/materialization/layout declared separately from
  backend scheduling. `prepare` validates and binds actual indices, offsets,
  values and strides; `execute_prepared` does not silently repack or fall back.
  Backend adapters declare per-op/device support; NumPy is the first reference
  execution, not the oracle itself. Use independent scalar/high-precision
  references that apply the declared casts and accumulation rules. Keep JAX
  and other adapters optional until independently checked. Leave NVIDIA probes
  and `ppp_canonical` unchanged.
- **Separate oracle and realization contracts:** Give the portable oracle,
  historical oracle, and boundary-faithful realization distinct versions.
  For Gate v2, pin FP16 input and store points and FP32 working arithmetic;
  in particular, declare the FP16 reduction temporary before the separate
  broadcast launch. The archived Gate-v2 oracle keeps some intermediate values
  in FP32, so its final-output check at `atol=0.05, rtol=0.03` is archival
  evidence, not the portable release tolerance or proof of store-point fidelity.
  Include an adversarial reduction→broadcast witness where omitting the FP16
  store changes a later result. Verify declared boundary values separately from
  final outputs; use the boundary-faithful oracle for a realization that
  claims those stores, not an all-FP32 graph with a final half cast.
- **Torch CPU release matrix:** Freeze the required witnesses at M0. NumPy
  checks every required source-complete case. Torch CPU checks one canonical
  motif per Gate-v2 `(operation, relation_family)` at **all three** square
  anchors (**240 cases**); choose the lexicographically first upstream motif
  ID in each cell before execution, retaining both historically admitted and
  refused categories. Torch also checks every required source-complete H3 reference
  case, and the fork/join and matched-chain controls. Development-only and
  diagnostic-only rows are not required Torch checks unless explicitly listed
  in the M0 matrix. Use a separate CI lane with a pinned CPU Torch dependency
  and bounded shards. A missing dependency, witness, failed parity check, or
  incomplete required shard blocks the corresponding release, not a skip.
- **API/CLI:** keep current `families(tier)`, `generate_cases(tier, profile)`,
  `list/freeze --tier` and v1 `check` behavior. Add keyword-only `suite` selection
  and CLI `--suite {h3-semantics,h3-programs,operation-gate-v2,mechanism-controls}`
  for v2. Smoke selects bounded coverage witnesses; `reference` materializes
  exact historical case *specs*; `development` generates new opted-in controls.
  Do not call a generated reference spec “native-qualified.” Check by manifest
  version; do not upgrade v1 snapshots in place. A missing **required** backend
  in the v2 checker yields an explicit non-pass, not a pass based on NumPy alone.
- **Snapshot:** immutable, canonically serialized case manifests plus coverage
  and split receipts, seed policy, oracle version, dependency/source hashes,
  recipe checksums and upstream crosswalk. Version the new format separately; keep
  v1 code paths and source hashes intact so old frozen snapshots remain verifiable.
  Exclude volatile timestamps/absolute paths from canonical hashes. Materialize
  inputs lazily, one case at a time or in bounded shards. Review Apache-2.0
  provenance and notices for every adapted source; preserve the root MIT license
  and `SEMANTIC_FINGERPRINTS.json` unchanged.

**Port example:** import a historical `row_logsumexp` motif and map the target
node to a stable row reduction with explicitly broadcast FP16-boundary output;
keep the following nodes and the 128/256/512 shape tuples unchanged. Record
`spec_parity` against the archived executable graph and `portable_correctness`
against the separate oracle. With a new CPU input fixture, record
`historical_input_byte_parity=not_verified`, even if the seed is identical.
Carry `historical_admitted=true` only in the offline evidence crosswalk.
A new shape, fixture or backend does not inherit admission.

## 4. Implementation milestones and exit gates

| Milestone | Work | Exit criterion |
|---|---|---|
| **M0 — Coverage and compatibility freeze** | Freeze the per-panel/per-row source inventory and status, the spec/input/oracle receipt schemas, and the required Torch witness IDs and CPU dependency lock. Capture legacy manifests with the pinned current API/wheel. | Every required cell has an owner and verified sources or an explicit blocker. No report-only row is counted as exact. The Gate-v2 witness matrix has 240 entries, and all six legacy payload fingerprints and counts match. |
| **M1 — V2 case core + first vertical slice** | Implement schema, registry, immutable snapshots and lazy execution. Start with a Gate-v2 `row_logsumexp` motif, an `affine_state_scan` fixture, and fork/join vs matched chain. If H3 source is not complete, use new control IDs; do not mark the exact cells covered. | Generator, shape inference, port-aware ID, portable oracle, coverage row, `freeze/check`, and Accorde source-only import work for the slice. Reorder/rename parity passes; malformed axes, slots, casts, state and indices, unsupported tie policies, and unsupported backends fail closed. |
| **M2 — Operation coverage** | Port all ten Gate-v2 contracts and all eight motif families, including the four refusal categories, then the four development controls. Crosswalk original-384 and O18 reference source rows without importing training labels. | **240/240 motif IDs and 720/720 case specs** pass `spec_parity`. All 720 pass bounded NumPy portable correctness shards and all 240 frozen Torch witnesses pass; verify the FP16 reduction→broadcast boundary witness. Report historical input-byte parity separately, including unverified cases. No pooled pass on a failed category. |
| **M3 — H3 coverage** | Port the seven route semantics, ordered pairs and graph/topology panels as their approved source bundles become available, then stress and diagnostic controls. Use same-order/different-edge and reversed-edge witnesses to catch flattening. | **7/7** scoped routes and **25/25** ordered transition pairs at both sizes are representable and correctness-checked. Full exact H3 coverage requires all required source rows to pass the crosswalk and the required NumPy/Torch checks. Report blocked rows separately; distinguish logical and captured nodes. Known negatives do not become positives. |
| **M4 — Consumer and release qualification** | Export an installable Amora wheel and reference/spec/coverage manifests. Accorde implements a hash-pinned adapter and verifies the label-free crosswalk on a representative source-complete fixture per released panel, including mismatched/tampered cases. | Existing CLI/tests/legacy receipts still pass, a fresh checkout regenerates identical canonical manifests, the frozen NumPy/Torch matrix and wheel smoke pass, and Accorde verifies source-only preflight. Report exact claims per panel and keep blocked panels out of exact-coverage claims. No native timing needed. |
| **M5 — Optional new scientific qualification** | Only with separate approval, bind exact H100/`sm_90`, toolchain, binaries, launch DAG and assay; qualify compile/oracle, P³/timeline/attribution and timing quality on representative and max cases before any complete new panel. | Freeze study-specific splits, candidates and predictions before timing. Report correctness, label quality, coverage and aggregate/per-family/tail metrics; apply each original gate's estimand and thresholds only in its own scope. If changed backend/shape/protocol lacks evidence, report `not_evaluated` or `unsupported`, never reuse labels or inherit promotion. |

The implementation should land in reviewable increments: schema/slice, full
operation coverage, H3 panels, then Amora packaging and the separately versioned
Accorde adapter. The optional performance work is not a dependency for shipping
portable coverage. An unmaterializable reference row counts as blocked, not
skipped or covered. Only claim full reconstruction when all required exact
reference panels have approved source and complete crosswalks.

## 5. Validation details

1. **Determinism and compatibility:** all six legacy case payload SHA-256s stay
   unchanged (smoke 152/168/184; native specs 304/336/368). Repeat-run stable
   output for each v2 suite with clean processes and reordered source metadata;
   reject file tampering, unknown versions, and stale code/recipe hashes. Ensure
   import and catalogue discovery work without NumPy/Torch/JAX. Check install
   from a clean wheel with declared extras.
2. **Independent correctness:** validate exact index tensors, padding and
   discrete outputs; use per-operation, dtype-specific floating tolerances
   declared before testing, separately from archived tolerances. Include
   malformed inputs, singleton/empty cases only where legal, extreme values,
   ties/rounding at half-steps, divisibility and alignment guards. Test packed
   vs strided buffers, shared fan-out vs copied inputs, noncommutative merge
   order, and multi-output ordering. Verify every benchmark node and graph
   output against the applicable portable or boundary-faithful oracle. Track
   archived-oracle comparison and historical input-byte parity separately.
   Fail required coverage for any missing or failed NumPy/Torch matrix row;
   report optional backends separately.
3. **Exposure and axis audits:** published, rejected and development rows are
   historical/exposed, not new sealed observations. Split by operation contract,
   canonical graph or required topology family, numeric shape regime, and
   realization according to the declared axis. Keep all sizes of an operation-
   gate motif together; prevent identity/compiled-equivalence leakage and report
   duplicate support. Freeze new four-cell units together and retain refusal
   cases. IDs, hashes, names, gate actions and measured profiling data are not
   predictor inputs. Standalone op pass plus standalone shape pass does not imply
   joint transfer.
4. **Future hardware timing:** keep unprofiled graph latency, profiler occupancy,
   and attributed per-node time as different estimands. Verify input/binary hashes,
   actual layout, capture/replay, CV/spread/precision and repeat coherence under
   the exact protocol before consuming a label. For a Gate-v2-like study, retain
   its motif-macro >6% rule, sample adequacy, recall/admitted-failure bounds, and
   fail closed with no positives; for an H3-like route, retain its per-family/
   semantic, tail and coverage gates, not just pooled MdAPE. No fitting, tuning,
   selective recollection or reuse after opening confirmation. This plan does
   not authorize a new H100, MI300X, Ascend or TPU campaign.

**Baseline checked for this plan:** [current portable core tests](../tests/benchmarks/test_portable_core.py)
ran on CPU: **9 passed, 1 skipped** (Torch unavailable in the selected Python
environment). That checks the existing package, not the proposed expansion.

## Revision History

| Revision | Timestamp | Change |
|---|---|---|
| r1 | 2026-11-23 15:22 -0800 | Pinned the live Amora baseline, Accorde H3 route/closeout evidence, later bounded value pilot, and Gate-v2 external registry. Defined exact coverage, legacy preservation, ownership and phased qualification; rejected inherited performance promotion and bulk evidence import. |
| r2 | 2026-11-23 15:48 -0800 | Applied the four review findings using the pinned Gate-v2 fixture sources and H3 archive inventory: marked report-only H3 rows blocked, pinned input/oracle dependencies, separated spec parity, portable correctness and input-byte replay, distinguished FP16 store boundaries from the archived oracle, and froze the Torch matrix selection rule. |
