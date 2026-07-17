# Plan: Benchmark Suite Architecture

## Summary

Extend AMORA with a benchmark layer alongside the canonical 36
microarchitecture probes.

The 36 probes remain the small, diagnostic, hardware-first suite used to
identify individual architectural behaviors. A benchmark definition evaluates a
versioned collection of static or generated kernel-and-shape cases and answers a
different question: how well do hardware measurements, GCoM simulation, and
future models agree across a representative shape space?

The first generated benchmark definition is an AMORA-owned PPP
canonical-kernel generator for H100.
Copy and adapt the canonical kernel definitions, deterministic shape programs,
case tags, replay contracts, and workload-generation logic into AMORA. The
generator becomes the source of truth for `kernel + tensor_shape` cases.

Do not treat a historical 2,500-case corpus as a named benchmark definition.
The case count is a materialization parameter. A run may request 2,500, 5,600,
or another budget; adding a canonical kernel or changing a shape program
creates a new generator/suite revision and therefore a new materialized case
set. Existing materializations remain reproducible by their recorded
generator revision, parameters, and content digest.

AMORA owns:

- canonical PPP kernel specifications and generated execution contracts;
- deterministic shape-generation programs, curated anchors, and tags;
- case materialization and its immutable manifest;
- hardware/simulator execution, evidence normalization, comparison, and
  reports.

The benchmark domain has two kinds of definitions:

- **Static suites:** pre-existing or hand-curated case inventories live
  directly at repository-root `benchmarks/<benchmark-suite>/`, for example
  `benchmarks/MLPerf/`.
- **Generated suites:** code that deterministically produces a case inventory
  lives at repository-root
  `benchmark_generators/<benchmark-id>/`, for example
  `benchmark_generators/ppp_canonical/`.

Both definition kinds materialize into the same immutable case-set manifest,
then use the same AMORA execution and reporting engine.

### Static Suite Contract

A static suite is a checked-in, already enumerated inventory. Its root
directory contains `suite.yaml` and case data or a declared import adapter:

```yaml
schema_version: 1
definition_kind: static_suite
benchmark_id: MLPerf
benchmark_revision: 1
cases:
  format: jsonl
  path: cases.jsonl
  sha256: <full sha256>
default_target:
  vendor: nvidia
grouping_dimensions: [model, scenario, precision]
```

`amora benchmarks materialize MLPerf` validates the declared immutable case
inventory, assigns the same canonical `case_key` format, and writes a
case-set manifest. It does not regenerate or reorder legacy cases. A revision
to the static suite inventory increments `benchmark_revision`; a selected subset
or backend target produces a distinct materialized manifest digest.

## Decision

Do not:

- add generated benchmark cases to `amora.probes.nvidia.baseline.PROBES`;
- mix benchmark-case rows into the existing `probes-<sku>.md` reports;
- identify cases only by fixture paths or display names.

Use three ownership boundaries:

```text
amora/
  benchmarking/
    __init__.py
    schema.py                   # case, materialization, run, comparison records
    registry.py                 # static-suite and generator discovery
    materialize.py              # definition -> immutable case-set manifest
    execution.py                # sharding, resume, retries, artifact writing
    contracts.py                # backend-neutral replay/trace/result contracts
  backends/
    nvidia/
      benchmark.py              # execute a generic benchmark replay contract
    gcom_cuda/
      benchmark.py              # trace/simulate a generic benchmark contract
  reports/
    benchmark_report.py         # suite-level aggregation and figures

benchmarks/
  MLPerf/
    suite.yaml                  # static suite declaration
    cases.jsonl                 # legacy/curated immutable inventory
    README.md
  <benchmark-suite>/
    ...

benchmark_generators/
  ppp_canonical/
    generator.py                # generator revision and materialization entrypoint
    cases.py                    # deterministic allocation and case materialization
    presets.py                  # optional h100_2500/h100_5600 named parameters
    kernels/
      gemm.py                   # copied/adapted kernel-specific generator/spec
      attention.py
      ...
    replay.py                   # generator-local replay and trace construction
    contracts.py                # measurement and component aggregation semantics
    templates/                  # source-generation templates, if they are needed
  <generator-id>/
    ...
```

`benchmarks/<benchmark-suite>/` is direct on purpose: it is the catalog for
static/legacy suite definitions, including `benchmarks/MLPerf/`. Do not put
those suites beneath an extra `suites/` or `legacy/` wrapper.

`benchmark_generators/<benchmark-id>/` is separate on purpose: a generator is
source code and a parameterization contract, not one benchmark inventory. Its
presets can produce many materialized case sets. This prevents a 2,500-case
and a 5,600-case PPP set from looking like separate source suites.

There is intentionally no catch-all `workloads/`, `assets/`, or backend
adapter directory beneath either catalog. `amora/benchmarking/` owns generic
materialization and execution; definition directories own only their semantic
inputs. Existing backend packages own CUDA, NCU, tracer, and simulator
execution. This prevents duplicate backend logic while keeping both static
suites and generators self-contained.

Generated CUDA sources, compiled binaries, traces, measured rows, and
simulator logs belong under ignored `out/benchmarks/`, never under source
packages.

Use `benchmark`, not `probe`, in public APIs and CLI names. A benchmark case
is an execution evaluation, not a microarchitecture probe.

Do not use `benchmarks/ppp-canonical-2500/` as the source definition:

- `ppp-canonical-2500` is neither a static suite nor an importable generator
  package name.
- `2500` is one case-set parameter, not the identity of its generator. Putting
  it in source would incorrectly imply a second generator for a 5,600-case
  materialization.
- a standard 2,500-case configuration may be a named preset
  (`h100_2500`) in `benchmark_generators/ppp_canonical/presets.py`, a CLI
  `--preset h100-2500`, and a
  human-readable manifest label. The immutable case-set digest remains the
  actual run identity.

## Benchmark Revision And Case-Set Manifest

Every static suite or generator has a stable `benchmark_id`. Every
materialized case set has an immutable manifest. The manifest, not a hardcoded
case count, is the comparison denominator.

```yaml
schema_version: 1
benchmark_id: ppp_canonical
benchmark_revision: 1
definition_kind: generator
kind: generated_benchmark_case_set
target:
  vendor: nvidia
  family: hopper
  hardware_sku: h100-80g
  arch_profile: sm_90_h100
case_key_version: 1
generator:
  module: benchmark_generators.ppp_canonical
  revision: 1
  git_commit: <full generator git commit>
  git_dirty: false
  source_sha256: <sha256 over generator package source files>
  generator_digest: <full sha256 of generator/kernel/shape/contract inputs>
  seed: 20260717
  allocation: balanced_per_kernel
  case_count_requested: 5600
  curated_classes: [corner, connecting, interpolation, ood]
  generated_class: sweep
kernels:
  - id: aligned_gemm_fp16
    revision: 1
  - id: flash_attention_fwd
    revision: 1
case_count_materialized: <actual generated count>
case_set_digest: <sha256 over canonical manifest rows and generation config>
```

The generator contract controls the materialization. The materialized manifest
records the exact selected kernels, per-kernel revisions, generation
parameters, and sorted case records. A consumer must be able to distinguish:

- the generator ID and revision;
- the generator Git commit, dirty state, and source SHA-256;
- the requested case count, seed, allocation policy, and kernel set;
- the immutable case-set digest;
- the subset selected for a run;
- the cases actually completed;
- cases unavailable for a backend.

Materialization must reject duplicate stable case keys, invalid shapes,
unsupported kernel/shape combinations, an empty allocation, and a requested
case count that cannot be satisfied exactly. It must write both requested and
actual counts. `--cases N` means exactly `N` cases; a distinct `--at-most N`
mode may be added later for exploratory runs. A partial backend run is valid,
but its report denominator is the manifest's materialized case count, with
selected/completed/comparable counts shown separately.

## Hardware Classification And Execution Order

Before any detailed GCoM comparison, run every materialized
`kernel + tensor_shape` case once on the target hardware through a lightweight
NCU collection recipe. This is a classification pass, not the detailed
profiling pass.

The classification recipe must collect only the basic execution facts needed
to rank cases, including:

- total executed instructions: `smsp__inst_executed.sum` or the resolved
  architecture-equivalent metric;
- kernel duration/cycle metadata required to validate a real execution;
- selected kernel name and launch identity;
- the NCU metric names, command, tool version, and collection mode.

The pass must not request the detailed stall-reason set, PC sampling,
source-level attribution, or the full comparison counter set. Store its
metrics under a named recipe such as `ncu_basic_v1`; do not call it
"unprofiled", because it is still an NCU collection run.

After all selected hardware cases have one valid `ncu_basic_v1` result:

1. Sort valid cases globally by `(total_instructions, case_key)`.
2. Split the ordered population into equal-count terciles:
   `small`, `medium`, and `large`.
3. Persist the rank, ordinal, thresholds, population size, metric name, and
   classification-run digest into an immutable classification overlay keyed by
   the case-set digest.
4. Cases with no valid instruction counter retain
   `size_rank=unclassified` with an explicit reason and are excluded from
   tercile thresholds.

Equal instruction counts at a boundary are resolved deterministically with
`case_key`; ranks must be stable when the same classification results and
manifest are reused. The report must distinguish `materialized`, `classified`,
and `unclassified` counts. Do not recompute ranks from an incomplete shard.

Detailed comparison then runs in rank order:

1. `small`: collect detailed hardware NCU evidence, trace, simulate with
   GCoM, and publish the first comparison report.
2. `medium`: repeat only after reviewing small-rank evidence, failures,
   semantic mismatches, and simulator throughput.
3. `large`: repeat only after the medium-rank review.

Every detailed run selects cases by the persisted classification-overlay rank,
not by rerunning an ad hoc instruction-count query. This makes results
resumable and keeps a later simulator-only rerun tied to the original hardware
classification.

The rank is an execution-order and reporting dimension, not a claim that
instruction count alone captures runtime, memory pressure, or simulation cost.
Reports must continue to stratify by kernel, shape class, and semantic status.

## Canonical Case Model

A materializer produces one `BenchmarkCase` record per generated benchmark
case:

```json
{
  "case_key": "ppp_canonical:r1:sm_90_h100:flash_attention_fwd:r1:B2_D96_H16_S512",
  "benchmark_id": "ppp_canonical",
  "benchmark_revision": 1,
  "definition_kind": "generator",
  "kernel_id": "flash_attention_fwd",
  "kernel_revision": 1,
  "shape": {"B": 2, "D": 96, "H": 16, "S": 512},
  "shape_key": "B2_D96_H16_S512",
  "shape_class": "sweep",
  "size_rank": "small",
  "size_rank_ordinal": 247,
  "axis_tags": ["B", "D", "H", "S"],
  "regime_tags": ["generated", "attention"],
  "execution_contract": {
    "measurement_semantics": "bounded_sampled_attention",
    "kernel_name_hw": "flash_attention_fwd_kernel",
    "replay_contract_revision": 1
  },
  "case_generation": {
    "generator_digest": "<full sha256>",
    "generator_git_commit": "<full commit>",
    "generator_source_sha256": "<full sha256>",
    "seed": 20260717
  },
  "classification_ref": {
    "recipe": "ncu_basic_v1",
    "instruction_metric": "smsp__inst_executed.sum",
    "total_instructions": 314665,
    "population_size": 5600,
    "result_digest": "<sha256>"
  }
}
```

`case_key` is canonical and must be generated from:

1. `benchmark_id`;
2. a case-key schema version;
3. kernel ID;
4. lexicographically ordered shape dimensions.

Do not use generated source paths or display names as the key. A future generator
that carries multi-kernel graphs must add an explicit `variant_id` or
`component_id` before case-key construction.

The PPP materializer should preserve:

- `shape_class`: `corner`, `connecting`, `interpolation`, `ood`, or
  `sweep`;
- `axis_tags`, `regime_tags`, and `tags`;
- persisted `size_rank`, ordinal, and classification-overlay provenance;
- `measurement_semantics`;
- kernel, replay-contract, and generator revisions;
- component records for fused kernels.

Do not assume every case has one CUDA launch. Cases such as
`rmsnorm_gemm_fp16` and `gelu_gemm_fp16` have components. The benchmark schema
must represent both:

- a top-level end-to-end case result, and
- optional component results with their own kernel/measurement identities.

## Backend Adapter Contract

The benchmark definition supplies deterministic execution contracts, but it does not
automatically make every case reproducible on every backend. Therefore
separate the benchmark definition from per-backend executability.

Each adapter returns a `BenchmarkCaseResult`:

```json
{
  "case_key": "...",
  "backend": "nvidia_cuda",
  "run_status": "measured",
  "availability": "available",
  "measurement": {
    "value": 17354.42,
    "unit": "cycles",
    "semantic": "kernel_cycle_label"
  },
  "metrics": {},
  "components": [],
  "provenance": {}
}
```

Use these states:

- `measured`: hardware replay completed with valid evidence;
- `simulated`: simulator completed with a valid core execution result;
- `unsupported`: adapter intentionally cannot execute the case;
- `missing_artifact`: suite case exists but lacks required executable, trace,
  or source artifact;
- `failed`: attempted execution did not complete;
- `skipped`: excluded by a user-selected filter/shard.

The first PPP adapters have two deliberate modes:

1. **Hardware replay:** build or invoke the AMORA-owned PPP execution contract,
   then collect hardware timing/NCU evidence into `nvidia_cuda` case results.
2. **Trace/simulate:** execute or trace a case only when an explicit
   generator replay adapter exists for that kernel. The adapter must record the
   executable/source hash, command, selected hardware kernel regex, launch
   semantics, and any bounded-workload parameters.

Do not compare results from different kernel or replay-contract revisions as
though they were identical. A case run always carries the generator, kernel,
and replay-contract revisions that produced it.

The initial `gcom_cuda` adapter should consume an explicit trace artifact for
each case. It must not claim that all PPP cases are simulatable merely because
they materialize successfully. Case status will initially be sparse until
generator-specific trace generation/replay adapters land.

## CLI And Execution Model

Add a separate CLI namespace:

```text
amora benchmarks list
amora benchmarks inspect ppp_canonical
amora benchmarks inspect MLPerf
amora benchmarks materialize ppp_canonical \
  --target nvidia:h100-80g \
  --cases 5600 --seed 20260717 \
  --manifest out/benchmarks/ppp_canonical/r1/<case-set-digest>/manifest.json
amora benchmarks materialize MLPerf \
  --manifest out/benchmarks/MLPerf/r1/<case-set-digest>/manifest.json
amora benchmarks classify ppp_canonical \
  --backend nvidia_cuda --recipe ncu-basic-v1 \
  --manifest <case-set-manifest.json> \
  --out out/benchmarks/ppp_canonical/r1/<case-set-digest>/classification/nvidia_cuda/hopper/h100-80g/<run_id>
amora benchmarks run ppp_canonical \
  --backend nvidia_cuda --mode detailed-ncu \
  --manifest <case-set-manifest.json> \
  --size-rank small \
  --out out/benchmarks/ppp_canonical/r1/<case-set-digest>/runs/nvidia_cuda/hopper/h100-80g/<run_id>
amora benchmarks run ppp_canonical \
  --backend gcom_cuda --manifest <case-set-manifest.json> \
  --size-rank small \
  --shard 3/32 --resume \
  --out out/benchmarks/ppp_canonical/r1/<case-set-digest>/runs/gcom_cuda/hopper/gcom_h100/<run_id>
amora benchmarks compare \
  --manifest <case-set-manifest.json> \
  --real <hardware-run.jsonl> \
  --sim <gcom-run.jsonl> \
  --out-dir reports/benchmarks/ppp_canonical/r1/<case-set-digest>/comparisons/nvidia_cuda-h100-80g__gcom_cuda-gcom_h100
```

Execution requirements:

- deterministic case ordering by `case_key`;
- require a completed persisted hardware classification before `--size-rank`;
- detailed rank progression is `small`, then `medium`, then `large`; each
  next rank requires an explicit review marker in the parent comparison
  metadata;
- explicit sharding by stable hash or ordinal, recorded in run metadata;
- idempotent `--resume` keyed by `(case_key, backend, run contract hash)`;
- bounded retry policy with persisted failures;
- per-case timeout plus suite-wide concurrency limits;
- raw trace, NCU, and simulator artifacts under ignored `out/`, not in
  committed reports.

The first implementation should support `--limit`, `--kernel`,
`--shape-class`, `--regime-tag`, `--case-key`, `--size-rank`, and `--shard`
before attempting full-corpus GCoM simulation.

## Result Storage

Keep case-level results out of the generic `ProbeResult` schema. A benchmark
case has a different identity, potentially multiple components, and a much
larger cardinality.

Use immutable JSONL shards plus a compact run manifest:

```text
out/
  benchmarks/
    ppp_canonical/
      r1/
        <case-set-digest>/
          manifest.json
          classification/
            nvidia_cuda/
              hopper/
                h100-80g/
                  <run_id>/
                    run.json
                    cases-00000.jsonl
                    rank_assignment.json
          build-cache/
            <compile-contract-digest>/
          runs/
            nvidia_cuda/
              hopper/
                h100-80g/
                  <run_id>/
                    run.json
                    cases-00000.jsonl
                    failures.jsonl
                    artifacts/
            gcom_cuda/
              hopper/
                gcom_h100/
                  <run_id>/
                    run.json
                    cases-00000.jsonl
                    failures.jsonl
                    traces/
                    simulator-logs/
```

`run.json` must include:

- materialized case-set digest, benchmark revision, and generator/kernel
  revisions when applicable;
- generator Git commit, dirty state, and source SHA-256 for generated
  definitions;
- AMORA revision;
- backend/SKU/profile;
- exact filter and shard definition;
- NCU collection recipe and resolved metric names when the run is hardware
  classification or detailed profiling;
- classification-run digest and persisted `size_rank` population when the run
  selects ranked cases;
- tool/device/config/version metadata;
- executable, tracer, and simulator hashes where relevant;
- counts by status and failure reason;
- start/end timestamps;
- parent run IDs for incremental/resumed runs.

The materialized case index and raw case-level result shards are generated
artifacts. Keep them untracked by default. Commit only suite registry code,
small fixture manifests specifically created for tests, report renderers, and
curated summary reports when requested.

## Comparison Semantics

Comparison must occur per matching `case_key` and component identity. Do not
compare a simulator cycle estimate against a hardware measurement with a different
measurement semantic.

Every comparison row includes:

- `case_key`, `kernel_id`, shape, shape class, and tags;
- hardware and simulator values and units;
- measurement semantic compatibility;
- absolute error, absolute percentage error, and signed log-ratio where valid;
- hardware/simulator provenance;
- `comparison_status`:
  `comparable`, `semantic_mismatch`, `missing_hw`, `missing_sim`,
  `unsupported`, `failed`, or `excluded`;
- optional bottleneck/resource and stall-histogram comparison fields.

Use `cycles` only when both sides represent the same kernel-level convention.
The benchmark replay contract must carry its measurement semantics through
hardware and simulator runs. If a GCoM trace is bounded/sampled differently,
mark the row `semantic_mismatch`; do not compute a misleading error.

Fused benchmark cases must have:

- an end-to-end row only when hardware and simulation share the same
  aggregation contract; and
- component rows for valid component-to-component analysis.

## Reporting

Do not render every case row as the primary Markdown surface. A 5,600-case
materialization should not produce a 5,600-row Markdown page. Produce a
layered benchmark report:

```text
reports/
  benchmarks/
    README.md
    ppp_canonical/
      r1/
        <case-set-digest>/
          README.md                # index of recorded runs and comparisons
          manifest.json            # copied compact case-set identity/provenance
          comparisons/
            nvidia_cuda-h100-80g__gcom_cuda-gcom_h100/
              SUMMARY.md
              comparison.json
              figures/
                coverage-by-kernel.svg
                error-distribution.svg
                error-by-kernel.svg
                error-by-shape-class.svg
                hardware-vs-sim-scatter.svg
                worst-cases.csv
              cases/
                comparison.jsonl.gz  # generated/untracked by default
```

`SUMMARY.md` should answer, in this order:

1. **What ran:** benchmark/generator/kernel revisions, materialized case-set digest,
   generation parameters, hardware/simulator SKUs, run IDs, number of cases
   materialized, classified, unclassified, selected, comparable, missing,
   unsupported, and failed. State the NCU basic classification recipe and
   instruction-count metric.
2. **Can values be trusted:** semantic-match coverage and count of
   `semantic_mismatch` rows.
3. **How accurate:** MdAPE, p50/p90/p95/p99 APE, geometric mean ratio,
   signed-bias median, and R-squared only when the matched population is large
   enough and its assumptions are stated.
4. **Where accuracy changes:** first by persisted `small`, `medium`, and
   `large` instruction-count rank, then by kernel and shape class, including
   `corner`, `connecting`, `interpolation`, `ood`, and `sweep`.
5. **What fails:** top worst cases with shapes, semantics, and links into
   raw-case artifacts; do not rank cases with invalid/missing comparisons.
6. **Why:** bottleneck/resource or stall-reason rollups only for case pairs
   that have compatible metrics.

Required figures:

- coverage stacked bars by kernel and result status;
- classification coverage and instruction-count distribution, with tercile
  boundaries and unclassified count;
- CDF/ECDF of APE, with missing/unsupported excluded but counted separately;
- small/medium/large error distribution and cumulative comparison coverage;
- per-kernel error distribution (box/violin or percentile bars);
- shape-class error distribution;
- log-log hardware-vs-simulator scatter with equality line, separated by
  kernel or rendered as small multiples;
- top-k worst comparable cases table, not a chart alone.

The detailed case table belongs in `comparison.json` and compressed
JSONL/Parquet. It should be filterable by kernel, kernel revision, shape class,
regime tag, status, and error rank. Markdown should link to those artifacts
and show only aggregate and top-k views.

Unlike probe reports, benchmark reports are benchmark-first and case-manifest
first: `reports/benchmarks/<benchmark_id>/<benchmark_revision>/<case-set-digest>/`.
The comparison target appears beneath `comparisons/`, because one materialized
case set may be run on several hardware and simulator targets. This prevents
microprobe identity and benchmark-case identity from colliding. The top-level
`reports/README.md` should link both report families.

## Future Generators

The benchmark engine must not encode PPP-specific names outside
`benchmark_generators/ppp_canonical/`. A future generator must
provide:

- `benchmark_id`, `benchmark_revision`, and generator revision;
- case materializer;
- target/vendor/architecture metadata;
- benchmark replay/trace adapter capabilities;
- required measurement semantics;
- grouping dimensions for reports;
- license/provenance information for imported assets.

Examples:

- a Blackwell materialization of the PPP generator;
- a CUTLASS or Triton benchmark definition;
- MLPerf-derived kernels with explicitly licensed/reproducible inputs;
- ROCm/HIP suites;
- graph-level suites that identify components and aggregation rules.

Each future generator should add a local generator package and optional
checked-in small test manifest. Backend execution extensions belong beside the
backend, not in the generator package. A new generator must not add a
special-case branch to the generic execution or report renderer.

## Phased Implementation

### Phase 1: PPP Generator And Materialization

- Copy and adapt the PPP canonical-kernel, shape-generation, and replay
  contracts into `benchmark_generators/ppp_canonical/`.
- Add benchmark schema, registry, deterministic materializer, and case-set
  manifest writer.
- Materialize parameterized H100 case sets and preserve kernel/shape-class
  tags.
- Add `amora benchmarks list`, `inspect`, and `materialize`.

Acceptance:

- identical generator revision and parameters yield the same sorted case set
  and digest;
- changing the requested case count, seed, kernel revision, or generator
  revision yields a different manifest identity;
- `--cases N` materializes exactly `N` cases or fails with a capacity/allocation
  diagnostic;
- a materialized case set has unique case keys and an explicit actual count;
- no generated sources, binaries, traces, or rows are written beneath
  `amora/`.

### Phase 2: Hardware Replay Reference

- Build/replay every selected PPP case through AMORA-owned contracts on the
  target hardware using the lightweight `ncu_basic_v1` classification recipe.
- Persist total instruction count, resolved metric names, kernel/launch
  identity, generator Git commit, generator source SHA-256, generated-source
  hash, executable hash, and measurement semantics.
- Freeze global small/medium/large tercile assignments in an immutable
  classification overlay.
- Render classification coverage and instruction-distribution reports.

Acceptance:

- every selected hardware result maps to exactly one `case_key`;
- every materialized case is classified or explicitly unclassified;
- all valid classified cases have a deterministic small/medium/large rank;
- the report records the NCU metric/recipe and rank boundaries;
- detailed GCoM comparison cannot start without the frozen classification
  artifact.

### Phase 3: Small-Rank Detailed Comparison

- Define one benchmark replay/trace contract per supported PPP kernel.
- Run detailed NCU profiling, trace generation, and GCoM simulation only for
  persisted `size_rank=small` cases.
- Collect detailed counters and stall-reason histograms on both sides where
  the measurement contract supports them.
- Record trace/config/binary version contracts per case.

Acceptance:

- every attempted small-rank simulation has a valid trace contract and raw
  simulator artifact;
- unsupported kernels are explicit and do not block other shards;
- a resumed shard does not rerun completed matching cases;
- the small-rank report includes coverage, scalar/counter comparisons, and
  stall-reason comparisons.

### Phase 4: Medium And Large Rank Progression

- Review the small-rank report before enabling `medium`; persist an explicit
  review marker with the accepted run IDs, known failures, and semantic
  decisions.
- Run detailed hardware/GCoM comparison for `medium`, review it, then repeat
  for `large`.
- Keep rank populations and previous-rank results immutable; later ranks add
  coverage rather than replacing earlier evidence.

Acceptance:

- detailed execution order is small -> medium -> large;
- each transition has a persisted review marker;
- aggregate metrics are never calculated over missing/semantic-mismatch rows;
- each figure states rank population and exclusions;
- worst-case links resolve to raw-case evidence.

### Phase 5: Full Reporting And Generalization

- Implement the full case/rank/kernel/shape-class aggregation and required
  figures.
- Add a benchmark index to `reports/README.md`.
- Keep figures as report assets and raw rows in generated artifacts.
- Add a fixture-backed unit-test suite with a small synthetic local benchmark
  package.
- Document the generator-package interface and generator revision policy.
- Add a direct static suite under `benchmarks/<benchmark-suite>/` and a second
  generator without changing generic runner/report code.

Acceptance:

- the static suite and both generators materialize, run, and report through
  the same public commands;
- generator-specific logic stays inside generator source/replay modules.

## Test Strategy

Unit tests:

- stable `case_key` construction and shape ordering;
- generator/case-set digest validation;
- generator Git commit/dirty-state/source-SHA capture and validation;
- static-suite inventory hash validation and preservation of declared order;
- deterministic budget allocation and per-kernel coverage;
- duplicate/corrupt case-manifest detection;
- NCU basic-recipe validation, instruction-count parsing, and classification
  failure handling;
- deterministic global tercile assignment, including equal-boundary
  `case_key` tie breaks and `unclassified` rows;
- rank-filter selection and required small -> medium -> large review markers;
- semantic-compatibility gating;
- shard determinism and resume keys;
- aggregate denominator and percentile calculations;
- report renderer links and figure metadata.

Integration tests use a tiny checked-in local benchmark package, never a full
production materialization. GPU/NCU/GCoM integration tests are separately
marked and run only when required tooling exists.

## Resolved Decisions

### OD1: Hardware Classification Before Detailed Comparison

Accepted. Run every selected kernel-and-shape case on hardware first with the
lightweight `ncu_basic_v1` recipe, collecting at least total executed
instructions plus execution-validity metadata. Persist global
small/medium/large ranks from that complete classification population. Run
detailed NCU/GCoM comparison, including stall reasons and detailed stats, in
small -> medium -> large order.

### OD2: Canonical Hardware Accuracy Metric

Deferred. Study the first detailed small-rank runs before selecting a single
canonical suite-level accuracy metric. Until then, retain named per-kernel
measurement contracts and reject comparisons with incompatible semantics.

### OD3: Committed Summary Reports

Accepted. Commit curated, reproducible summary reports and figures when they
are intentionally published. Keep full case rows, traces, logs, NCU exports,
and other large generated evidence under ignored `out/benchmarks/`.

### OD4: Generated Source Provenance And Cache

Accepted. Cache generated PPP sources under `out/benchmarks/.../build-cache/`,
keyed by generator/kernel/shape/compile-contract digests. Persist the
generator Git commit hash, dirty state, generator-source SHA-256, generated
source SHA-256, compile command, executable SHA-256, and generation command
in the materialized manifest and each applicable run record.

## Implementation Notes

### Phase 1: Materialization

Implemented on branch `bench`:

- `amora benchmarks list`, `inspect`, and `materialize`;
- AMORA-local `benchmark_generators/ppp_canonical/` with all nine canonical
  kernel identities and deterministic generated shape streams;
- exact `h100_2500` and `h100_5600` presets;
- immutable manifests with generator Git/dirty/source-hash provenance,
  canonical case keys, generator digest, and case-set digest;
- package data for the generator's CUDA replay templates.

The presets materialize exactly 2,500 and 5,600 unique cases. Allocation is
balanced while respecting actual per-kernel candidate capacity, so a constrained
kernel can receive fewer cases and unused allocation is redistributed
deterministically.

### Phase 2: NCU Basic Classification

Implemented on branch `bench`:

- local AMORA-owned CUDA replay templates and argument contracts for all nine
  canonical PPP kernel families;
- generic NVIDIA `ncu_basic_v1` collection of executed instructions, elapsed
  cycles, and duration;
- compile-contract-aware cache keys, warmup launch skipping, source/binary
  SHA-256, and complete NCU command provenance;
- immutable classification overlays with global tercile ranks;
- partial classification overlays deliberately omit ranks, preventing a
  bounded smoke run from redefining the full-population thresholds.

H100 validation on 2026-07-17 (CUDA 12.8, `sm_90`) classified one representative
case from each canonical kernel family. All nine classified successfully:

| kernel | total instructions |
|---|---:|
| `aligned_gemm_fp16` | 784,916 |
| `embedding` | 18,087,936 |
| `flash_attention_fwd` | 10,808,832 |
| `flashmla_dense_decode` | 487,296 |
| `gelu` | 370,900,992 |
| `gelu_gemm_fp16` | 1,183,584 |
| `megamoe_fp8` | 28,508,160 |
| `rmsnorm` | 72,560,640 |
| `rmsnorm_gemm_fp16` | 93,909,138 |

The resulting representative rank split was 3 small, 3 medium, and 3 large,
with instruction-count boundaries `small_max=1,183,584`,
`medium_max=28,508,160`, and `large_max=370,900,992`. This validates the
classification contract only; it does not replace the required all-case
hardware classification for a full materialized case set.
