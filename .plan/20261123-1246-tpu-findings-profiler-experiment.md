# TPU7x profiler experiment: trace scope, timestamp alignment, and overhead

Run `20261123-1246-profiler-r2`, 2026-11-23. Follow-up to the [TPU literature
findings](20261117-1646-tpu-findings-compiler-profiler-simulator.md) and separate
from the [accepted TPU7x baseline](../reports/google_tpu/tpu7x/qualification-summary.md).

**Decision:** Use explicit `TRACE_COMPUTE_AND_SYNC` with `tpu_perf_counters=False`
for TPU7x synchronization diagnostics on this stack. Keep `device_duration_ns`
unset. Neither the raw XPlane field formula nor the rendered Perfetto timeline
reliably aligns these device and host events. Counter visibility is confirmed;
per-invocation counter attribution and simulator calibration are not.

## Campaign

- `gke-tpu-f20b10b7-2pbv`, TPU7x, eight logical TensorCores on four chips; Python
  3.12.3, JAX/jaxlib 0.12.0.1, libtpu 0.0.50.1, NumPy 2.5.5. The runtime lock and
  digest matched the baseline. The experiment used its own TPU log directory,
  recorded as part of the target configuration. No workers were stopped.
- Base commit `f67fb13`; 21 deployed source files matched the commit before
  execution. The accepted probe definitions were reused, with K=1 for all
  timing lanes. Each case held its seed/inputs, executable, StableHLO and
  optimized HLO fixed across lanes. All five passed the existing numerical
  oracle before and after the campaign. Exclusive-ownership scans passed
  before and after each case and again after the campaign exited.
- Five cases × four block orders × four lanes = 80 windows, 60 captures, and
  2,400 timed calls. Lanes: unprofiled; default TPU options; explicit
  compute/sync with counters disabled; explicit compute/sync with counters
  enabled. Host tracer level 2 and Python tracer level 0 were fixed in the
  profiled lanes. Every window warmed 10 calls, inserted a 20 ms no-dispatch
  marker, measured 30 individually marked synchronized calls, and inserted a
  second 20 ms no-dispatch marker. A balanced four-order rotation limited
  order effects. These are separate diagnostic timings, not a repeat of the
  earlier batched throughput qualification.
- Retained diagnostic R1: the locked JAX build does not expose
  `ProfileOptions.device_tracer_level`; it raised `AttributeError` before the
  first trace. R2 removed that field, used the supported options, and exited 0.

## Synchronized host overhead

Times are medians of four window medians, in microseconds per call. Percentages
are the median of the four within-block ratios to unprofiled time, minus one.
The timed interval starts before invocation and ends after all outputs are
ready. Capture setup, trace stop/export, and idle markers are outside it.

| Case | Off (µs) | Default (µs; overhead) | Compute/sync (µs; overhead) | Compute/sync + counters (µs; overhead) |
|---|---:|---:|---:|---:|
| BF16 GEMM 1024×1024×1024 | 121.13 | 128.84; +5.5% | 128.45; +5.8% | 127.79; +5.4% |
| BF16 GEMM tail, M=511/N=513/K=257 | 123.41 | 129.84; +5.2% | 131.11; +6.4% | 130.08; +5.3% |
| FP32 vector 1024×1024 | 122.80 | 128.40; +4.9% | 128.98; +4.8% | 128.63; +4.3% |
| Pallas DMA 1024×1024, tile 256×128 | 135.98 | 143.34; +5.7% | 142.76; +4.9% | 142.99; +5.1% |
| Four-participant `psum`, 65,536 FP32 elements per participant | 239.48 | 246.31; +3.1% | 248.67; +4.1% | 247.81; +3.5% |

The maximum within-window relative MAD was 4.36%; no lane's relative MAD
across the four window medians exceeded 1%. The paired median overhead was
3.1–6.4%; individual block deltas ranged from 1.6% to 7.7%. These observations
are not a constant that can be subtracted from other host timings. File export
for completed windows overlapped some later windows on the host; accelerator
ownership was exclusive, but the host CPU was not reserved.

Across cases, median capture setup took 78–115 ms. Median stop/export costs
were 1.52–2.06 s for compute/sync without counters, 2.78–7.77 s for default,
and 9.22–9.62 s for compute/sync with counters. These include JAX's file export
and Perfetto conversion; they are not device overhead.

## What the traces showed

- **Synchronization:** The explicit mode added `Tensor Core Sync Flag` tracks
  absent from default captures. With counters disabled, TPU:0 contained 520
  sync records per GEMM window, 440 for vector, 4,280 for Pallas DMA, and 2,800
  for `psum`. Counts include warmups and are not instruction or DMA counts.
  `Pallas Primitives` and `TC Overlay` were empty on the observed TPU tracks;
  this mode alone did not expose a timed listing of Pallas DMA operations.
- **Counter capture:** Default and explicit counter-enabled captures each
  contained 94,039 distinct names with `counter_value` on TPU counter tracks.
  Explicitly disabling counters removed those records, but not the separate
  `P State` event. The named counter records did not set either `XEvent.data`
  member (timestamp offset or occurrence count). They are capture-level
  observations, not timestamped per-call deltas. The same names on multiple
  tracks do not establish independent chip/core measurements.
- **Module scope:** Selected `XLA Modules` events yielded 40 instances on
  each expected logical device per trace: TPU:0 for four cases, TPU:0–3 for
  `psum`. Compiled op children and parallel device spans must not be summed as
  program elapsed time. The four-device envelope, when used for diagnosis,
  is earliest module start to latest module end, not isolated ICI latency.

## Timestamp gate: not admitted

For each capture, the checker required one `XLA Modules` interval per expected
track within each of 30 sequenced synchronized host markers, ten module
instances per track during warmup, a consistent module identity per track,
and no module overlap with either no-dispatch interval.

| Timestamp interpretation | Captures passing all controls |
|---|---:|
| Raw XPlane `XLine.timestamp_ns * 1000 + XEvent.offset_ps` | 20/60 |
| Rendered Perfetto timeline | 20/60; same pass/fail by capture |
| Raw XPlane offsets alone, as a diagnostic | 60/60 |

For example, in the first aligned GEMM default trace the first measured host
marker's raw offset was 516,644.653 µs and the device module started at
516,722.887 µs. Adding the host line's base of 2,077,135 ns moved the marker
start to 518,721.788 µs. That trace passed only 9/30 sample associations with
the schema interpretation and had 15 module overlaps with the idle markers.
This is an export-specific discrepancy, not a correction to the public
[XPlane schema](https://github.com/openxla/xla/blob/main/third_party/tsl/tsl/profiler/protobuf/xplane.proto).
Ignoring line bases is not a general parser rule, even though it restored the
expected warmup/sample/idle counts in this controlled campaign.

**Follow-up:** The [independent timestamp verification](20261123-1319-tpu-findings-timestamp-normalization.md)
confirmed that adding the host line base is correct. Device-to-host alignment
failed against independently bracketed calls before export; the apparent
success of raw offsets alone is not a valid normalization rule. A verified
clock mapping or a stable execution-ID join remains necessary. Use the
unprofiled host lanes for host performance only. Do not fit SCALE-Sim cycles
to those values as if they were core latency. No simulator was run, and no
other TPU generation was qualified.

## Evidence

Local artifact directory: `out/tpu_profiler_experiments/20261123-1246/`.
`native-receipts.tar.gz` contains the timings, per-case ownership and correctness,
compiler IR, campaign/protocol metadata, and the full remote file hash inventory.
The archive SHA-256 is
`0f728f0de1eb48d2ef900297c6a7009a1473e4cd91118171692cfb174f5792de`;
122 included run files matched the 305-entry on-host inventory. The on-host
analysis parsed all 60 XPlane/Perfetto pairs; its independently transferred JSON
matched SHA-256
`f3ebdc0116d720d2796d6ac48e13c9dd1cd0f0c2730550bea86504482baac02e`.
The aligned GEMM XPlane and Pallas compute/sync XPlane/Perfetto pair were
hash-verified locally. Local Pallas reanalysis matched the on-host row.
The full raw dataset (~1.8 GiB) remains on the TPU host; the local artifact
contains a verified subset, not a complete raw trace mirror.

## Revision History

| Revision | Date | Change |
|---|---|---|
| r1 | 2026-11-23 | Recorded the matched TPU7x campaign, independent evidence checks, and limits on timestamp and counter attribution. Preserved the original baseline qualification. |
| r2 | 2026-11-23 | Linked the independent normalization verification. Superseded offset-only containment as a candidate parser rule; the raw observations and qualification limits are unchanged. |
