# TPU7x XPlane timestamp normalization verification

2026-11-23. Follow-up to the [matched profiler experiment](20261123-1246-tpu-findings-profiler-experiment.md).

**Host normalization verified. TPU-to-host clock alignment failed.** Do not
strip `XLine.timestamp_ns` or apply a fixed clock correction. Keep Amora's
`device_duration_ns` unset and do not use this timeline as simulator truth.

## Verified timestamp rule

The locked JAX/jaxlib 0.12.0.1 build identifies JAX revision
`dc4af684446e65ee98eee8090e1ff6f8211473fb`, whose [build manifest](https://github.com/jax-ml/jax/blob/dc4af684446e65ee98eee8090e1ff6f8211473fb/MODULE.bazel#L32-L44)
pins OpenXLA `d867969800d4a8fd045505eabc65e1e66f30794f`. Its
[collector](https://github.com/openxla/xla/blob/d867969800d4a8fd045505eabc65e1e66f30794f/third_party/tsl/tsl/profiler/lib/profiler_session.cc#L145-L153)
[postprocesses XSpace](https://github.com/openxla/xla/blob/d867969800d4a8fd045505eabc65e1e66f30794f/xla/tsl/profiler/convert/post_process_single_host_xplane.cc#L103-L117)
before serialization. [Normalization](https://github.com/openxla/xla/blob/d867969800d4a8fd045505eabc65e1e66f30794f/xla/tsl/profiler/utils/xplane_utils.cc#L729-L758)
rebases epoch-valued line timestamps to session start, leaves already-small
line bases alone, and does not remove line-relative event offsets. The
[visitor](https://github.com/openxla/xla/blob/d867969800d4a8fd045505eabc65e1e66f30794f/xla/tsl/profiler/utils/xplane_visitor.h#L190-L205)
adds line base and event offset.

For the observed normalized host lines, using `Task Environment.profile_start_time`
as the recorded session epoch in ns:

```text
start_relative_ps = XLine.timestamp_ns * 1000 + XEvent.offset_ps
start_epoch_ns    = profile_start_time + start_relative_ps / 1000
end_relative_ps   = start_relative_ps + XEvent.duration_ps
```

Apply this to events with `offset_ps`, not to aggregated events. Keep
arithmetic relative until converting to ns; epoch picoseconds overflow int64.
The formula recovers the timestamp represented by a plane. It does not by
itself prove that another plane's clock is synchronized.

## Independent controls and results

On the previously qualified TPU7x instance, with the same runtime lock and
libtpu 0.0.50.1, three fresh sessions captured `TRACE_COMPUTE_AND_SYNC` without
performance counters. Each captured two distinct compiled kernels in a known
12-call sequence with irregular 1–29 ms gaps and explicit no-dispatch windows.
The observer sampled `CLOCK_REALTIME` around each TraceAnnotation entry/exit
and immediately before/after each synchronized call. A second host thread
provided an independent annotation line; no timed TPU work ran there. The two
kernels passed exact output checks. Ownership and release scans passed.

The native `jaxlib._profiler.ProfilerSession.stop()` returned serialized
XSpace before Python file export. For all three captures, the stored export
protobuf decoded equal to the native protobuf despite different byte hashes.
Perfetto starts equaled the schema-derived starts for all 45 selected host
markers and 36 selected device modules at the exported precision. The
cross-plane mismatch therefore already exists in the collected XSpace; it is
not introduced by the Python file or Perfetto conversion. This does not
identify the defect inside libtpu or the device clock translation.

| Session | Host line base (µs) | Host markers inside both independent wall-clock brackets, schema / offset-only | TPU modules fully inside the independent dispatch-to-ready bracket | TPU events starting before the Python call |
|---|---:|---:|---:|---:|
| 0 | 512.870 | 15/15 / 0/15 | 0/12 | 12/12 |
| 1 | 161.380 | 15/15 / 0/15 | 4/12 | 8/12 |
| 2 | 190.237 | 15/15 / 0/15 | 1/12 | 11/12 |
| Total | — | **45/45 / 0/45** | **5/36** | **31/36** |

All 36 device modules had the expected A/B identities and order on TPU:0;
there were no such modules on the other visible cores or overlapping the
no-dispatch markers. Discarding the host line base moved the host windows
backwards and made 32/36 modules appear contained, while falsifying the
independent host-time check for all 45 markers. This explains why the earlier
containment-only result was not evidence for an offset-only parser rule.

An additive correction to device timestamps would have to be in
**[528.97, 661.72] µs** in session 0, **[32.35, 171.71] µs** in session 1, and
**[52.81, 197.21] µs** in session 2 to fit every call. These are feasible bounds
from synchronization brackets, not estimates of the actual correction. Their
intersection is empty, so no fixed correction fits these captures.

## Decision and reproducibility

Preserve documented host timestamp arithmetic. Keep recorded device spans as
trace diagnostics only. The remaining gate is a verified device-clock mapping
or a stable execution identity and explicitly qualified device interval that
does not rely on cross-clock containment. The original TPU7x qualification,
other generation statuses, and simulator calibration status are unchanged.

Local artifacts: `out/tpu_profiler_experiments/20261123-1319-normalization/`.
`downloads/` contains the four SHA-verified archives with all raw and rendered
traces used by this verification, host markers, compiler IR and ownership
receipts. `capture.py` and `check.py` reproduce the campaign and checker.
`normalization-verification.json` holds per-call bounds and detailed checks;
its SHA-256 is
`50c5b84da873f11267da57a6ff6bb107c001db813fe75cb9f5aed234b116d7d8`.
This study used one runtime/host and does not qualify other TPU generations.

## Revision History

| Revision | Date | Change |
|---|---|---|
| r1 | 2026-11-23 | Verified the host schema against independent wall-clock brackets and pinned source; rejected offset-only host normalization and a common device correction. |
