# Portable-core V2 accelerator qualification and TPU latency

V2 uses the same frozen workloads on CPU, CUDA and TPU. The accelerator checks
validate device placement and correctness; they do not collect timing. TPU
latency collection is a separate step that consumes verified correctness
receipts. Neither step changes the legacy `--tier` catalogue or the V2 CPU
release matrix.

## Freeze and qualify a shard

Use an authorized host with the matching runtime: CUDA-enabled PyTorch for
`cuda_torch`, or JAX/jaxlib/libtpu with a verified TPU runtime lock for `tpu_jax`.
Run from the repository root in the environment for that backend. This example
uses a small H3 development-control suite, not unreleased historical H3 cases.

```bash
pip install -e '.[portable-core]'
amora benchmarks inspect-v2
amora-portable-core freeze --suite h3-programs --profile smoke \
  --out out/portable_v2/h3-smoke
```

On the CUDA host, with an idle NVIDIA device visible to Torch:

```bash
amora benchmarks qualify-v2 --backend cuda_torch --device cuda:0 \
  --frozen out/portable_v2/h3-smoke --out out/portable_v2/cuda-h3-smoke
```

On a TPU host, use the verified topology settings described in the
[TPU backend guide](tpu-backend.md). The example assumes `tpu-instance.json`
was generated for that node and `runtime-lock.txt` is present.

```bash
python tools/tpu/instance.py --config tpu-instance.json -- \
  -m amora benchmarks qualify-v2 --backend tpu_jax --device tpu:0 \
  --expected-generation tpu7x --frozen out/portable_v2/h3-smoke \
  --out out/portable_v2/tpu-h3-smoke
```

`inspect-v2` reports declared capabilities without opening a device. To qualify
a larger snapshot, select up to 32 cases with repeated `--case` flags; use
`--manifest-sha256` to bind an independently approved manifest. Each run writes
`request.json`, `result.json`, worker logs, and rounding controls in a new
output directory. A request containing unsupported rows returns nonzero even
when its other rows pass. Keep those denial receipts; do not pack strided views
or substitute a CPU result. TPU produces StableHLO and optimized HLO for both
output-only and node-capture variants. Eager Torch CUDA produces correctness
receipts, not compiler IR or SASS.

The [accelerator execution tracker](../../.plan/trackers/20261123-1650-portable-core-v2-cuda-tpu-backends.md)
records the native H100 and TPU7x runs. On each target, 1,130 of 1,141 frozen
cases passed; seven duplicate scatter and four physical layout cases were
rejected as expected. These results are specific to the recorded realizations.

## Collect verified TPU latency labels

The [latency collector](../../tools/portable_core_v2/collect_tpu_latency.py)
accepts the verified full-corpus bundle: `full-pins.json`,
`tpu-verification.json`, `evidence/full-summary.json`, the evidence inventory,
correctness receipts and output-only compiler IR. Copy the frozen snapshots and
verified plan to the same TPU target/runtime used for correctness. The timing
run expects snapshots at `<frozen-root>/<suite-key>/`.

```bash
python tools/portable_core_v2/collect_tpu_latency.py plan \
  --artifacts out/portable_v2_backends/20261123-1816-full \
  --out out/portable_v2/latency-plan.json

# On the admitted TPU target with the frozen snapshots and plan staged:
python tools/tpu/instance.py --config tpu-instance.json -- \
  tools/portable_core_v2/collect_tpu_latency.py run \
  --plan out/portable_v2/latency-plan.json --frozen-root full-frozen \
  --suite operation-gate-v2-reference --start 0 --stop 2 \
  --out out/portable_v2/latency-shard-0000

# After retrieving the raw shard receipts, repeat --shard for every shard:
python tools/portable_core_v2/collect_tpu_latency.py assemble \
  --plan out/portable_v2/latency-plan.json \
  --shard out/portable_v2/latency-shard-0000/shard.json \
  --out out/portable_v2/latency-corpus.json
```

The two-case example leaves the corpus incomplete. The run checks input bytes,
placement, output correctness, and compiler identity; it allows differences
in optimized HLO source-stack metadata only. The label is warmed singleton
**host dispatch-to-output-ready latency**; it does not include initial
compilation, input staging, output transfer to the host, or profiler capture.
Assembly recomputes the median and both relative-MAD checks from the raw
samples. Use `--retry-shard` only for the single allowed repeat of a previously
unstable case; accepted labels cannot be replaced.

The [latency and Accorde tracker](../../.plan/trackers/20261124-1557-portable-v2-tpu-latency-accorde.md)
records the verified 1,130-label TPU7x corpus and grouped held-out fits. The
Accorde dataset handoff needs the assembled ledger, the frozen manifests, and
the output-only StableHLO files from the correctness bundle. Keep raw native
samples, compiler/worker archives and model binaries in local artifacts.
CUDA latency, other TPU generation labels, and calibrated TPU-only cycles are
outside this dataset.

## Revision History

| Revision | Date (UTC) | Change |
|---|---|---|
| r1 | 2026-11-25 | Documented V2 qualification, device artifact boundaries, and the verified TPU latency handoff. |
