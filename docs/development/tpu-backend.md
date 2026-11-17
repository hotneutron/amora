# TPU backend operation and qualification

AMORA's TPU backend uses JAX/PJRT/libtpu. `amora tpu` is the CLI name,
`tpu_jax` is the result identity, and `google_tpu` is the report vendor.
The backend separates generation profiles, runtime identities and deployed
instances. An installed or declared feature is not a hardware qualification.
Current milestone status is in the [execution tracker](../../.plan/trackers/20261116-1659-tpu-backend.md).
The [first TPU7x qualification](../../reports/google_tpu/tpu7x/qualification-summary.md)
is complete; other generations require separate native admission.

## Install and inspect

Current qualification-machine access:

```bash
ssh -p 2002 'zhangpeihua.1/307a053a-d97b-467e-a667-84ba2b50b7bd/maas@101.47.15.191'
```

This route is one instance of the multi-generation backend. Bootstrap and
qualification are repeated when the instance changes.

The base AMORA package can list probes without NumPy, JAX or libtpu:

```bash
python -m amora tpu list
python -m amora tpu inspect-capabilities
```

On an exclusive Linux TPU host, install a compatible runtime:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[portable-core,test,tpu]'
.venv/bin/python -m amora tpu inspect-capabilities --timeout 120
```

The initial Ubuntu host lacked ensurepip. Its tested bootstrap was a targeted
`sudo -n apt-get install -y python3.12-venv`, followed by a user-owned environment.
This also updated the associated Ubuntu Python packages; the execution audit
records the installation output. Other hosts can use their own verified Python
bootstrap. Docker/account-group changes are unnecessary.

Before JAX initialization the worker scans visible accelerator file descriptors,
using noninteractive sudo for a complete Linux scan where available. Busy or
unverifiable ownership returns an explicit reason. It never resets devices or
terminates a competing workload. The discovery worker exits before measurement
starts; every measurement has its own bounded worker and records ownership
before and after execution.

`--device tpu:N` indexes locally addressable JAX devices. It does not imply
exclusive ownership of that physical core: libtpu may acquire all visible chips.
The capability payload records global/local devices, process index, coordinates,
core identifiers, raw `device_kind`, resolved profile, scoped architecture data,
and exposed memory statistics. Missing fields remain unknown.

On GKE provision-only nodes, the TPU metadata service may omit the runtime's
topology keys. The current instance's node labels establish TPU7x, four chips
and `2x2x1`; native discovery succeeds with those explicit bounds and
`TPU_SKIP_MDS_QUERY=1`. Generate the configuration from the current node rather
than copying it to another instance:

```bash
python3 tools/tpu/configure_gke.py --out tpu-instance.json
python3 tools/tpu/instance.py --config tpu-instance.json --python .venv/bin/python \
  -- -m amora tpu inspect-capabilities --timeout 120
```

The configuration helper reads only this node's metadata through the existing
kubelet credentials. It rejects multi-host topology and creates a task-owned
runtime log directory. The execution wrapper rejects a different hostname and
records every permitted TPU setting; manifests retain the instance configuration.

## Reproduce the runtime

Keep a local wheelhouse and an exact hash-locked `runtime-lock.txt` in the
checkout where the commands run. Runtime identity includes that lock's hash.
Acceptance also checks that every locked version matches the installed runtime.
An unlocked or mismatched runtime retains diagnostic observations with
`inconclusive` status and cannot be published as a curated qualification report.
The first environment used JAX/jaxlib 0.12.0.1 and libtpu 0.0.50.1 on Python
3.12; this combination is qualified for the documented TPU7x instance/recipes.
Other generations may need different locks.

```bash
.venv/bin/python -m pip download --dest wheelhouse 'jax[tpu]' numpy pytest setuptools wheel
.venv/bin/python tools/tpu/lock_runtime.py --wheelhouse wheelhouse
python3.12 -m venv .venv-replay
.venv-replay/bin/python -m pip install \
  --no-index --find-links wheelhouse --require-hashes -r runtime-lock.txt
```

The execution audit includes all 19 wheel filenames and SHA-256 values for the
initial environment. Lock creation must hash the actual downloaded wheels;
package version strings alone do not establish the artifact identity. AMORA
copies available lock/wheel-manifest files into each measurement run. Replay
both a TPU vector and BF16/FP32 matrix correctness smoke before accepting a
new generation/runtime/instance combination. CPU replay is a separate check.

## Measure probes

Run from the repository root with the qualified environment. The provided
recipes are editable inputs; their canonical digest is frozen into each run.

For a configured GKE instance, run the prepared qualification campaign:

```bash
python3 tools/tpu/instance.py --config tpu-instance.json --python .venv/bin/python \
  -- tools/tpu/campaign.py --campaign-id native-001 --out-root out/tpu --timeout 900
```

This executes native correctness smoke, core28 shards, the single-device and
Pallas recipes, a 2/4/8-device collective sweep, and optional trace capture.
Each stage has an immutable receipt and preserves failed or noisy results.

```bash
.venv/bin/python -m amora tpu measure \
  --recipe docs/development/tpu-single-device.json \
  --out-root out/tpu --run-id baseline-001 --timeout 900
.venv/bin/python -m amora tpu measure \
  --recipe docs/development/tpu-pallas.json \
  --out-root out/tpu --run-id pallas-001 --timeout 900
.venv/bin/python -m amora tpu measure \
  --recipe docs/development/tpu-collectives.json \
  --out-root out/tpu --run-id collectives-001 --timeout 900
```

Pallas and collective recipes need their corresponding runtime/hardware
features. Collective participant counts must fit the actual local mesh. The
initial recipe uses two participants; create additional frozen recipes to
qualify larger meshes. Profile selection is automatic from exact sourced
aliases; `--profile` can supply missing identity, while a contradictory profile
is rejected. Unknown future devices retain their raw identity.

| Probe | Implementation | Measured scope |
|---|---|---|
| `topology.device_attributes` | Runtime device and profile inventory | Metadata; direct device values and sourced architecture facts remain distinct. |
| `runtime.dispatch_latency` | Observable one-element JIT computation | Dispatch-to-completion distribution. |
| `matrix.matmul_throughput` | BF16 operands, FP32 dot output | Logical FLOP/s for the compiled program. |
| `matrix.dependent_chain` | Repeated square product; admitted recipe uses dense dyadic Hadamard weights and dyadic inputs with bounded norm | Compiled-chain scaling for that fixture; arbitrary random chains and isolated instruction latency are not qualified. |
| `vector.elementwise_throughput` | Observable FP32 multiply/add | Effective program rate. |
| `transfer.host_device` | Fresh H2D allocation; D2H first creates a fresh device buffer to bypass JAX's host cache | D2H includes the device-local copy and host transfer; its interval does not isolate either component. |
| `global_memory.streaming` | Pallas HBM/VMEM tiled stream | Requested-byte rate, with generated-code artifacts. |
| `local_memory.transfer_sweep` | Explicit Pallas DMA copies, VMEM scratch and semaphore waits | Whole-program transfer/tile behavior; no isolated VMEM latency claim. |
| `interconnect.collectives` | `shard_map` with `psum`, `all_gather` or `ppermute` | Whole compiled collective program on the recorded mesh/routing. |

## Portable workload admission

The TPU bridge traces the canonical node graph with already validated host
inputs. It performs no host validation, NumPy reads or input staging in the
compiled body. Existing generator/oracle files and their six fingerprints are
unchanged. The bridge's adapted mathematical code retains Apache-2.0 licensing.
CPU correctness validates graph fidelity; native TPU execution needs its own run.

```bash
PYTHONPATH=. .venv/bin/python tools/tpu/qualify.py \
  --tier 28 --profile smoke --shard-size 32 \
  --campaign-id core28-001 --out-root out/tpu --timeout 900
```

Each shard is bounded and immutable. The aggregate receipt requires all 184
unique smoke cases, one target/runtime identity and successful correctness.
The core20/core24 subsets contain 152/168 cases. Duplicate, unexpected, missing
or failed cases cannot qualify a tier. Current matrix and FP32 tolerances come
from portable-core r1. A numerical-policy change needs documented error evidence,
a versioned policy and new freeze; existing receipts remain intact.

## Timing and artifacts

Each recipe declares warmup iterations, measured samples, repeats and batch K.
The default is 5 warmups and 30 measured samples in each of three repeats.
Warmups use the same K-invocation batch, output retention and drain as measured
units; `warmup_invocations` records the resulting execution count. The accepted
throughput recipe uses 20 warmup batches and K=32; D2H has a separate K=1 recipe.
One sample records K executable invocations followed by a drain of every output
leaf; outputs stay live until that drain. Inputs are not donated and executable
output allocation is included. Compiler lowering/compilation, staging and the
first execution are recorded separately from warm measurements.

`dispatch_to_completion_ns` is a synchronized host interval.
`median_ns_per_invocation` amortizes K invocations and is not device-only latency.
Relative MAD gates each repeat and repeat medians; noisy throughput results
remain `inconclusive`. Dispatch probes retain their distribution. Logical FLOPs
and requested bytes divided by this interval produce effective rates, not direct
MXU utilization, physical HBM bandwidth or hardware instruction counts.

Set `capture_profile: true` in a recipe for a separate JAX trace lane. Its
artifacts are retained, but device-only intervals remain unqualified until their
timestamp/interval semantics are validated. A missing profiler reports its reason.

Run directories follow `out/tpu/<recipe-id>/<recipe-digest>/<run-id>/`:

- `recipe.json`, `source.json`, `capabilities.json` and available runtime lock files;
- `semantic-verification.json`, `correctness.json` and incremental `cases/` receipts;
- `samples.jsonl`, `report.json`, `compiler/` and optional `profile/`;
- final ownership/worker records and `manifest.json`, which hashes every artifact.

The validator rejects escaping paths, missing files and changed hashes.
Existing runs cannot be overwritten. Failed initialization and timed-out workers
retain partial evidence; they cannot become successful CPU fallback results.
StableHLO/optimized HLO live in manifest artifact entries and launch references;
binary/disassembly identity fields are reserved for actual binary/disassembly.

## Reports and tests

Render an accepted run's JSON report:

```bash
.venv/bin/python -m amora tpu report --input out/selected-run/report.json --out-dir reports
```

The actual run path is returned by `measure`. Reports are under
`reports/google_tpu/<generation>/<target-digest>/<runtime-digest>/<run-id>/`,
with `probes-<sku>.md` and `environment.md`. Family indexes use the same composite
identity, preserving different instances/runtimes with the same SKU. Registered
TPU report groups organize runtime/topology, compute, memory/transfer and fabric.
`docs/vendors/google/` follows the existing vendor-document taxonomy.
Compiler references in launch descriptors and the source manifest resolve
against the recorded source evidence root. Historical raw compiler metadata
can contain program-local basenames; use the canonical launch/manifest paths.
Report creation validates a sibling run manifest when present and rejects runs
whose qualification status is not passed.

```bash
JAX_PLATFORMS=cpu python -m pytest -q
AMORA_TEST_TPU=1 .venv/bin/python -m pytest tests/backends/tpu/test_hardware.py -q
```

Collection does not initialize JAX/libtpu. TPU hardware tests need the explicit
environment opt-in; CPU synthetic tests never count as generation hardware
evidence. The existing NVIDIA/NCU/GCoM benchmark pipeline is independent of
portable TPU admission; CUDA-template PPP replay is outside this backend.

## Revision History

| Revision | UTC date | Change |
|---|---|---|
| r1 | 2026-11-17 | Documented implemented CLI, instance/profile/runtime boundaries, tested bootstrap and lock replay, immutable campaigns, measurement semantics, and separate CPU/hardware qualification. |
| r2 | 2026-11-17 | Added the user-supplied replacement SSH command and explicit requalification on instance changes. |
| r3 | 2026-11-17 | Documented source-backed GKE topology configuration, hostname-scoped execution and the prepared native campaign. Native discovery confirmed TPU7x with eight TensorCores on four chips; correctness/performance campaigns remain separately qualified. |
| r4 | 2026-11-17T19:46:22Z | Recorded completed TPU7x qualification and independent replay, matched-batch warmup, fresh-buffer D2H and bounded-chain scopes, canonical artifact references, and the distinction between this instance's native evidence and other catalogued generations. |
