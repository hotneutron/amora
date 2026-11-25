# Comments: TPU Backend Plan

Reviewed plan: `20261116-1659-tpu-backend.md` (r1, sha256
`214afbcef8c7b6f7f16c9810db6fb82933758b3ac6f63686afb7c5f30ef3ce5f`).

Reviewed on: 2026-11-17. Rechecked against r2 on 2026-11-17.

Scope: repository-contract verification, measurement methodology, and
implementation readiness. This document is a review artifact, not an approved
plan.

Status: the r1 findings in "Required changes" were addressed by r2; the
"Recheck of r2" section records the disposition. The r1 text is retained as the
reviewer's original artifact.

## Position

The direction is sound and the evidence discipline is better than most AMORA
plans. JAX/PJRT first, Pallas only after per-generation compile-and-execute
qualification, explicit timing domains, immutable artifacts, and a
runtime-bootstrap gate are the right decisions.

Do not start T0 until the changes in "Required changes" are folded into the
plan. The rest are design clarifications that will otherwise surface as
rework during T1-T4.

This position applied to r1. r2 addressed the required changes; see
"Recheck of r2" below.

## Verification performed

Claims checked against baseline `6955c5256645c7303f73f5d3e96e5f99c42656be`
(HEAD at review time):

- The portable-core smoke counts are 152/168/184 for tiers 20/24/28. The case
  ID sets are strictly nested; each tier adds exactly 16 cases. Verified by
  running `generate_cases(tier, "smoke")` from `benchmark_generators.portable_core`.
- `SEMANTIC_FINGERPRINTS.json` contains six tier/profile entries.
- The NVIDIA registry contains 36 probes, not the counts stated in the root
  README.
- `benchmark_generators/portable_core/__main__.py` enforces a fixed 120-second
  whole-run bound in `check`, so the plan's sharded-coverage concern is real.
- The live CLI exposes `gcom_cuda`; `benchmarks classify` hardcodes
  `ncu_basic_v1`/NVIDIA and `benchmarks detail` hardcodes NVIDIA NCU + GCoM.
- The host-inspection record matches the plan's table for hostname, OS, Python,
  CPU count, PCI IDs, VFIO access, missing runtime modules, Docker socket
  denial, and missing `ensurepip`.
- `out/` is ignored by Git; the source baseline commit exists.

Claims that do not hold or are not supported are called out below.

## Required changes

### R1. The report layout will not produce `reports/google_tpu/<family>`

The plan names CLI `tpu`, serialized backend identity `tpu_jax`, and curated
report root `reports/google_tpu/<verified-family>/`. Those three names do not
connect in the current renderer.

Evidence:

- `amora/reports/markdown_report.py` maps only `nvidia_cuda` to a vendor
  (`_VENDOR_BY_BACKEND`). Any other backend falls back to
  `backend.split("_", 1)[0]`, so `tpu_jax` becomes `tpu`.
- `_MODEL_PATTERNS` has no TPU generation patterns, so family/SKU derivation
  returns `unknown-*`.
- `derive_sku` scrapes `"NN GB"` from the device name, which TPU device names
  do not reliably provide.
- `_build_family_environment` hardcodes `CUDA:` and `GPU:` fields, which will
  render as `None` for TPU.
- The implementation-surface table lists `amora/reports/probe_groups.py` but
  omits `amora/reports/markdown_report.py`.

Suggested change:

- Add `tpu_jax -> google_tpu` to `_VENDOR_BY_BACKEND`.
- Either add TPU device-name patterns (generation plus SKU) or make
  `--vendor/--family/--sku` mandatory for `amora tpu report`.
- Make the family environment section backend-neutral so it does not print
  CUDA/GPU fields for TPU.
- Add `amora/reports/markdown_report.py` to the implementation surface.
- Require the TPU capability payload to expose a `devices` list with a `name`
  field, because `_primary_device_name` and the report header depend on it.

### R2. Vendor documentation path contradicts the existing taxonomy

The plan proposes `docs/vendors/google_tpu/sources.md`. The repository already
has `docs/vendors/google/architectures/`, `docs/vendors/google/tpu/`, and
`docs/vendors/google/xla/`, and
`.plan/20260617-1659-amora-vendor-document-structure.md` fixes the vendor
directory as `google/`.

Suggested change: use `docs/vendors/google/tpu/sources.md` (or
`docs/vendors/google/sources.md`, matching the NVIDIA layout). Keep
`reports/google_tpu/` as the backend-label report root; that asymmetry matches
`reports/gcom_cuda/`.

### R3. "pip/uv/conda absent from PATH" is not supported by the audit record

The recorded inspection only tests `python3 -c "import venv,ensurepip"` and
`importlib.util.find_spec` for runtime modules. It does not test `pip`, `pip3`,
`uv`, or `conda`.

Suggested change: either add those probes to `host-inspection.json` or reword
the plan to "ensurepip absent; pip availability unverified". Do not state an
unrecorded observation as verified.

### R4. `google-sudoers` membership is ignored

The identity output in `out/tpu_backend_plan/20261116/host-inspection.json`
shows the account belongs to `google-sudoers`. That likely means passwordless
`sudo`, which changes T0's bootstrap options and the shared-host risk posture.
The plan currently treats the account as unprivileged because the Docker
socket is denied.

Suggested change: T0 must run `sudo -n true`, record the result, and state the
policy. If sudo is available, `apt-get install python3-venv` and Docker-group
access become explicit options; if not, the user-level bootstrap path is the
only path. Either way, record the decision.

### R5. Deployment and package egress are missing prerequisites

T0 assumes the repository is already on the host and that the JAX/libtpu wheel
set can be downloaded. The plan has no step for either.

Suggested change: add two T0 prerequisites:

- Verify outbound HTTPS to PyPI or an internal mirror, and record the index URL.
- Choose and record the transfer mechanism (`rsync`, `git clone`, or an
  internal artifact store), including credentials handling.

Produce the hash-locked environment on the host, and replay one smoke run from
the lock in the same milestone rather than deferring the replay to T5.

## Design gaps

### D1. Device ownership and exclusive access

`tpu:0` pins an array to one logical device; it does not shrink libtpu's
runtime footprint. libtpu also has process-level ownership/lock behavior, and
the host is multi-tenant (other users' directories are present).

Add to every timed run, not only collectives:

- Record the device-holder/lock state before the run.
- Require a clean, exclusive run for accepted measurements.
- Make the bounded discovery worker exit before the measurement worker starts,
  and state that measurement itself runs in the bounded worker with a timeout.
  The current plan lists `worker.py` "process ownership" but never says the
  measurement lane uses the same boundary.

### D2. Define the timed unit and output-allocation policy

Step 4 says "time invocation" with `perf_counter_ns`. For small and medium
shapes, Python dispatch can dominate device time. Each recipe should declare:

- One dispatch-to-completion per sample (latency), or K back-to-back
  invocations with one drain (throughput), and record K.
- Whether outputs are allocated per invocation or reused/donated, and whether
  input buffers are donated.
- Warmup iteration count separately from measured sample count.

Without this, "throughput" curves can measure the host dispatch path.

### D3. The JIT bridge cannot reuse `JaxBackend.execute`

`JaxBackend.execute` calls `device_put`/`jnp.asarray` on every operand and uses
`self.numpy()` for gather inside the op loop. Traced inside `jax.jit`, that
embeds inputs as constants or synchronizes the host.

The new `amora/backends/tpu/portable_core.py` must trace `case["nodes"]`
directly with already-validated attributes (`indices_validated`,
`offsets_validated`), and keep `validate_inputs` outside the trace. State this
explicitly; "adapt the existing prepared execution path" is not enough guidance
for the implementer.

### D4. Fingerprint verification has no procedure

The plan says "freeze and verify all six tier/profile fingerprints", but
`freeze` writes a manifest digest, not the `case_payload_sha256` recorded in
`SEMANTIC_FINGERPRINTS.json`. There is no existing tool that compares the two.

Suggested change: define the comparison (canonical case-payload SHA-256 per
tier/profile) and where the receipt is stored, or drop the claim.

### D5. Tolerance policy needs a legitimate escalation path

"Do not relax tolerances to hide JIT drift" is correct, but the plan provides
no path for a genuine platform difference. TPU reduction order and scatter-add
ordering can differ from NumPy; N27 uses duplicate scatter indices, and FP32
transcendentals may differ.

Suggested change: allow a new frozen tolerance revision only with evidence and
a recorded rationale; otherwise T2 can deadlock on a legitimate difference.
This is not permission to relax tolerances silently.

### D6. State the benchmark scope explicitly

`amora benchmarks classify/detail` remain NVIDIA/GCoM-only in this plan; TPU
execution of `ppp_canonical` is out of scope. The plan implies it but never
says it, so a reader could expect TPU PPP replay.

### D7. Prefer `shard_map` over `sharded_jit`

`sharded_jit` is an obsolete name. Use `shard_map` (or `jit` plus
`NamedSharding`) for T4, and name the actual collective primitive behind
"collective-permute" so the probe implementation is unambiguous.

### D8. Compiler-artifact hashes have no schema slot

`ProbeIdentity` exposes only `source_hash`, `binary_hash`, and
`disassembly_hash`. State that StableHLO/Mosaic/optimized-HLO hashes live in
the manifest and/or `launch.extras`, and that `disassembly_hash` stays empty
rather than being overloaded.

### D9. Make the naming contract explicit

Add one line: CLI `tpu`, serialized identity `tpu_jax`, report vendor
`google_tpu`. The three names are coherent, but they will drift without an
explicit contract.

### D10. Fix the stale count in `probe_groups.py`

The file says "30 probes" while registering 36. Since TPU work will touch this
file, correct the comment at the same time.

### D11. Terminology: TensorCore is not a TPU unit

The "do not infer" row says "TensorCore count". For TPU this should be "MXU
count" (or "matrix-unit count"). The rest of the plan avoids NVIDIA concepts
correctly; this one slips.

## Minor

- "At least 30 warm samples" is ambiguous. Separate warmup iterations from
  measured samples and record both.
- The tracked plan embeds the complete SSH principal and public IP. If the plan
  is committed, consider redacting them to a placeholder and keeping the exact
  command in an untracked note.
- `reports/google_tpu` versus `docs/vendors/google` is intentional; add one
  sentence so a future reader does not "fix" one to match the other.
- `tests/conftest.py` runs NVIDIA discovery at collection time. TPU marker
  gating must not import JAX or initialize libtpu during collection; the plan
  says this, but the test layout should make the boundary explicit.
- The root README's probe table lists only 8 of 36 probes. Since `README.md` is
  already in the implementation surface, include correcting that table in T5.

## Suggested amendment checklist

| Plan section | Edit |
|---|---|
| Implementation surface | Add `amora/reports/markdown_report.py`; adjust the docs path to `docs/vendors/google/tpu/`. |
| Machine baseline | Reword the `pip/uv/conda` row to match the audit record; add the sudo check. |
| T0 | Add deployment, package-egress, sudo, exclusivity, and lock-replay prerequisites. |
| Measurement contract | Add timed-unit, K, buffer-donation, and output-allocation policy; add device-hold recording. |
| T2 | Name the traced node-graph bridge; define fingerprint verification; add the tolerance-revision path. |
| T3/T4 | Use `shard_map`; name collective primitives; state the NVIDIA/GCoM-only benchmark scope. |
| Artifacts | State where compiler-artifact hashes live. |
| Naming | Add the `tpu` / `tpu_jax` / `google_tpu` contract line. |
| Reporting | Add the `--vendor/--family/--sku` requirement or TPU device patterns. |

## Recheck of r2

Target: `20261116-1659-tpu-backend.md`, r2, sha256
`35b4adf270ddfe11490d87af10eea93af4768975da63b38153a096926faee9dc`.

### Verdict

r2 resolves every actionable r1 item. The three r1 claims that were rejected
or narrowed in the plan's review table are correctly rejected; they were
rechecked independently and withdrawn. No blocking issue remains. The remaining
items are internal consistency details, not design flaws.

### Verification performed

- Recomputed `digest(generate_cases(tier, profile))` for all six tier/profile
  combinations. All six match `SEMANTIC_FINGERPRINTS.json`, with counts
  152/304, 168/336, 184/368.
- Reproduced the synthetic report claim: a `tpu_jax` report with device name
  `TPU v5p` derives `('tpu', 'unknown', 'tpu-v5p')`.
- Reviewed `review-source-checks.json`, `review-host-checks.json`,
  `review-wheel-egress.json`, `review-baseline.json`, and
  `plan-validation-r2.json`.
- Confirmed `sudo -n true` exits 0; `pip`/`pip3`/`uv`/`conda` are absent;
  `git` exists and `rsync` does not; PyPI `jax`/`libtpu` index requests and the
  libtpu wheel HEAD return 200.
- Confirmed `docs/vendors/google/{architectures,tpu,xla}` exist and are empty.
- Confirmed `test_case_fingerprints_preserve_upstream_contracts` exists and is
  parameterized over all three tiers and both profiles.
- Confirmed `plan-validation-r2.json` passes links, shell syntax, whitespace,
  fences, repo-relative paths, review preservation, suggestion recording,
  multiple-generation coverage, and revision history.
- Confirmed libtpu 0.0.53 publishes cp311, cp312, cp313 and newer x86_64
  wheels; the cp311 wheel in the HEAD probe does not block the host's Python
  3.12.

### Disposition of the r1 review

| r1 item | r2 disposition | Recheck note |
|---|---|---|
| R1 report layout | Addressed | `markdown_report.py` is now in the implementation surface; backend-to-vendor mapping, structured identity, selected-target primary device, and backend-neutral environment rendering are explicit. |
| R2 docs taxonomy | Addressed | `docs/vendors/google/tpu/` and `docs/vendors/google/architectures/`; report naming remains `google_tpu` with a stated rationale. |
| R3 package-tool evidence | Addressed | Original record preserved; fresh `shutil.which` supplement attached. |
| R4 sudo/bootstrap | Addressed | Privilege row and T0 now record `sudo -n true` and the selected method. |
| R5 deployment/egress | Addressed | T0 now covers source archive delivery, egress/wheelhouse, lock replay, and receipt binding. |
| D1 ownership/workers | Addressed | Exclusive-scope evidence, bounded measurement worker, and cleanup scope are explicit; the multi-tenancy assertion was correctly narrowed. |
| D2 timed unit/buffers | Addressed | Latency versus K-invocation throughput, donation, allocation, aliasing, and warmup/sample counts are defined. |
| D3 JIT bridge | Correctly narrowed | `indices_validated=True` avoids the gather host copy; `device_put`/`asarray` do not inherently constant-fold tracers. The accepted pure node-graph trace is the right implementation. |
| D4 fingerprints | Correctly rejected | The test exists and all six digests match. The r1 claim was wrong. |
| D5 tolerances | Addressed | Evidence-backed, versioned re-freeze path is defined. |
| D6 benchmark scope | Addressed | NVIDIA/GCoM-only classify/detail scope is explicit. |
| D7 sharding terminology | Addressed | `shard_map`/`NamedSharding` and `psum`/`all_gather`/`ppermute` are named; the `sharded_jit` correction is fair. |
| D8 compiler hashes | Addressed | `manifest.compiler_artifacts` plus `launch.extras` references. |
| D9 naming | Addressed | CLI/backend/report/vendor contract is explicit. |
| D10 stale probe count | Addressed | Corrected in T5 while touching the file. |
| D11 TPU units | Correctly rejected | Official JAX Pallas TPU docs use `TensorCore` and `TensorCoreMesh`; the scoped TensorCore/MXU/device/chip distinction is better than the r1 replacement suggestion. |
| Minor items | Addressed | Warmup wording, collection-time boundaries, README table, and SSH deferral are recorded. |

### Remaining consistency items

1. **`semantic-verification.json` is not in the run artifact contract.**
   T1/T2 says to store fingerprint expected/actual digests and counts in
   `semantic-verification.json`, but the artifact list in the measurement
   contract names only `capabilities.json`, `recipe.json`, `manifest.json`,
   `correctness.json`, `samples.jsonl`, `report.json`, `compiler/`, and optional
   `profile/`. Add the file to the list or fold it into `correctness.json`.

2. **TPU report index key and file naming are underspecified.**
   The plan correctly says `manifest['skus'][sku]` must not merge distinct
   instances/runtimes, but the current renderer also writes
   `probes-<sku>.md` under a single family directory. Two targets with the same
   SKU would overwrite that file even if the manifest key changes. Define the
   composite key and file layout (for example
   `<target-config-digest>/<runtime-lock-digest>/probes-<sku>.md`) before T1
   acceptance.

3. **Profile package data is not in the `pyproject.toml` row.**
   The `profiles/*.json` row says "Package the profile data in the wheel", but
   the `pyproject.toml` responsibility only mentions the optional dependency
   set. Add `amora/backends/tpu/profiles/*.json` to
   `[tool.setuptools.package-data]` so installed wheels do not silently omit
   the catalog.

4. **`probe_groups.py` responsibility does not say whether TPU groups are
   registered.** The file is listed for "TPU routing" and the stale 30-probe
   comment, but the plan does not state whether TPU probes get curated groups
   or use the prefix fallback. Either is defensible; record the choice so the
   report layout is deterministic.

5. **T0 bootstrap acquisition is still implicit.**
   The host has no `pip`, `pip3`, `uv`, or `conda`, and `ensurepip` is absent.
   T0 says "default to a user-owned environment with a pinned bootstrap tool"
   and that sudo permits `python3-venv`, but it does not name how the tool is
   obtained and verified. State the default path (targeted
   `apt-get install python3-venv`, or a hash-checked standalone `uv` download)
   and record its SHA-256. The libtpu cp312 wheel exists, so Python 3.12 is
   viable once a bootstrap tool is present.

### Scope risk

G0 asks for sourced profiles for seven generations before the first release.
That is the right direction for the multiple-generation requirement, but it can
delay hardware qualification. Consider defining a minimum-viable G0: the
profile schema, the observed first-instance profile, unknown/conflict handling,
and synthetic fixtures. Add the remaining generation rows as source research
lands; the plan already permits catalog coverage without hardware-supported
claims. This is a scheduling recommendation, not a defect.

### Residual risk

The tracked plan still contains the complete SSH principal and public IP. The
review table records this as an optional publication decision; that is a
reasonable local choice, but it remains a publication risk if `.plan/` is
pushed to a shared remote.

### Recommendation

Accept r2 as the implementation baseline. Fix items 1-4 in the next plan edit
(or as explicit G0/T1 tasks), decide item 5 in T0, and start G0/T0.

## Revision History

| Revision | Date | Change |
|---|---|---|
| r1 | 2026-11-17 | Initial review of the r1 plan; verified repository contracts and portable-core counts, recorded five required changes, eleven design gaps, and minor corrections. |
| r2 | 2026-11-17 | Rechecked r2; verified all six fingerprints, the synthetic report derivation, host/egress supplements, and plan validation. Confirmed the r1 D3/D4/D11 corrections; recorded five consistency items and one scope risk. |
