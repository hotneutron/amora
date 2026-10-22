# Portable core kernel package — decision and verification

Status: **packaged and CPU-verified; native performance and predictor integration not claimed**

## Decision and source inputs

AMORA is a good destination for the reusable workload-definition layer: it already
owns `benchmark_generators/` separately from probes and measurement backends.
The package lives at `benchmark_generators/portable_core/`. Predictor fitting,
OOD gates and campaign evidence remain in Accorde, not in this package.

Source repository: `hotneutron/accorde`, base `7c85e754`, plus the exact separately
hashed uncommitted files in `tools/cross_isa_campaign/` and
`tools/cross_isa_expansion/`. Source/packaged hashes and the requests verbatim are
in `benchmark_generators/portable_core/PROVENANCE.json`. Machine-local request
text is kept in JSON, not used as an operational Markdown path.

## Changes and preserved boundaries

- Eight source modules copied with Apache-2.0/adaptation notices and relocated
  into package-local `base`/`expanded` namespaces; mathematical contracts unchanged.
- Public lazy API, tuple outputs and dedicated `amora-portable-core` CLI added.
- NumPy is a `portable-core` optional extra in the root project. Torch/JAX/NPU
  remain separately installed framework/runtime choices.
- Full Apache license, NOTICE and semantic fingerprints ship as package data.
  AMORA's root MIT license is unchanged and does not relicense imported components.
- Existing NVIDIA probes, benchmark registry/classification/detail pipeline and
  user-modified measurement/test files were not changed.
- No raw input banks, performance reports, checkpoints or Accorde runtime stores
  copied. Current Accorde snapshots are untouched historical records.
- Future workload changes should use this AMORA package/version as the canonical
  definition, not maintain two unpinned editable forks.

## Verification

- New package plus existing materialization tests: **19 passed**.
- All six tier/profile payload fingerprints match the source.
- NumPy/PyTorch CPU checks cover all184 core28 smoke cases and tuple/empty outputs.
- Import/discovery works with array-library imports blocked.
- Kernel-only wheel built:52,437 bytes, pure Python.
- Wheel SHA-256:
  `5c0e622a508980ed5793d62670e4f146add49f98fd5e2b45b57f7499579fa104`.
- Isolated wheel import and184 NumPy cases pass; license/provenance assets present.
- No native performance labels or model fits produced.

The first build hit a full external pip cache. A no-cache local build succeeded;
no user cache was deleted and no dependencies were downloaded.

## Use

From the repository root:

```bash
pip install -e '.[portable-core,test]'
python -m benchmark_generators.portable_core list --tier 28
python -m benchmark_generators.portable_core freeze --tier 28 --out out/portable_core/core28
python -m benchmark_generators.portable_core check \
  --frozen out/portable_core/core28 --out out/portable_core/core28-check \
  --backends numpy,torch,jax
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest tests/benchmarks/test_portable_core.py tests/benchmarks/test_materialize.py -q
```

The detached kernel-only wheel is under `out/portable_core_package/dist/`;
`out/portable_core_package/wheel_verification.json` records isolated verification.
Use either the AMORA checkout installation or the detached wheel in a clean
consumer environment; they expose the same namespace/CLI and should not be
installed as conflicting unpinned versions.

## Limits

This is portable mathematical execution, not a universal native-code package.
BF16-valued matrix operands are promoted for FP32 arithmetic in the Torch path;
native BF16 instructions are not thereby proved. Scan/atomics/carry/fusion,
TPU/Ascend devices and device-only timing require separate admission. The new
package is not automatically supported by the old CUDA-template NCU runner.

## Commit request and scope

User prompt, verbatim:

```text
commit the amora changes.
```

Package changes are committed independently from the pre-existing Gluon work:
`benchmark_generators/portable_core/`, `tests/benchmarks/test_portable_core.py`,
`pyproject.toml` and this plan only. The combined package/materialization/NCU/
stall-cycle focused suite passes70 tests. Raw traces and the separate exact-TMA
request are not part of this package commit. No push is requested.

## Revision History

|Revision|Timestamp|Change|
|---|---|---|
|r1|2026-10-22T01:14:17Z|Recorded source inputs,isolated workload-package fit,namespace/license changes,19-test and detached-wheel verification; rejected bulk campaign/model import and unqualified native integration; existing user work and root license preserved.|
|r2|2026-10-22T04:17:37Z|Preserved the commit request; scoped the portable package commit separately from Gluon work and raw/unrelated artifacts; recorded70 focused passing tests and no push.|
