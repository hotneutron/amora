# Portable-core v2 execution tracker

Plan: [H3 and Operation Gate v2 reconstruction](../20261123-1522-plan-portable-core-h3-operation-coverage.md), r2. Evidence: [source lock](../20261123-1522-portable-core-h3-operation-evidence.json). Baseline: `f67fb13b838cd96fd0c69783588c237b4e38c7fb`.

## Request

```text
execute the plan with a tracker
```

## Milestones

| ID | Status | Acceptance evidence or blocker |
|---|---|---|
| M0 | complete | Verified the 53 evidence blobs, original registry digest, 240 deterministic Torch witnesses, and report-only H3 blocker inventory. |
| M1 | complete | Typed v2 program/case/realization identities, typed port and state validation, prepared CPU adapters, independent FP64/scalar references, immutable snapshots, and fail-closed CLI dispatch are implemented. All six legacy fingerprints match. |
| M2 | complete for the pinned suites | Ported 240 Gate-v2 motifs/720 specs and 96 development motifs/288 specs. Passed 720 NumPy + 240 Torch required Gate-v2 checks and 288 development NumPy checks. Historic CUDA input-byte parity remains `not_verified`. The original-384 and O18 studies remain inventory references, not imported timings or new admitted cases. |
| M3 | controls complete; exact reference blocked | Passed 40 H3 semantic and access cases, 84 program cases (50 directed pair/size, 32 topology/shape, two fork/chain), and nine diagnostic cases under new IDs. Both H3 reference entrypoints return `blocked_missing_source`; Accorde owns approved source-only exports and case crosswalks. |
| M4 | local verification complete; Accorde sign-off pending | Built and smoke-tested the wheel, reproduced frozen manifests, checked the source-only consumer contract, and added a pinned CPU parity lane. No hosted CI was invoked. Accorde's acceptance in its own release environment remains separate. |
| M5 | not authorized | No native timing, fitting, measurement, gate transfer, or hardware campaign. |

## Validation and handoff

- `out/portable_core_v2/release-r2/release-cpu.json`: **1,141** case specs and
  **1,505 / 1,505** required CPU checks (NumPy plus pinned Torch 2.8.0).
- Focused v1/v2 suite: **33 passed**. Full local regression: **241 passed,
  16 skipped**. The first check found and fixed the rotary odd-column oracle
  and legacy core20 receipt dispatch; the failure receipts are retained locally.
- The offline importer reproduced the packaged source bundle byte-for-byte.
  The final wheel is under `out/portable_core_v2/final-wheel/`; its SHA-256 is
  `81e71acbd0d50eb0d61489d8da5cee9e5930a337bc92a49e8004144a24a13a3c`.
  A clean installed-wheel environment reproduced all five frozen manifests,
  validated their crosswalks with [verify_export.py](../../tools/portable_core_v2/verify_export.py),
  and rejected an incorrect independent manifest digest. No sibling repo was modified.
- Exact H3 requires an approved export with upstream case IDs, ordered graphs,
  shape/domain bindings, and input/oracle recipes. Historical input-byte parity
  and all scientific timing/fit/admission remain outside the verified scope.

## Revision History

| Revision | Date | Change |
|---|---|---|
| r1 | 2026-11-23 | Started execution of plan r2; recorded exact-source and hardware authorization boundaries. |
| r2 | 2026-11-23 | Recorded the v2 implementation, exact Gate-v2 spec and development corpus, complete source-available CPU matrix, legacy preservation, and wheel/preflight validation. Exact H3 and external consumer acceptance remain blocked/open. |
