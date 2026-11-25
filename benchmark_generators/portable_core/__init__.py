# SPDX-License-Identifier: Apache-2.0
"""Portable core20/core24/core28 mathematical workloads; imports are lazy."""

__version__ = "0.1.0"
_CORE20_SNAPSHOT_SCHEMA = "cross-isa-core20-v1"


def families(tier=28, *, suite=None):
    if suite is not None:
        return tuple(dict.fromkeys(case.family for case in generate_cases(suite=suite)))
    if tier not in (20, 24, 28):
        raise ValueError("tier must be 20, 24 or 28")
    core = tuple([f"A{i:02}" for i in range(1, 9)] + [f"C{i:02}" for i in range(1, 9)] + [f"D{i:02}" for i in range(1, 5)])
    return core + tuple(f"N{i}" for i in range(21, tier + 1))


def describe(tier=28, *, suite=None):
    if suite is not None:
        from .expanded.v2.suites import PROFILES, coverage
        return {"benchmark_id": "portable_core", "revision": 2, "suite": suite,
                "families": list(families(suite=suite)), "profiles": list(PROFILES),
                "coverage": coverage(suite), "native_performance_qualified": False}
    return {"benchmark_id": "portable_core", "revision": 1, "tier": tier,
            "families": list(families(tier)), "profiles": ["smoke", "native"],
            "case_counts": {"smoke": {20: 152, 24: 168, 28: 184}[tier],
                            "native": {20: 304, 24: 336, 28: 368}[tier]},
            "license": "Apache-2.0", "native_performance_qualified": False,
            "scope": "portable mathematical programs; native execution/timing requires separate admission"}


def generate_cases(tier=28, profile="smoke", *, suite=None):
    if suite is not None:
        from .expanded.v2.suites import generate
        return generate(suite, profile)
    families(tier)
    if tier == 20:
        from .base.core import generate_cases as generate
        return generate(profile)
    from .expanded.core import generate_cases as generate
    return generate(tier, profile)


def make_inputs(case):
    from .expanded.v2.spec import Case
    if isinstance(case, Case):
        from .expanded.v2.execution import make_inputs as generate
    else:
        from .expanded.core import make_inputs as generate
    return generate(case)


def create_backend(name="numpy", device="cpu", *, suite=None):
    if suite is not None:
        from .expanded.v2.execution import Backend
        from .expanded.v2.suites import SUITES
        if suite not in SUITES:
            raise ValueError("unknown suite")
        return Backend(name, device)
    from .expanded.backends import create_backend as create
    return create(name, device)


def evaluate(case, backend, inputs=None):
    """Return a tuple of outputs for all tiers, including multi-output N24."""
    from .expanded.v2.spec import Case
    if isinstance(case, Case):
        from .expanded.v2.execution import Backend, execute_prepared, prepare
        if not isinstance(backend, Backend):
            raise ValueError("v2 cases require create_backend(..., suite=...)")
        return execute_prepared(prepare(case, backend, inputs))
    from .expanded.core import evaluate as execute
    return execute(case, backend, inputs)


def oracle(case, inputs):
    from .expanded.v2.spec import Case
    if isinstance(case, Case):
        from .expanded.v2.reference import oracle as reference
    else:
        from .expanded.core import oracle as reference
    return reference(case, inputs)


def freeze(output, tier=28, *, suite=None, profile=None):
    if suite is not None:
        from .expanded.v2.runner import freeze as materialize
        return materialize(output, suite, profile or "smoke")
    if profile is not None:
        raise ValueError("legacy freeze always emits both smoke and native profiles")
    from pathlib import Path
    families(tier)
    if tier == 20:
        from .base.runner import freeze as materialize
        return materialize(Path(output))
    from .expanded.runner import freeze as materialize
    return materialize(Path(output), tier)


def verify_snapshot(folder):
    from pathlib import Path
    import hashlib
    import json
    folder = Path(folder)
    receipt = json.loads((folder / "freeze_receipt.json").read_text())
    if receipt.get("schema") not in (None, _CORE20_SNAPSHOT_SCHEMA):
        from .expanded.v2.runner import verify
        return verify(folder)
    if receipt.get("tier", 20) == 20:
        from .base.runner import verify_runtime_sources
        verify_runtime_sources(folder)
    else:
        from .expanded.runner import verify_sources
        verify_sources(folder)
    for name, expected in receipt["files"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError("frozen artifact hash mismatch")
    return receipt
