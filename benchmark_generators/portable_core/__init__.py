# SPDX-License-Identifier: Apache-2.0
"""Portable core20/core24/core28 mathematical workloads; imports are lazy."""

__version__ = "0.1.0"


def families(tier=28):
    if tier not in (20, 24, 28):
        raise ValueError("tier must be 20, 24 or 28")
    core = tuple([f"A{i:02}" for i in range(1, 9)] + [f"C{i:02}" for i in range(1, 9)] + [f"D{i:02}" for i in range(1, 5)])
    return core + tuple(f"N{i}" for i in range(21, tier + 1))


def describe(tier=28):
    return {"benchmark_id": "portable_core", "revision": 1, "tier": tier,
            "families": list(families(tier)), "profiles": ["smoke", "native"],
            "case_counts": {"smoke": {20: 152, 24: 168, 28: 184}[tier],
                            "native": {20: 304, 24: 336, 28: 368}[tier]},
            "license": "Apache-2.0", "native_performance_qualified": False,
            "scope": "portable mathematical programs; native execution/timing requires separate admission"}


def generate_cases(tier=28, profile="smoke"):
    families(tier)
    if tier == 20:
        from .base.core import generate_cases as generate
        return generate(profile)
    from .expanded.core import generate_cases as generate
    return generate(tier, profile)


def make_inputs(case):
    from .expanded.core import make_inputs as generate
    return generate(case)


def create_backend(name="numpy", device="cpu"):
    from .expanded.backends import create_backend as create
    return create(name, device)


def evaluate(case, backend, inputs=None):
    """Return a tuple of outputs for all tiers, including multi-output N24."""
    from .expanded.core import evaluate as execute
    return execute(case, backend, inputs)


def oracle(case, inputs):
    from .expanded.core import oracle as reference
    return reference(case, inputs)


def freeze(output, tier=28):
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
