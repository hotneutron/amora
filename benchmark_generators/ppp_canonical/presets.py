"""Named materialization parameters for the canonical PPP generator."""

from __future__ import annotations

from typing import Any


PRESETS: dict[str, dict[str, Any]] = {
    "h100_2500": {"case_count": 2500, "seed": 20260717},
    "h100_5600": {"case_count": 5600, "seed": 20260717},
    "v100_2500": {
        "case_count": 2500,
        "seed": 20260717,
        "exclude_kernels": ("megamoe_fp8",),
    },
}
