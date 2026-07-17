"""Named materialization parameters for the canonical PPP generator."""

from __future__ import annotations


PRESETS: dict[str, dict[str, int]] = {
    "h100_2500": {"case_count": 2500, "seed": 20260717},
    "h100_5600": {"case_count": 5600, "seed": 20260717},
}
