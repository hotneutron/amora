"""CUPTI integration placeholders.

The baseline NVIDIA implementation initially uses NCU-compatible logical
metrics and reports CUPTI support as a capability. Programmatic CUPTI Range
Profiling can be added behind this module without changing probe result schemas.
"""

from __future__ import annotations


def cupti_support_reason() -> str:
    return "CUPTI programmatic profiling is not implemented in the baseline cutline"
