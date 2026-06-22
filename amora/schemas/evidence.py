"""Evidence labels used by probes and reports."""

from enum import Enum


class EvidenceTier(str, Enum):
    """How directly a value was observed."""

    PUBLISHED_FACT = "published_fact"
    DIRECT_METADATA = "direct_metadata"
    DIRECT_COUNTER = "direct_counter"
    TOOL_DERIVED_COUNTER = "tool_derived_counter"
    INSTRUMENTED_STREAM = "instrumented_stream"
    TIMING_DIRECT = "timing_direct"
    SIMULATOR_TRACE = "simulator_trace"
    COUPLED_INFERENCE = "coupled_inference"
    UNSUPPORTED = "unsupported"


class FitStatus(str, Enum):
    """Identifiability status for a measurement or estimate."""

    DIRECT = "direct"
    UNIQUELY_IDENTIFIED = "uniquely_identified"
    BOUNDED = "bounded"
    CONDITIONALLY_IDENTIFIED = "conditionally_identified"
    UNDERCONSTRAINED = "underconstrained"
    BEHAVIORAL_ONLY = "behavioral_only"
    UNSUPPORTED = "unsupported"


class UncertaintyCategory(str, Enum):
    """Compact uncertainty categories for report summaries."""

    STABLE_SCALAR = "stable_scalar"
    BOUNDED_RANGE = "bounded_range"
    CONDITIONAL_SCALAR = "conditional_scalar"
    MULTI_FIT = "multi_fit"
    BEHAVIORAL_CLASS = "behavioral_class"
    INDETERMINATE = "indeterminate"
