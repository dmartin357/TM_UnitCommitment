"""
GraphBuilderConfig — tunable parameters for the Stage 2 graph builder.
"""

from dataclasses import dataclass


@dataclass
class GraphBuilderConfig:
    # ── Per-unit rectification ────────────────────────────────────────────────
    # When enabled, a ramp violation on a given unit is a candidate for
    # rectification if the violation falls within rectification_multiplier ×
    # the applicable ramp limit.  The power level is adjusted to the limit and
    # the edge is kept.  Violations larger than the threshold discard the edge.
    # When disabled, any ramp violation immediately discards the edge.
    enable_unit_rectification: bool = True
    rectification_multiplier: float = 2.0

    # ── Net adjustment tolerance ──────────────────────────────────────────────
    # When enabled, the cumulative net MW adjustment applied across all units
    # on a candidate edge must not exceed net_adjustment_tolerance × the
    # period's demand for both periods i and i+1.  Edges exceeding this bound
    # are discarded even if all per-unit ramp checks passed.
    # 0.05 → ±5% of demand.
    # Disable to see how many edges this check alone is eliminating.
    enable_net_adjustment_check: bool = True
    net_adjustment_tolerance: float = 0.05
