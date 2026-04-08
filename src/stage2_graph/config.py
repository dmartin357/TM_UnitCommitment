"""
GraphBuilderConfig — tunable parameters for the Stage 2 graph builder.
"""

from dataclasses import dataclass


@dataclass
class GraphBuilderConfig:
    # ── Rectification ─────────────────────────────────────────────────────────
    # A ramp violation on a given unit is a candidate for rectification if the
    # actual ramp falls within this multiplier × the applicable ramp limit.
    # E.g. 2.0 → allow attempts to fix violations up to 2× the ramp limit.
    # Violations larger than this immediately discard the edge.
    rectification_multiplier: float = 2.0

    # ── Net adjustment tolerance ──────────────────────────────────────────────
    # After processing all units on a candidate edge, the cumulative net MW
    # adjustment applied to each period's dispatch must not exceed this fraction
    # of that period's demand.  Edges exceeding the tolerance for either period
    # are discarded.
    # 0.05 → ±5 % of demand.
    net_adjustment_tolerance: float = 0.05
