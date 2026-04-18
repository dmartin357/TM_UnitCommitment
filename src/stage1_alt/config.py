"""
AltStage1Config — tunable parameters for the alternative Stage 1 sampler.

This module implements a Monte Carlo commitment sampler.  Rather than running
a GA + ED solve per time period, it repeatedly samples feasible commitment
patterns by randomly cutting generators until power-capability thresholds are
violated.  The output is a per-unit commitment frequency for each time period.

Thresholds are defined relative to the non-renewable demand bounds:

    max_nonren_demand = expected_demand - min_forecasted_renewable
    min_nonren_demand = expected_demand - max_forecasted_renewable

    upper_threshold = max_nonren_demand × (1 + upper_tolerance)
    lower_threshold = max(0, min_nonren_demand × (1 − lower_tolerance))

The upper threshold is the minimum committed pmax the sampler will tolerate
(capacity reserve floor).  The lower threshold is the maximum committed pmin
the sampler will tolerate (must-run ceiling).

No optimization (ED or otherwise) is performed; this is purely combinatorial
sampling designed to estimate which units are reliably needed vs. marginal.
"""

from dataclasses import dataclass


@dataclass
class AltStage1Config:
    # ── Sampling ──────────────────────────────────────────────────────────────
    # Number of commitment samples to collect per time period.
    n_samples: int = 1_000

    # ── Stopping criteria thresholds ──────────────────────────────────────────
    # upper_tolerance: committed pmax must stay above max_nonren_demand × (1 + tol).
    # lower_tolerance: committed pmin must stay below max(0, min_nonren_demand × (1 − tol)).
    # 0.05 → ±5% margin relative to the non-renewable demand bounds.
    upper_tolerance: float = 0.05
    lower_tolerance: float = 0.05

    # ── Generator ranking (for cut location) ─────────────────────────────────
    # Generators are sorted by this attribute before the cut location is sampled.
    # Contiguous groups of generators in this sorted order are cut together.
    sort_attribute: str = "power_output_maximum"
    sort_ascending: bool = False     # False → descending (largest first)

    # ── Cut group size ────────────────────────────────────────────────────────
    # Each cut removes a contiguous slice of [cut_size_min, cut_size_max]
    # generators from the sorted list.  Clamped to available generators.
    cut_size_min: int = 1
    cut_size_max: int = 1

    # ── Cut group location distribution ──────────────────────────────────────
    # 'uniform' — starting index drawn uniformly from all valid positions.
    # 'end'     — starting index drawn with probability proportional to its
    #             distance from the beginning (biases cuts toward smaller
    #             generators at the end of the pmax-sorted list).
    location_dist_type: str = "uniform"

    # ── Renewable uncertainty ─────────────────────────────────────────────────
    # Reserved for future use (e.g., adjusting thresholds or cut-size bias).
    renewable_uncertainty: float = 0.0
