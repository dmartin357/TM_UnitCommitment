"""
PreSolveConfig — tunable parameters for the pre-solve stage sampler.

This module implements a Monte Carlo commitment sampler.  For each sample it
applies the same iterative-cut stopping conditions as the Stage 1 GA initial-
population generator:

    1. new_pmax >= thermal_max_demand   (thermal_max = thermal_demand + reg_up_req)
    2. new_reg_up_potential  >= reg_up_req
    3. new_reg_down_potential >= reg_down_req

    where reg_up_potential_i  = min(ramp_up_limit_i,  pmax_i - pmin_i)
          reg_down_potential_i = min(ramp_down_limit_i, pmax_i - pmin_i)

No optimization (ED or otherwise) is performed; this is purely combinatorial
sampling designed to build a distribution of the number of units committed per
time period.  That distribution drives startup/shutdown targets for Stage 2.
"""

from dataclasses import dataclass


@dataclass
class PreSolveConfig:
    # ── Sampling ──────────────────────────────────────────────────────────────
    # Number of commitment samples to collect per time period.
    n_samples: int = 1_000

    # ── Generator ranking (for cut location) ─────────────────────────────────
    # Generators are sorted by this attribute; cuts are drawn uniformly from
    # the remaining committed set (matching Stage 1 'uniform' mode).
    sort_attribute: str = "power_output_maximum"
    sort_ascending: bool = False     # False → descending (largest first)

    # ── Target percentile ─────────────────────────────────────────────────────
    # Percentile of the n_committed distribution used as the period target when
    # computing startup/shutdown deltas between adjacent periods.
    target_percentile: float = 50.0  # 50.0 = median
