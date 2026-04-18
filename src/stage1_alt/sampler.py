"""
Alternative Stage 1 — Monte Carlo commitment sampler (single time period).

Algorithm
---------
For each sample:
  1. Start with all generators committed.
  2. Repeatedly:
       a. Sample a cut-group size k ~ Uniform[cut_size_min, cut_size_max].
       b. Sample a cut-group start index from the sorted generator list.
       c. Test whether removing that group would violate either threshold:
            • sum(pmax, after) < upper_threshold  → reserve too low, stop
            • sum(pmin, after) > lower_threshold  → must-run too high, stop
       d. If a threshold would be violated: stop, record current committed set.
       e. Otherwise: apply the cut and loop.
  3. Increment per-unit counters for every committed unit in this sample.
  4. Repeat until n_samples samples are collected.

Thresholds are derived from the non-renewable demand bounds passed in by the
caller (computed from renewable forecast data):

    upper_threshold = demand_upper × (1 + upper_tolerance)
    lower_threshold = max(0, demand_lower × (1 − lower_tolerance))

where:
    demand_upper = expected_demand − min_forecasted_renewable
    demand_lower = expected_demand − max_forecasted_renewable

Output
------
AltStage1PeriodResult holding per-unit commit counts plus metadata.
Call .commit_frequency to get the [0, 1] frequency for each unit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .config import AltStage1Config


@dataclass
class AltStage1PeriodResult:
    """Statistics collected for one time period."""
    demand: float           # expected total demand (MW), for display only
    demand_upper: float     # max non-renewable demand = expected - min_renewable
    demand_lower: float     # min non-renewable demand = expected - max_renewable
    upper_threshold: float  # demand_upper × (1 + upper_tolerance)
    lower_threshold: float  # max(0, demand_lower × (1 − lower_tolerance))
    total_samples: int
    commit_counts: dict[str, int]   # {unit_name: n_times_committed}
    wall_seconds: float

    @property
    def commit_frequency(self) -> dict[str, float]:
        """Return commitment frequency [0, 1] for each unit."""
        if self.total_samples == 0:
            return {n: 0.0 for n in self.commit_counts}
        return {n: c / self.total_samples for n, c in self.commit_counts.items()}

    @property
    def mean_committed(self) -> float:
        """Average number of committed generators per sample."""
        if self.total_samples == 0:
            return 0.0
        return sum(self.commit_counts.values()) / self.total_samples


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sort_generators(generators: dict, config: AltStage1Config) -> list[str]:
    """Return generator names sorted by sort_attribute."""
    return sorted(
        generators.keys(),
        key=lambda n: generators[n].get(config.sort_attribute, 0.0),
        reverse=not config.sort_ascending,
    )


def _run_one_sample(
    sorted_names: list[str],
    pmaxes: dict[str, float],
    pmins: dict[str, float],
    total_pmax: float,
    total_pmin: float,
    upper_threshold: float,
    lower_threshold: float,
    config: AltStage1Config,
    rng: np.random.Generator,
) -> list[str]:
    """
    Run one sampling pass and return the list of committed generator names.

    The returned list preserves the original sort order for reproducibility.
    """
    committed = list(sorted_names)   # sorted order; all committed initially
    cur_pmax = total_pmax
    cur_pmin = total_pmin

    while True:
        n_avail = len(committed)
        if n_avail == 0:
            break

        # Sample cut group size (clamped to available units)
        hi = min(config.cut_size_max, n_avail)
        lo = min(config.cut_size_min, hi)
        cut_size = int(rng.integers(lo, hi + 1))

        # Sample cut start index
        max_start = n_avail - cut_size
        if max_start <= 0:
            start_idx = 0
        elif config.location_dist_type == "end":
            # Weight toward the end: p[i] ∝ i + 1  (larger index → more probable)
            weights = np.arange(1, max_start + 2, dtype=float)
            weights /= weights.sum()
            start_idx = int(rng.choice(max_start + 1, p=weights))
        else:  # "uniform" (default)
            start_idx = int(rng.integers(0, max_start + 1))

        # Compute what pmax/pmin would be after this cut
        cut_slice = committed[start_idx : start_idx + cut_size]
        new_pmax = cur_pmax - sum(pmaxes[n] for n in cut_slice)
        new_pmin = cur_pmin - sum(pmins[n] for n in cut_slice)

        # Check stopping criteria BEFORE applying the cut.
        # Stop if the cut would push either bound past its threshold:
        #   pmax < upper_threshold → not enough capacity to cover max thermal demand
        #   pmin < lower_threshold → minimum output would drop too low
        if new_pmax < upper_threshold or new_pmin < lower_threshold:
            break   # current committed set is this sample's result

        # Apply the cut
        del committed[start_idx : start_idx + cut_size]
        cur_pmax = new_pmax
        cur_pmin = new_pmin

    return committed


# ── Public API ────────────────────────────────────────────────────────────────

def run_alt_stage1_period(
    generators: dict,
    demand: float,
    demand_upper: float,
    demand_lower: float,
    config: AltStage1Config,
    seed: int = 42,
) -> AltStage1PeriodResult:
    """
    Collect commitment-frequency statistics for one time period.

    Parameters
    ----------
    generators   : {name: gen_data} thermal generator dict (from pglib-uc JSON).
    demand       : expected total demand (MW) — stored for display only.
    demand_upper : max non-renewable demand = expected_demand - min_renewable (MW).
    demand_lower : min non-renewable demand = expected_demand - max_renewable (MW).
    config       : AltStage1Config.
    seed         : integer seed for this period's RNG.

    Returns
    -------
    AltStage1PeriodResult
    """
    t0 = time.monotonic()
    rng = np.random.default_rng(seed)

    sorted_names = _sort_generators(generators, config)
    pmaxes = {n: float(generators[n].get("power_output_maximum", 0.0)) for n in sorted_names}
    pmins  = {n: float(generators[n].get("power_output_minimum",  0.0)) for n in sorted_names}

    total_pmax = sum(pmaxes.values())
    total_pmin = sum(pmins.values())

    upper_threshold = demand_upper * (1.0 + config.upper_tolerance)
    lower_threshold = max(0.0, demand_lower * (1.0 - config.lower_tolerance))

    commit_counts: dict[str, int] = {n: 0 for n in sorted_names}

    for _ in range(config.n_samples):
        committed = _run_one_sample(
            sorted_names, pmaxes, pmins,
            total_pmax, total_pmin,
            upper_threshold, lower_threshold,
            config, rng,
        )
        for name in committed:
            commit_counts[name] += 1

    wall = time.monotonic() - t0

    return AltStage1PeriodResult(
        demand=demand,
        demand_upper=demand_upper,
        demand_lower=demand_lower,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        total_samples=config.n_samples,
        commit_counts=commit_counts,
        wall_seconds=wall,
    )
