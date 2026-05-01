"""
Pre-solve stage — Monte Carlo commitment sampler (single time period).

Algorithm
---------
For each sample:
  1. Start with all generators committed.
  2. Randomly select one committed generator to cut.
  3. Check whether removing it would violate any stopping condition:
       • new_pmax              < thermal_max_demand   (pmax capacity floor)
       • new_reg_up_potential  < reg_up_req           (upward ramp reserve)
       • new_reg_down_potential < reg_down_req        (downward ramp reserve)
     where:
       thermal_max_demand    = thermal_demand + reg_up_req
       reg_up_potential_i    = min(ramp_up_limit_i,  pmax_i - pmin_i)
       reg_down_potential_i  = min(ramp_down_limit_i, pmax_i - pmin_i)
  4. If any condition would be violated: stop, record the committed set size.
  5. Otherwise: apply the cut and loop.

These are the same stopping conditions as the Stage 1 GA initial-population
generator, so the sampled chromosomes respect the same feasibility constraints.
No ED solve is performed; the goal is a distribution of n_committed per period.

Output
------
PreSolvePeriodResult holding:
  - per-unit commit counts (to compute frequency tables)
  - n_committed_samples list (to compute the n_committed distribution)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import PreSolveConfig


@dataclass
class PreSolvePeriodResult:
    """Statistics collected for one time period."""
    thermal_demand: float        # expected thermal demand (MW)
    reg_up_req: float            # MW of upward ramp the fleet must retain
    reg_down_req: float          # MW of downward ramp the fleet must retain
    total_samples: int
    commit_counts: dict[str, int]         # {unit_name: n_times_committed}
    n_committed_samples: list[int]        # n_committed for each sample
    wall_seconds: float

    # ── Per-unit frequency ────────────────────────────────────────────────────

    @property
    def commit_frequency(self) -> dict[str, float]:
        """Commitment frequency [0, 1] for each unit."""
        if self.total_samples == 0:
            return {n: 0.0 for n in self.commit_counts}
        return {n: c / self.total_samples for n, c in self.commit_counts.items()}

    # ── n_committed distribution ──────────────────────────────────────────────

    @property
    def mean_n_committed(self) -> float:
        if not self.n_committed_samples:
            return 0.0
        return float(np.mean(self.n_committed_samples))

    @property
    def median_n_committed(self) -> float:
        if not self.n_committed_samples:
            return 0.0
        return float(np.median(self.n_committed_samples))

    def percentile_n_committed(self, p: float) -> float:
        """Return the p-th percentile (0–100) of the n_committed distribution."""
        if not self.n_committed_samples:
            return 0.0
        return float(np.percentile(self.n_committed_samples, p))

    def target_n_committed(self, percentile: float = 50.0) -> int:
        """Round the target percentile to the nearest integer."""
        return round(self.percentile_n_committed(percentile))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _precompute(
    generators: dict,
    config: PreSolveConfig,
) -> tuple[list[str], dict[str, float], dict[str, float], dict[str, float]]:
    """
    Return (sorted_names, pmax, reg_up_potential, reg_down_potential).

    reg_up_potential_i  = min(ramp_up_limit_i,  pmax_i - pmin_i)
    reg_down_potential_i = min(ramp_down_limit_i, pmax_i - pmin_i)
    """
    sorted_names = sorted(
        generators.keys(),
        key=lambda n: generators[n].get(config.sort_attribute, 0.0),
        reverse=not config.sort_ascending,
    )
    pmax = {}
    reg_up = {}
    reg_down = {}
    for n, g in generators.items():
        p_max = float(g.get("power_output_maximum", 0.0))
        p_min = float(g.get("power_output_minimum", 0.0))
        headroom = p_max - p_min
        pmax[n]     = p_max
        reg_up[n]   = min(float(g.get("ramp_up_limit",   0.0)), headroom)
        reg_down[n] = min(float(g.get("ramp_down_limit", 0.0)), headroom)
    return sorted_names, pmax, reg_up, reg_down


def _run_one_sample(
    sorted_names: list[str],
    pmax: dict[str, float],
    reg_up_pot: dict[str, float],
    reg_down_pot: dict[str, float],
    cur_pmax: float,
    cur_reg_up: float,
    cur_reg_down: float,
    thermal_max_demand: float,
    reg_up_req: float,
    reg_down_req: float,
    rng: np.random.Generator,
) -> list[str]:
    """
    Run one cutting pass and return the list of committed generator names.

    Generators are removed one at a time in a uniformly random order.
    The first cut that would violate a stopping condition is rejected and
    the algorithm stops; all prior cuts are accepted.
    """
    committed = list(sorted_names)

    while committed:
        cut_idx = int(rng.integers(0, len(committed)))
        cut_gen = committed[cut_idx]

        new_pmax     = cur_pmax     - pmax[cut_gen]
        new_reg_up   = cur_reg_up   - reg_up_pot[cut_gen]
        new_reg_down = cur_reg_down - reg_down_pot[cut_gen]

        if (
            new_pmax     < thermal_max_demand
            or new_reg_up  < reg_up_req
            or new_reg_down < reg_down_req
        ):
            break

        committed.pop(cut_idx)
        cur_pmax     = new_pmax
        cur_reg_up   = new_reg_up
        cur_reg_down = new_reg_down

    return committed


# ── Public API ────────────────────────────────────────────────────────────────

def run_pre_solve_period(
    generators: dict,
    thermal_demand: float,
    reg_up_req: float,
    reg_down_req: float,
    config: PreSolveConfig,
    seed: int = 42,
) -> PreSolvePeriodResult:
    """
    Collect n_committed distribution statistics for one time period.

    Parameters
    ----------
    generators     : {name: gen_data} thermal generator dict (from pglib-uc JSON).
    thermal_demand : expected thermal demand (MW) = total_demand - renewable_expected.
    reg_up_req     : MW of upward ramp the committed fleet must retain.
    reg_down_req   : MW of downward ramp the committed fleet must retain.
    config         : PreSolveConfig.
    seed           : integer seed for this period's RNG.

    Returns
    -------
    PreSolvePeriodResult
    """
    t0 = time.monotonic()
    rng = np.random.default_rng(seed)

    sorted_names, pmax, reg_up_pot, reg_down_pot = _precompute(generators, config)

    total_pmax     = sum(pmax[n]         for n in sorted_names)
    total_reg_up   = sum(reg_up_pot[n]   for n in sorted_names)
    total_reg_down = sum(reg_down_pot[n] for n in sorted_names)

    thermal_max_demand = thermal_demand + reg_up_req

    commit_counts: dict[str, int] = {n: 0 for n in sorted_names}
    n_committed_samples: list[int] = []

    for _ in range(config.n_samples):
        committed = _run_one_sample(
            sorted_names, pmax, reg_up_pot, reg_down_pot,
            total_pmax, total_reg_up, total_reg_down,
            thermal_max_demand, reg_up_req, reg_down_req,
            rng,
        )
        n_committed_samples.append(len(committed))
        for name in committed:
            commit_counts[name] += 1

    return PreSolvePeriodResult(
        thermal_demand=thermal_demand,
        reg_up_req=reg_up_req,
        reg_down_req=reg_down_req,
        total_samples=config.n_samples,
        commit_counts=commit_counts,
        n_committed_samples=n_committed_samples,
        wall_seconds=time.monotonic() - t0,
    )
