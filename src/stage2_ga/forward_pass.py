"""
Stage 2 GA — sequential forward pass.

Algorithm (per transition t → t+1)
------------------------------------
1. Candidates  : all chromosomes from the Stage 1 population for period t+1.
2. For each candidate (working copy — originals never modified):
     a. Repair   : enforce min up/down time constraints from the current fleet state.
     b. Mutate   : adjust n_committed toward the pre-solve target by flipping
                   eligible units uniformly at random.
     c. ED       : single-period economic dispatch for t+1 with per-unit ramp
                   bounds derived from t's dispatch.
3. Winner       : candidate with the lowest ED cost → becomes the new fleet state.

The loop runs for all n_periods−1 transitions.  Period 0 is the known initial
state (from instance t0 fields); the algorithm produces committed/dispatch
decisions for periods 1 through n_periods−1.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

import numpy as np

from ..stage1_ga.ed.piecewise_linear import EDInfeasible, solve_ed_piecewise_linear
from ..stage1_ga.population import BoundedPopulation
from .config import Stage2Config
from .repair import repair_min_updown
from .unit_state import FleetState, advance_fleet_state, fleet_state_from_t0

logger = logging.getLogger(__name__)


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class PeriodDecision:
    """The winning commitment/dispatch decision for one period."""
    period: int
    committed_names: list[str]
    dispatch: dict[str, float]      # {name: MW}
    ed_cost: float                  # production cost from ED (excludes startup costs)
    startup_cost: float             # sum of startup costs for units that turned on
    n_committed: int
    n_startups: int
    n_shutdowns: int
    n_candidates_evaluated: int
    n_ed_feasible: int
    n_ed_infeasible: int


@dataclass
class Stage2Result:
    """Full output of a Stage 2 forward pass."""
    decisions: list[PeriodDecision]   # one per period 1..n_periods-1
    generators: dict                  # reference to original generator dict

    @property
    def total_ed_cost(self) -> float:
        return sum(d.ed_cost for d in self.decisions)

    @property
    def total_startup_cost(self) -> float:
        return sum(d.startup_cost for d in self.decisions)

    @property
    def total_cost(self) -> float:
        return self.total_ed_cost + self.total_startup_cost

    def print_summary(self) -> None:
        n = len(self.decisions)
        print(f"\n{'=' * 100}")
        print(f"  Stage 2 GA — Forward Pass Summary  ({n} transitions)")
        print(f"{'=' * 100}")
        print(f"  {'t':>4}  {'Commit':>7}  {'Start':>6}  {'Shut':>5}  "
              f"{'ED Cost ($)':>14}  {'SU Cost ($)':>12}  "
              f"{'Candidates':>11}  {'EDFeas':>7}  {'EDInf':>6}")
        print(f"  {'-'*4}  {'-'*7}  {'-'*6}  {'-'*5}  "
              f"{'-'*14}  {'-'*12}  {'-'*11}  {'-'*7}  {'-'*6}")
        for d in self.decisions:
            print(
                f"  {d.period:>4}  {d.n_committed:>7}  {d.n_startups:>6}  "
                f"{d.n_shutdowns:>5}  "
                f"{d.ed_cost:>14,.2f}  {d.startup_cost:>12,.2f}  "
                f"{d.n_candidates_evaluated:>11}  {d.n_ed_feasible:>7}  "
                f"{d.n_ed_infeasible:>6}"
            )
        print(f"{'=' * 100}")
        print(f"  Total ED cost      : ${self.total_ed_cost:>18,.2f}")
        print(f"  Total startup cost : ${self.total_startup_cost:>18,.2f}")
        print(f"  Total cost         : ${self.total_cost:>18,.2f}")
        print(f"{'=' * 100}\n", flush=True)


# ── Ramp-bound computation ────────────────────────────────────────────────────

def _compute_ramp_bounds(
    committed_names: list[str],
    fleet_state: FleetState,
    generators: dict,
) -> dict[str, tuple[float, float]]:
    """
    Compute per-unit output bounds for t+1 based on t's dispatch and ramp limits.

    Continuing units (ON in t, ON in t+1):
      lb = max(pmin, dispatch_t - ramp_down_limit)
      ub = min(pmax, dispatch_t + ramp_up_limit)

    Startup units (OFF in t, ON in t+1):
      lb = pmin
      ub = min(pmax, ramp_startup_limit)
    """
    bounds = {}
    for name in committed_names:
        gen   = generators[name]
        state = fleet_state[name]
        pmin  = float(gen["piecewise_production"][0]["mw"])
        pmax  = float(gen["piecewise_production"][-1]["mw"])

        if state.committed:
            # Continuing — apply normal ramp limits around previous dispatch
            ramp_up   = float(gen.get("ramp_up_limit",   pmax - pmin))
            ramp_down = float(gen.get("ramp_down_limit", pmax - pmin))
            lb = max(pmin, state.dispatch - ramp_down)
            ub = min(pmax, state.dispatch + ramp_up)
        else:
            # Startup — first-period output capped by ramp_startup_limit
            su_ramp = float(gen.get("ramp_startup_limit", pmax))
            lb = pmin
            ub = min(pmax, su_ramp)

        bounds[name] = (lb, ub)
    return bounds


# ── Startup cost computation ──────────────────────────────────────────────────

def _startup_cost(
    name: str,
    fleet_state: FleetState,
    generators: dict,
) -> float:
    """
    Return the startup cost for a unit that was OFF and is turning ON.

    Startup cost is tiered by how long the unit has been offline (the 'lag'
    field in the startup list).  The cheapest tier whose lag ≤ time_down is used.
    """
    gen   = generators[name]
    state = fleet_state[name]
    tiers = gen.get("startup", [])
    if not tiers:
        return 0.0
    # Tiers are ordered by lag ascending; find the lowest cost whose lag is met
    cost = tiers[0]["cost"]   # default: first (cheapest/hottest) tier
    for tier in tiers:
        if state.time_in_state >= tier["lag"]:
            cost = tier["cost"]
        else:
            break
    return float(cost)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_stage2_forward_pass(
    populations: list[BoundedPopulation],
    generators: dict,
    thermal_demand_values: list[float],
    config: Stage2Config,
    rng: np.random.Generator | None = None,
) -> Stage2Result:
    """
    Run the Stage 2 sequential forward pass.

    For each transition t → t+1:
      1. Randomly shuffle the Stage 1 candidates for t+1.
      2. For each candidate in shuffled order:
           a. Repair min up/down time violations.
           b. Run ED with ramp bounds from t's dispatch.
           c. Accept the first feasible candidate as the winner.
      3. Advance the fleet state to the winner's commitment/dispatch.

    Parameters
    ----------
    populations           : Stage 1 BoundedPopulation for each time period.
                            Period 0 is not used (initial state is from t0 fields).
    generators            : full generator dict from instance JSON.
    thermal_demand_values : expected thermal demand per period (MW).
    config                : Stage2Config.
    rng                   : NumPy random Generator (created if None).

    Returns
    -------
    Stage2Result
    """
    if rng is None:
        rng = np.random.default_rng(config.rng_seed)

    n_periods   = len(thermal_demand_values)
    fleet_state: FleetState = fleet_state_from_t0(generators)
    decisions:   list[PeriodDecision] = []

    for t_next in range(1, n_periods):
        pop    = populations[t_next]
        demand = thermal_demand_values[t_next]

        all_chroms = list(pop)
        if not all_chroms:
            logger.warning("t=%d: no Stage 1 chromosomes available, skipping.", t_next)
            continue

        # Randomly shuffle candidates — first feasible one wins
        order = rng.permutation(len(all_chroms))
        candidates = [all_chroms[i] for i in order]

        gen_names      = candidates[0].gen_names
        prev_committed = {name for name, state in fleet_state.items() if state.committed}

        winner:         PeriodDecision | None = None
        n_ed_feasible   = 0
        n_ed_infeasible = 0

        for chrom in candidates:
            # Repair min up/down time violations
            bits = repair_min_updown(chrom.bits, gen_names, fleet_state, generators)

            committed_names = [name for name, b in zip(gen_names, bits) if b == 1]
            committed_set   = set(committed_names)

            # Augment demand to committed pmin if necessary
            committed_pmin = sum(
                float(generators[n]["piecewise_production"][0]["mw"])
                for n in committed_names
            )
            effective_demand = max(demand, committed_pmin)

            # Ramp bounds from previous period dispatch
            ramp_bounds = _compute_ramp_bounds(committed_names, fleet_state, generators)

            try:
                cost, dispatch = solve_ed_piecewise_linear(
                    committed_names=committed_names,
                    generators=generators,
                    demand=effective_demand,
                    solver=config.solver,
                    output_bounds=ramp_bounds,
                )
                n_ed_feasible += 1
            except EDInfeasible:
                n_ed_infeasible += 1
                continue

            startups  = committed_set - prev_committed
            shutdowns = prev_committed - committed_set
            su_cost   = sum(_startup_cost(n, fleet_state, generators) for n in startups)

            winner = PeriodDecision(
                period=t_next,
                committed_names=committed_names,
                dispatch=dispatch,
                ed_cost=cost,
                startup_cost=su_cost,
                n_committed=len(committed_names),
                n_startups=len(startups),
                n_shutdowns=len(shutdowns),
                n_candidates_evaluated=n_ed_feasible + n_ed_infeasible,
                n_ed_feasible=n_ed_feasible,
                n_ed_infeasible=n_ed_infeasible,
            )
            break   # first feasible candidate wins

        if winner is None:
            logger.error(
                "t=%d: all %d candidates were ED-infeasible — no winner found.",
                t_next, len(candidates),
            )
            continue

        decisions.append(winner)

        fleet_state = advance_fleet_state(
            fleet_state,
            committed_names=set(winner.committed_names),
            dispatch=winner.dispatch,
        )

        logger.info(
            "t=%d  committed=%d  startups=%d  shutdowns=%d  "
            "ed_cost=%.2f  su_cost=%.2f  tried=%d  ed_infeas=%d",
            t_next,
            winner.n_committed,
            winner.n_startups, winner.n_shutdowns,
            winner.ed_cost, winner.startup_cost,
            winner.n_candidates_evaluated, n_ed_infeasible,
        )

    return Stage2Result(decisions=decisions, generators=generators)
