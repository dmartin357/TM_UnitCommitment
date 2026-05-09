"""
GA v2 — single forward pass (one complete T-period UC solution).

Given a committed thermal set for period 1 (produced by cutting from the
t=0 initial state), this module runs a greedy forward pass through all
remaining periods and returns a ForwardSolution.

At each period t:
  1. Identify hydro (fixed dispatch) and variable renewables (wind/solar).
  2. Compute thermal demand target for the pmax violation check.
  3. Classify thermal units: constrained_ON / constrained_OFF / free.
  4. Generate n_candidates commitment sets via uniform random cutting.
  5. Run single-period ED (thermals + renewables) for each candidate.
  6. Score feasible candidates on weighted rank of (cost, reg range).
  7. Winner advances fleet state to the next period.
"""

from __future__ import annotations

import logging

import numpy as np

from ..stage1_ga.ed.piecewise_linear import EDInfeasible
from ..stage2_ga.unit_state import (
    FleetState,
    advance_fleet_state,
    fleet_state_from_t0,
)
from .candidate import ForwardSolution, PeriodCandidate
from .config import GAv2Config
from .constraint_check import classify_thermal_units
from .cutting import classify_renewables, generate_cut_candidate, renewable_contribution
from .ed_single_period import solve_ed_with_renewables
from .scoring import select_winner

logger = logging.getLogger(__name__)


# ── Ramp bounds (same logic as Stage 2) ──────────────────────────────────────

def _compute_ramp_bounds(
    committed_names: list[str],
    fleet_state: FleetState,
    generators: dict,
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for name in committed_names:
        gen   = generators[name]
        state = fleet_state[name]
        pmin  = float(gen.get("power_output_minimum", 0.0))
        pmax  = float(gen.get("power_output_maximum", 0.0))

        if state.committed:
            ramp_up   = float(gen.get("ramp_up_limit",   pmax - pmin))
            ramp_down = float(gen.get("ramp_down_limit", pmax - pmin))
            lb = max(pmin, state.dispatch - ramp_down)
            ub = min(pmax, state.dispatch + ramp_up)
        else:
            su_ramp = float(gen.get("ramp_startup_limit", pmax))
            lb = pmin
            ub = min(pmax, su_ramp)

        bounds[name] = (lb, ub)
    return bounds


# ── Startup cost (same logic as Stage 2) ─────────────────────────────────────

def _startup_cost(
    name: str,
    fleet_state: FleetState,
    generators: dict,
) -> float:
    gen   = generators[name]
    state = fleet_state[name]
    tiers = gen.get("startup", [])
    if not tiers:
        return 0.0
    cost = tiers[0]["cost"]
    for tier in tiers:
        if state.time_in_state >= tier["lag"]:
            cost = tier["cost"]
        else:
            break
    return float(cost)


# ── Period decision builder ───────────────────────────────────────────────────

def _evaluate_candidate(
    committed: set[str],
    t: int,
    fleet_state: FleetState,
    generators: dict,
    variable_ren: dict,
    effective_demand: float,
    prev_committed: set[str],
    config: GAv2Config,
) -> PeriodCandidate | None:
    """
    Run ED for a candidate committed set and return a PeriodCandidate, or None
    if the ED is infeasible.
    """
    ramp_bounds = _compute_ramp_bounds(list(committed), fleet_state, generators)

    startups  = committed - prev_committed
    shutdowns = prev_committed - committed
    su_cost   = sum(_startup_cost(n, fleet_state, generators) for n in startups)

    try:
        thermal_cost, thermal_dispatch, ren_dispatch, reg_up, reg_down = (
            solve_ed_with_renewables(
                committed_thermal_names=list(committed),
                thermal_generators=generators,
                renewable_gens=variable_ren,
                demand=effective_demand,
                t=t,
                ramp_bounds=ramp_bounds,
                renewable_cost_per_mwh=config.renewable_cost_per_mwh,
                solver=config.solver,
            )
        )
    except EDInfeasible:
        return None

    return PeriodCandidate(
        period=t,
        committed_names=list(committed),
        dispatch_thermal=thermal_dispatch,
        dispatch_renewable=ren_dispatch,
        thermal_cost=thermal_cost,
        startup_cost=su_cost,
        reg_up=reg_up,
        reg_down=reg_down,
        n_committed=len(committed),
        n_startups=len(startups),
        n_shutdowns=len(shutdowns),
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def build_forward_solution(
    t1_committed: set[str],
    generators: dict,
    renewable_gens: dict,
    total_demand: list[float],
    n_periods: int,
    config: GAv2Config,
    rng: np.random.Generator,
) -> ForwardSolution:
    """
    Build one complete T-period UC solution given a period-1 committed set.

    Parameters
    ----------
    t1_committed   : committed thermal unit names for period 1.
    generators     : thermal generator dict from instance JSON.
    renewable_gens : all renewable generators from instance JSON.
    total_demand   : total demand per period (0-indexed, period 0 = initial).
    n_periods      : total number of periods including period 0.
    config         : GAv2Config.
    rng            : NumPy random Generator.

    Returns
    -------
    ForwardSolution with one PeriodCandidate per period 1 .. n_periods-1.
    """
    fleet_state: FleetState = fleet_state_from_t0(generators)
    decisions: list[PeriodCandidate] = []

    # ── Period 1: use the provided t1_committed directly ─────────────────────
    t = 1
    hydro_fixed, variable_ren = classify_renewables(renewable_gens, t)
    hydro_total     = sum(hydro_fixed.values())
    effective_demand = total_demand[t] - hydro_total
    prev_committed   = {n for n, s in fleet_state.items() if s.committed}

    cand = _evaluate_candidate(
        committed=t1_committed,
        t=t,
        fleet_state=fleet_state,
        generators=generators,
        variable_ren=variable_ren,
        effective_demand=effective_demand,
        prev_committed=prev_committed,
        config=config,
    )

    if cand is None:
        logger.warning("t=1: provided t1 candidate is ED-infeasible — solution may be incomplete.")
    else:
        decisions.append(cand)
        fleet_state = advance_fleet_state(
            fleet_state,
            committed_names=t1_committed,
            dispatch=cand.dispatch_thermal,
        )
        logger.info(
            "t=1  committed=%d  startups=%d  shutdowns=%d  "
            "ed_cost=%.2f  su_cost=%.2f",
            cand.n_committed, cand.n_startups, cand.n_shutdowns,
            cand.thermal_cost, cand.startup_cost,
        )

    # ── Periods 2 .. n_periods-1 ─────────────────────────────────────────────
    for t in range(2, n_periods):
        hydro_fixed, variable_ren = classify_renewables(renewable_gens, t)
        hydro_total     = sum(hydro_fixed.values())
        effective_demand = total_demand[t] - hydro_total

        ren_contr = renewable_contribution(variable_ren, t, config.renewable_fraction)
        reg_up_req = total_demand[t] * config.reg_up_req_fraction
        thermal_demand_target = max(max(effective_demand - ren_contr, 0.0), reg_up_req)

        constrained_on, _constrained_off, free = classify_thermal_units(
            fleet_state, generators
        )
        prev_committed = {n for n, s in fleet_state.items() if s.committed}

        feasible: list[PeriodCandidate] = []
        n_infeasible = 0

        for _ in range(config.n_candidates_per_period):
            committed = generate_cut_candidate(
                free_units=list(free),
                constrained_on=constrained_on,
                generators=generators,
                thermal_demand_target=thermal_demand_target,
                rng=rng,
                fleet_state=fleet_state,
            )

            result = _evaluate_candidate(
                committed=committed,
                t=t,
                fleet_state=fleet_state,
                generators=generators,
                variable_ren=variable_ren,
                effective_demand=effective_demand,
                prev_committed=prev_committed,
                config=config,
            )

            if result is None:
                n_infeasible += 1
                logger.debug("t=%d  candidate INFEAS (total=%d)", t, n_infeasible)
            else:
                feasible.append(result)

        if not feasible:
            logger.error(
                "t=%d: all %d candidates ED-infeasible — period skipped.",
                t, config.n_candidates_per_period,
            )
            continue

        winner = select_winner(feasible, config.economics_weight, config.regulation_weight)
        decisions.append(winner)

        fleet_state = advance_fleet_state(
            fleet_state,
            committed_names=set(winner.committed_names),
            dispatch=winner.dispatch_thermal,
        )

        logger.info(
            "t=%d  committed=%d  startups=%d  shutdowns=%d  "
            "ed_cost=%.2f  su_cost=%.2f  feasible=%d/%d",
            t,
            winner.n_committed, winner.n_startups, winner.n_shutdowns,
            winner.thermal_cost, winner.startup_cost,
            len(feasible), config.n_candidates_per_period,
        )

    return ForwardSolution(decisions=decisions)
