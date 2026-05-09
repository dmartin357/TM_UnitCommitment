"""
Candidate commitment generation via uniform random cutting.

Starting with all free units committed (plus the constrained_ON set),
the algorithm randomly removes free units one at a time until removing
the next unit would cause sum(pmax of remaining committed) to fall below
the thermal demand target.  The unit that would cause the violation is
kept, and cutting stops.

This mirrors the Stage 1 seeding philosophy but without the CDF weighting:
cuts are chosen uniformly at random from the remaining committed free pool.
"""

from __future__ import annotations

import re

import numpy as np

from ..stage2_ga.unit_state import FleetState


def classify_renewables(
    renewable_gens: dict,
    t: int,
) -> tuple[dict[str, float], dict[str, dict]]:
    """
    Separate all renewable generators into hydro (fixed) and variable (wind/solar).

    Returns
    -------
    hydro_fixed  : {name: MW} — fixed dispatch at period t (pmin = pmax for hydro)
    variable_ren : {name: gen_data} — wind/solar/CSP with [pmin[t], pmax[t]]
    """
    hydro_fixed: dict[str, float] = {}
    variable_ren: dict[str, dict] = {}

    for name, gen in renewable_gens.items():
        pmax_s = gen.get("power_output_maximum", [])
        pmax_t = float(pmax_s[t]) if t < len(pmax_s) else 0.0
        if re.search(r"HYDRO", name, re.IGNORECASE):
            hydro_fixed[name] = pmax_t
        else:
            variable_ren[name] = gen

    return hydro_fixed, variable_ren


def renewable_contribution(variable_ren: dict, t: int, fraction: float = 1.0) -> float:
    """Expected renewable contribution: fraction * pmax[t] summed over variable renewables."""
    total = 0.0
    for gen in variable_ren.values():
        pmax_s = gen.get("power_output_maximum", [])
        hi = float(pmax_s[t]) if t < len(pmax_s) else 0.0
        total += hi * fraction
    return total


def generate_cut_candidate(
    free_units: list[str],
    constrained_on: set[str],
    generators: dict,
    thermal_demand_target: float,
    rng: np.random.Generator,
    fleet_state: FleetState | None = None,
) -> set[str]:
    """
    Generate one candidate committed thermal set via random cutting.

    Parameters
    ----------
    free_units            : thermal units eligible for cutting (not constrained).
    constrained_on        : thermal units that must remain ON.
    generators            : full thermal generator dict from instance JSON.
    thermal_demand_target : MW threshold — stop cutting when sum(effective pmax)
                            would fall below this value.
    rng                   : NumPy random Generator.
    fleet_state           : Current fleet state.  When provided, each unit's
                            contribution to the pmax sum is ramp-adjusted:
                              - was ON : min(pmax, prev_dispatch + ramp_up_limit)
                              - was OFF : min(pmax, ramp_startup_limit)
                            Defaults to static pmax when None (e.g., testing).

    Returns
    -------
    Set of committed thermal unit names (survivors ∪ constrained_on).
    """
    pool = list(free_units)
    rng.shuffle(pool)

    committed_free: set[str] = set(pool)

    def _eff_pmax(name: str) -> float:
        gen  = generators[name]
        pmax = float(gen.get("power_output_maximum", 0.0))
        if fleet_state is None:
            return pmax
        pmin  = float(gen.get("power_output_minimum", 0.0))
        state = fleet_state[name]
        if state.committed:
            ramp_up = float(gen.get("ramp_up_limit", pmax - pmin))
            return min(pmax, state.dispatch + ramp_up)
        else:
            su_ramp = float(gen.get("ramp_startup_limit", pmax))
            return min(pmax, su_ramp)

    def _sum_pmax(committed_free_set: set[str]) -> float:
        return sum(_eff_pmax(n) for n in committed_free_set | constrained_on)

    for unit in pool:
        committed_free.discard(unit)
        if _sum_pmax(committed_free) < thermal_demand_target - 1e-6:
            committed_free.add(unit)  # restore — this cut would violate capacity
            break

    return committed_free | constrained_on
