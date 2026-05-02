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


def renewable_expected(variable_ren: dict, t: int) -> float:
    """Midpoint of [pmin[t], pmax[t]] summed over variable renewables."""
    total = 0.0
    for gen in variable_ren.values():
        pmin_s = gen.get("power_output_minimum", [])
        pmax_s = gen.get("power_output_maximum", [])
        lo = float(pmin_s[t]) if t < len(pmin_s) else 0.0
        hi = float(pmax_s[t]) if t < len(pmax_s) else 0.0
        total += (lo + hi) / 2.0
    return total


def generate_cut_candidate(
    free_units: list[str],
    constrained_on: set[str],
    generators: dict,
    thermal_demand_target: float,
    rng: np.random.Generator,
) -> set[str]:
    """
    Generate one candidate committed thermal set via random cutting.

    Parameters
    ----------
    free_units            : thermal units eligible for cutting (not constrained).
    constrained_on        : thermal units that must remain ON.
    generators            : full thermal generator dict from instance JSON.
    thermal_demand_target : MW threshold — stop cutting when sum(pmax) would
                            fall below this value.
    rng                   : NumPy random Generator.

    Returns
    -------
    Set of committed thermal unit names (survivors ∪ constrained_on).
    """
    pool = list(free_units)
    rng.shuffle(pool)  # random cut order

    committed_free: set[str] = set(pool)

    def _sum_pmax(committed_free_set: set[str]) -> float:
        thermal_pmax = sum(
            float(generators[n]["piecewise_production"][-1]["mw"])
            for n in committed_free_set
        )
        constrained_pmax = sum(
            float(generators[n]["piecewise_production"][-1]["mw"])
            for n in constrained_on
        )
        return thermal_pmax + constrained_pmax

    for unit in pool:
        committed_free.discard(unit)
        if _sum_pmax(committed_free) < thermal_demand_target - 1e-6:
            committed_free.add(unit)  # restore — this cut would violate pmax
            break

    return committed_free | constrained_on
