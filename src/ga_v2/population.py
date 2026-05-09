"""
GA v2 — initial population generator.

Generates `config.n_population` t=1 candidate committed sets via random
cutting from the t=0 initial state, then builds one complete ForwardSolution
per t=1 candidate.  Returns the full list of solutions as the initial
population.
"""

from __future__ import annotations

import re
import time

import numpy as np

from ..stage2_ga.unit_state import fleet_state_from_t0
from .candidate import ForwardSolution
from .config import GAv2Config
from .constraint_check import classify_thermal_units
from .cutting import classify_renewables, generate_cut_candidate, renewable_contribution
from .forward_pass import build_forward_solution


def generate_initial_population(
    generators: dict,
    renewable_gens: dict,
    total_demand: list[float],
    n_periods: int,
    config: GAv2Config,
    rng: np.random.Generator,
) -> list[ForwardSolution]:
    """
    Generate the GA v2 initial population.

    Parameters
    ----------
    generators     : thermal generator dict from instance JSON.
    renewable_gens : all renewable generators from instance JSON.
    total_demand   : total demand per period (0-indexed).
    n_periods      : total number of periods including period 0.
    config         : GAv2Config.
    rng            : NumPy random Generator.

    Returns
    -------
    List of ForwardSolution (length = config.n_population).
    """
    t_start = time.perf_counter()

    fleet_state_t0 = fleet_state_from_t0(generators)

    # ── Classify thermal units at t=1 ─────────────────────────────────────────
    constrained_on, _constrained_off, free = classify_thermal_units(
        fleet_state_t0, generators
    )

    # ── Thermal demand target at t=1 ──────────────────────────────────────────
    t1 = 1
    hydro_fixed, variable_ren = classify_renewables(renewable_gens, t1)
    hydro_total    = sum(hydro_fixed.values())
    ren_contr    = renewable_contribution(variable_ren, t1, config.renewable_fraction)
    reg_up_req   = total_demand[t1] * config.reg_up_req_fraction
    th_demand_t1 = max(max(total_demand[t1] - hydro_total - ren_contr, 0.0), reg_up_req)

    print(
        f"  Thermal demand target at t=1: {th_demand_t1:,.1f} MW  "
        f"(demand={total_demand[t1]:,.1f}  hydro={hydro_total:,.1f}  "
        f"ren_contr={ren_contr:,.1f}  reg_up_floor={reg_up_req:,.1f})"
    )
    print(
        f"  Constraint buckets at t=1:  "
        f"constrained_ON={len(constrained_on)}  "
        f"constrained_OFF={len(_constrained_off)}  "
        f"free={len(free)}"
    )

    # ── Generate n_population t=1 candidates ──────────────────────────────────
    print(f"\n  Generating {config.n_population} t=1 candidates via cutting...")
    t1_candidates: list[set[str]] = []
    for _ in range(config.n_population):
        committed = generate_cut_candidate(
            free_units=list(free),
            constrained_on=constrained_on,
            generators=generators,
            thermal_demand_target=th_demand_t1,
            rng=rng,
            fleet_state=fleet_state_t0,
        )
        t1_candidates.append(committed)

    committed_counts = [len(c) for c in t1_candidates]
    print(
        f"  t=1 candidates: committed count  "
        f"min={min(committed_counts)}  max={max(committed_counts)}  "
        f"avg={sum(committed_counts)/len(committed_counts):.1f}\n"
    )

    # ── Build one ForwardSolution per t=1 candidate ───────────────────────────
    solutions: list[ForwardSolution] = []
    for i, t1_committed in enumerate(t1_candidates):
        t_sol_start = time.perf_counter()
        print(
            f"  [{i+1:>3}/{config.n_population}]  "
            f"t=1 committed={len(t1_committed):>3}  … ",
            end="",
            flush=True,
        )
        sol = build_forward_solution(
            t1_committed=t1_committed,
            generators=generators,
            renewable_gens=renewable_gens,
            total_demand=total_demand,
            n_periods=n_periods,
            config=config,
            rng=rng,
        )
        elapsed = time.perf_counter() - t_sol_start
        print(
            f"total_cost=${sol.total_cost:>14,.2f}  "
            f"({elapsed:.1f}s)"
        )
        solutions.append(sol)

    total_elapsed = time.perf_counter() - t_start
    costs = sorted(s.total_cost for s in solutions)
    print(
        f"\n  Population summary ({config.n_population} solutions, "
        f"{total_elapsed:.1f}s total):"
    )
    print(f"    Best cost  : ${costs[0]:>14,.2f}")
    print(f"    Worst cost : ${costs[-1]:>14,.2f}")
    print(f"    Median cost: ${costs[len(costs)//2]:>14,.2f}")

    return solutions
