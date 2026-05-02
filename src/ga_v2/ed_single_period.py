"""
Single-period Economic Dispatch for GA v2.

Extends the Stage 1 piecewise-linear thermal ED with variable renewable
dispatch (wind, solar, CSP — anything that is NOT hydro).

Model
-----
  min   sum_{g in G_c}  cost_g(p_g)  +  sum_{w in W}  c_ren * pw_w
  s.t.  sum_{g} p_g  +  sum_{w} pw_w  =  demand     (balance)
        lb_g <= p_g <= ub_g             ∀ g          (ramp bounds)
        pmin_w[t] <= pw_w <= pmax_w[t]  ∀ w          (renewable bounds)

where:
  - demand  = total_demand[t] − sum(hydro fixed output)  (hydro is pre-subtracted)
  - lb_g, ub_g come from ramp_bounds (Stage 2-style ramp constraint enforcement)
  - c_ren = renewable_cost_per_mwh (default $0.01/MWh — tiny, keeps renewables
            at their upper bound without creating degenerate LP columns)

Returns
-------
(thermal_cost, thermal_dispatch, renewable_dispatch, reg_up, reg_down)

  thermal_cost       : float — objective contribution from thermals only
  thermal_dispatch   : dict[str, float] — {name: MW} for each committed thermal
  renewable_dispatch : dict[str, float] — {name: MW} for each variable renewable
  reg_up             : float — sum(ramp_ub − dispatch) across committed thermals
  reg_down           : float — sum(dispatch − ramp_lb) across committed thermals

Raises EDInfeasible if the LP has no feasible solution.
"""

from __future__ import annotations

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

from ..stage1_ga.ed.piecewise_linear import (
    EDInfeasible,
    _parse_segments,
    _fixed_cost_at_pmin,
    _resolve_solver,
)


def solve_ed_with_renewables(
    committed_thermal_names: list[str],
    thermal_generators: dict,
    renewable_gens: dict,
    demand: float,
    t: int,
    ramp_bounds: dict[str, tuple[float, float]] | None = None,
    renewable_cost_per_mwh: float = 0.01,
    solver: str = "auto",
) -> tuple[float, dict[str, float], dict[str, float], float, float]:
    """
    Solve single-period ED with piecewise-linear thermals + linear renewables.

    Parameters
    ----------
    committed_thermal_names : thermal generators that are ON this period.
    thermal_generators      : full thermal generator dict from instance JSON.
    renewable_gens          : non-hydro renewable generators for this period
                              (empty dict → thermal-only ED).
    demand                  : MW to be served (total demand minus hydro fixed).
    t                       : 0-indexed time period (for renewable bounds lookup).
    ramp_bounds             : optional per-thermal (lb, ub) from ramp constraints;
                              None → standard [pmin, pmax].
    renewable_cost_per_mwh  : $/MWh cost assigned to variable renewables.
    solver                  : 'auto', 'gurobi', 'cplex', or 'cbc'.

    Returns
    -------
    (thermal_cost, thermal_dispatch, renewable_dispatch, reg_up, reg_down)
    """
    # ── Pre-parse thermal parameters ──────────────────────────────────────────
    thermal_params: dict[str, dict] = {}
    total_th_pmin = 0.0
    total_th_pmax = 0.0
    effective_ramp_bounds: dict[str, tuple[float, float]] = {}

    for name in committed_thermal_names:
        pmin, pmax, segs = _parse_segments(thermal_generators[name])
        c0 = _fixed_cost_at_pmin(thermal_generators[name])

        if ramp_bounds and name in ramp_bounds:
            lb, ub = ramp_bounds[name]
            pmin = max(pmin, lb)
            pmax = min(pmax, ub)

        effective_ramp_bounds[name] = (pmin, pmax)
        thermal_params[name] = {"pmin": pmin, "pmax": pmax, "segs": segs, "c0": c0}
        total_th_pmin += pmin
        total_th_pmax += pmax

    # ── Pre-parse renewable parameters ───────────────────────────────────────
    ren_params: dict[str, dict] = {}
    total_ren_pmin = 0.0
    total_ren_pmax = 0.0

    for name, gen in renewable_gens.items():
        pmin_s = gen.get("power_output_minimum", [])
        pmax_s = gen.get("power_output_maximum", [])
        rp_min = float(pmin_s[t]) if t < len(pmin_s) else 0.0
        rp_max = float(pmax_s[t]) if t < len(pmax_s) else 0.0
        ren_params[name] = {"pmin": rp_min, "pmax": rp_max}
        total_ren_pmin += rp_min
        total_ren_pmax += rp_max

    # ── Feasibility pre-check ─────────────────────────────────────────────────
    combined_pmin = total_th_pmin + total_ren_pmin
    combined_pmax = total_th_pmax + total_ren_pmax

    # Augment demand if committed pmin exceeds demand (don't fail — augment)
    effective_demand = max(demand, combined_pmin)

    if effective_demand > combined_pmax + 1e-6:
        raise EDInfeasible(
            f"demand={effective_demand:.1f}  combined_pmax={combined_pmax:.1f}  "
            f"(th={total_th_pmax:.1f}  ren={total_ren_pmax:.1f})"
        )

    # ── Build and solve Pyomo model ───────────────────────────────────────────
    model = _build_model(
        committed_thermal_names,
        thermal_params,
        list(ren_params.keys()),
        ren_params,
        effective_demand,
        renewable_cost_per_mwh,
    )

    solver_name = _resolve_solver(solver)
    opt = pyo.SolverFactory(solver_name)
    result = opt.solve(model, tee=False)

    status    = result.solver.status
    term_cond = result.solver.termination_condition
    if (status != SolverStatus.ok or
            term_cond not in (TerminationCondition.optimal,
                              TerminationCondition.locallyOptimal)):
        raise EDInfeasible(
            f"Solver returned status={status}, termination={term_cond}."
        )

    # ── Extract thermal dispatch ──────────────────────────────────────────────
    thermal_dispatch: dict[str, float] = {}
    thermal_cost = 0.0
    for name in committed_thermal_names:
        params = thermal_params[name]
        delta_sum = sum(
            pyo.value(model.delta[name, k])
            for k in range(len(params["segs"]))
        )
        mw = params["pmin"] + delta_sum
        thermal_dispatch[name] = mw
        thermal_cost += params["c0"] + sum(
            params["segs"][k][1] * pyo.value(model.delta[name, k])
            for k in range(len(params["segs"]))
        )

    # ── Extract renewable dispatch ────────────────────────────────────────────
    renewable_dispatch: dict[str, float] = {}
    for name in ren_params:
        renewable_dispatch[name] = float(pyo.value(model.pw[name]))

    # ── Regulation range (thermal only) ──────────────────────────────────────
    reg_up   = sum(
        effective_ramp_bounds[n][1] - thermal_dispatch[n]
        for n in committed_thermal_names
    )
    reg_down = sum(
        thermal_dispatch[n] - effective_ramp_bounds[n][0]
        for n in committed_thermal_names
    )

    return thermal_cost, thermal_dispatch, renewable_dispatch, reg_up, reg_down


# ── Pyomo model builder ───────────────────────────────────────────────────────

def _build_model(
    thermal_names: list[str],
    thermal_params: dict,
    renewable_names: list[str],
    ren_params: dict,
    demand: float,
    renewable_cost_per_mwh: float,
) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()

    # ── Thermal: piecewise-linear delta variables ─────────────────────────────
    seg_indices = [
        (name, k)
        for name in thermal_names
        for k in range(len(thermal_params[name]["segs"]))
    ]
    m.GK = pyo.Set(initialize=seg_indices, dimen=2)
    m.delta = pyo.Var(m.GK, within=pyo.NonNegativeReals)

    def _seg_ub(model, name, k):
        dmw, _ = thermal_params[name]["segs"][k]
        return model.delta[name, k] <= dmw

    m.seg_ub = pyo.Constraint(m.GK, rule=_seg_ub)

    def _total_delta_ub(model, name):
        cap = thermal_params[name]["pmax"] - thermal_params[name]["pmin"]
        return sum(
            model.delta[name, k]
            for k in range(len(thermal_params[name]["segs"]))
        ) <= cap

    m.G = pyo.Set(initialize=thermal_names)
    m.total_delta_ub = pyo.Constraint(m.G, rule=_total_delta_ub)

    # ── Renewable variables ───────────────────────────────────────────────────
    m.W = pyo.Set(initialize=renewable_names)
    m.pw = pyo.Var(m.W, within=pyo.NonNegativeReals)

    for name in renewable_names:
        m.pw[name].setlb(ren_params[name]["pmin"])
        m.pw[name].setub(ren_params[name]["pmax"])

    # ── Demand balance ────────────────────────────────────────────────────────
    def _demand_bal(model):
        thermal_total = sum(
            thermal_params[name]["pmin"]
            + sum(
                model.delta[name, k]
                for k in range(len(thermal_params[name]["segs"]))
            )
            for name in thermal_names
        )
        ren_total = sum(model.pw[name] for name in renewable_names)
        return thermal_total + ren_total == demand

    m.demand_bal = pyo.Constraint(rule=_demand_bal)

    # ── Objective: thermal piecewise cost + cheap renewable cost ─────────────
    def _obj(model):
        thermal_cost = sum(
            thermal_params[name]["c0"]
            + sum(
                thermal_params[name]["segs"][k][1] * model.delta[name, k]
                for k in range(len(thermal_params[name]["segs"]))
            )
            for name in thermal_names
        )
        ren_cost = renewable_cost_per_mwh * sum(model.pw[name] for name in renewable_names)
        return thermal_cost + ren_cost

    m.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

    return m
