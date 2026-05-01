"""
Economic Dispatch with piecewise-linear production costs.

Formulation
-----------
Given a set of committed generators G_c and a demand D, solve:

    min   sum_{i in G_c}  [ cost_i(p_i) ]
    s.t.  sum_{i in G_c}  p_i  =  D
          pmin_i  <=  p_i  <=  pmax_i    for all i in G_c

where cost_i is a convex piecewise-linear function defined by the
`piecewise_production` breakpoints in the instance data:

    cost_i(p_i) = cost_{i,0}  +  sum_k  mc_{i,k} * delta_{i,k}
    p_i         = pmin_i       +  sum_k  delta_{i,k}
    0           <= delta_{i,k} <= mw_{i,k+1} - mw_{i,k}

The incremental (delta) formulation is used because it is a standard LP with
no SOS2 constraints needed (convexity guarantees that LP will fill segments in
order).

Returns
-------
(total_cost, dispatch)
  total_cost : float   — optimal objective value (includes cost at pmin for
                         each committed generator)
  dispatch   : dict    — {gen_name: power_output_MW}

Raises EDInfeasible if demand cannot be met with the committed generators.
"""

from __future__ import annotations

import math

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition


class EDInfeasible(Exception):
    """Raised when the ED has no feasible solution for this chromosome."""


# ── Data preprocessing ────────────────────────────────────────────────────────

def _parse_segments(gen_data: dict) -> tuple[float, float, list[tuple[float, float]]]:
    """
    Return (pmin, pmax, segments) where each segment is (delta_ub, marginal_cost).

    pmin = first breakpoint MW value
    pmax = last  breakpoint MW value
    segments = [(capacity_MW, $/MW), ...] for each linear piece
    """
    pts = gen_data["piecewise_production"]
    pmin = pts[0]["mw"]
    pmax = pts[-1]["mw"]
    segments = []
    for i in range(len(pts) - 1):
        dmw   = pts[i + 1]["mw"]   - pts[i]["mw"]
        dcost = pts[i + 1]["cost"] - pts[i]["cost"]
        if dmw <= 0:
            continue
        mc = dcost / dmw
        segments.append((dmw, mc))
    return pmin, pmax, segments


def _fixed_cost_at_pmin(gen_data: dict) -> float:
    """Cost incurred when the generator runs at its minimum output."""
    return gen_data["piecewise_production"][0]["cost"]


# ── Solver detection ──────────────────────────────────────────────────────────

_SOLVER_PRECEDENCE = ["gurobi", "cplex", "cbc"]


def _resolve_solver(preference: str) -> str:
    if preference != "auto":
        return preference
    for name in _SOLVER_PRECEDENCE:
        if pyo.SolverFactory(name).available():
            return name
    raise RuntimeError(
        "No LP solver found. Install Gurobi, CPLEX, or CBC (glpk also works)."
    )


# ── Main solver function ──────────────────────────────────────────────────────

def solve_ed_piecewise_linear(
    committed_names: list[str],
    generators: dict,
    demand: float,
    solver: str = "auto",
    output_bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Solve piecewise-linear ED for the given committed generators and demand.

    Parameters
    ----------
    committed_names : list of generator names that are committed (bit=1).
    generators      : full generator dict {name: gen_data} from instance JSON.
    demand          : total MW demand for this time period.
    solver          : 'auto', 'gurobi', 'cplex', or 'cbc'.
    output_bounds   : optional per-unit (lb, ub) tighter than [pmin, pmax].
                      Used by Stage 2 to enforce ramp constraints from the
                      previous period's dispatch.  None → standard [pmin, pmax].

    Returns
    -------
    (total_cost, dispatch) where dispatch = {name: MW}.

    Raises EDInfeasible if the LP is infeasible or unbounded.
    """
    if not committed_names:
        if demand <= 0:
            return 0.0, {}
        raise EDInfeasible("No generators committed but demand > 0.")

    # Pre-parse generator data
    gen_params: dict[str, dict] = {}
    total_pmin = 0.0
    total_pmax = 0.0
    for name in committed_names:
        pmin, pmax, segs = _parse_segments(generators[name])
        c0 = _fixed_cost_at_pmin(generators[name])
        # Apply optional tighter bounds (ramp constraints from Stage 2)
        if output_bounds and name in output_bounds:
            lb, ub = output_bounds[name]
            pmin = max(pmin, lb)
            pmax = min(pmax, ub)
        gen_params[name] = {"pmin": pmin, "pmax": pmax, "segs": segs, "c0": c0}
        total_pmin += pmin
        total_pmax += pmax

    # Quick feasibility check before building the model
    if demand < total_pmin - 1e-6:
        raise EDInfeasible(
            f"Demand ({demand:.1f} MW) < sum of pmin ({total_pmin:.1f} MW). "
            "Too many generators committed."
        )
    if demand > total_pmax + 1e-6:
        raise EDInfeasible(
            f"Demand ({demand:.1f} MW) > sum of pmax ({total_pmax:.1f} MW). "
            "Insufficient capacity."
        )

    # Build Pyomo model
    model = _build_model(committed_names, gen_params, demand)

    # Solve
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

    # Extract solution
    total_cost = pyo.value(model.obj)
    dispatch = {}
    for name in committed_names:
        pmin = gen_params[name]["pmin"]
        delta_sum = sum(
            pyo.value(model.delta[name, k])
            for k in range(len(gen_params[name]["segs"]))
        )
        dispatch[name] = pmin + delta_sum

    return total_cost, dispatch


# ── Model builder ─────────────────────────────────────────────────────────────

def _build_model(
    committed_names: list[str],
    gen_params: dict,
    demand: float,
) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()

    # Sets
    m.G = pyo.Set(initialize=committed_names)

    # Segment index set per generator: {(name, k)}
    seg_indices = [
        (name, k)
        for name in committed_names
        for k in range(len(gen_params[name]["segs"]))
    ]
    m.GK = pyo.Set(initialize=seg_indices, dimen=2)

    # Variables: delta_{name, k} = incremental output on segment k
    m.delta = pyo.Var(m.GK, within=pyo.NonNegativeReals)

    # Segment upper bounds — also cap total delta at (pmax - pmin) per unit
    def _seg_ub(model, name, k):
        dmw, _mc = gen_params[name]["segs"][k]
        return model.delta[name, k] <= dmw

    m.seg_ub = pyo.Constraint(m.GK, rule=_seg_ub)

    # Per-unit total-delta upper bound: sum(delta_k) <= pmax - pmin
    # (pmax may have been tightened by output_bounds; this enforces it)
    def _total_delta_ub(model, name):
        cap = gen_params[name]["pmax"] - gen_params[name]["pmin"]
        return sum(model.delta[name, k] for k in range(len(gen_params[name]["segs"]))) <= cap

    m.total_delta_ub = pyo.Constraint(m.G, rule=_total_delta_ub)

    # Demand balance — uses effective pmin per unit
    def _demand_bal(model):
        total = sum(
            gen_params[name]["pmin"]
            + sum(model.delta[name, k] for k in range(len(gen_params[name]["segs"])))
            for name in committed_names
        )
        return total == demand

    m.demand_bal = pyo.Constraint(rule=_demand_bal)

    # Objective: minimize total production cost
    def _obj(model):
        return sum(
            gen_params[name]["c0"]
            + sum(
                gen_params[name]["segs"][k][1] * model.delta[name, k]
                for k in range(len(gen_params[name]["segs"]))
            )
            for name in committed_names
        )

    m.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

    return m
