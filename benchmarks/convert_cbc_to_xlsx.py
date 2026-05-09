"""
Convert a CBC solution JSON (produced by uc_model.py) to the expanded
comparison xlsx format.

Usage (from repo root, with power-systems conda env active):
    python benchmarks/convert_cbc_to_xlsx.py

Requires:
  - benchmarks/cbc_solution.json   (produced by uc_model.py)
  - C:/gitrepos/power-grid-lib/pglib-uc/rts_gmlc/2020-01-27.json

Output:
  - benchmarks/cbc_solution.xlsx

Notes on approximations
-----------------------
The CBC JSON contains thermal variables (ug, pg, rg, vg, wg) but NOT the
renewable dispatch variable pw.  Renewable dispatch is approximated as:
  - Hydro  : pmax[t]  (pmin ≈ pmax for hydro, so this is accurate)
  - Wind / Solar / CSP / PV : pmax[t] (CBC dispatches renewables at maximum
    when no curtailment is forced by the reserves constraint — a good
    approximation for the purpose of generation mix visualization)

CBC pg[g,t] is power ABOVE pmin.  Total dispatch = pg[g,t] + pmin[g]*ug[g,t].
CBC time periods are 1-indexed (t=1..48).  Display periods are also 1..48.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io.xlsx_export import (
    classify_fuel,
    compute_all_reserves_per_period,
    compute_dispatch_cost_per_period,
    export_solution_xlsx,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

INSTANCE_PATH = Path("C:/gitrepos/power-grid-lib/pglib-uc/rts_gmlc/2020-01-27.json")
SOLUTION_JSON = Path(__file__).parent / "cbc_solution.json"
OUTPUT_XLSX   = Path(__file__).parent / "cbc_solution.xlsx"


# ── Instance helpers (mirrors smoke_test_stage1.py) ───────────────────────────

def _load_instance(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _compute_renewable_stats(
    renewable_gens: dict,
    n_periods: int,
) -> tuple[list[float], list[float], list[float]]:
    """Return (renewable_min, renewable_expected, renewable_max) per period."""
    ren_min = [0.0] * n_periods
    ren_max = [0.0] * n_periods

    for name, gen in renewable_gens.items():
        if re.search(r"HYDRO", name, re.IGNORECASE):
            continue  # hydro tracked separately
        pmin_s = gen.get("power_output_minimum", [])
        pmax_s = gen.get("power_output_maximum", [])
        for t in range(n_periods):
            if t < len(pmin_s):
                ren_min[t] += float(pmin_s[t])
            if t < len(pmax_s):
                ren_max[t] += float(pmax_s[t])

    ren_exp = [(ren_min[t] + ren_max[t]) / 2.0 for t in range(n_periods)]
    return ren_min, ren_exp, ren_max


def _compute_total_demand(data: dict, n_periods: int) -> list[float]:
    demand_raw = data["demand"]
    if isinstance(demand_raw, list):
        return [float(d) for d in demand_raw]
    n = len(next(iter(demand_raw.values())))
    return [sum(bus[t] for bus in demand_raw.values()) for t in range(n)]


# ── CBC solution processing ───────────────────────────────────────────────────

def _parse_cbc_thermal_dispatch(
    sol: dict,
    thermal_gens: dict,
    n_periods: int,
) -> tuple[dict[str, list[float]], list[list[str]], list[int]]:
    """
    Build thermal_dispatch (total MW), committed_per_period, committed_counts.

    CBC uses 1-indexed periods.  Output arrays are 0-indexed (0..n_periods-1).
    CBC pg[g,t] = power above pmin.  Total = pg + pmin * ug.
    """
    thermal_dispatch: dict[str, list[float]] = {}
    committed_per_period: list[list[str]] = [[] for _ in range(n_periods)]
    committed_counts: list[int] = [0] * n_periods

    ug_data = sol.get("ug", {})
    pg_data = sol.get("pg", {})

    for name, gen in thermal_gens.items():
        pmin = float(gen.get("power_output_minimum", 0.0))
        row: list[float] = []
        for t_display in range(1, n_periods + 1):   # CBC 1-indexed
            t_idx = t_display - 1                   # 0-indexed
            key   = f"{name},{t_display}"
            ug_v  = ug_data.get(key)
            pg_v  = pg_data.get(key)
            if ug_v is None:
                row.append(0.0)
                continue
            on = round(float(ug_v)) >= 1
            if on:
                total_mw = (float(pg_v) if pg_v is not None else 0.0) + pmin
                row.append(max(0.0, round(total_mw, 4)))
                committed_per_period[t_idx].append(name)
                committed_counts[t_idx] += 1
            else:
                row.append(0.0)
        thermal_dispatch[name] = row

    return thermal_dispatch, committed_per_period, committed_counts


def _parse_cbc_renewable_dispatch(
    sol: dict,
    renewable_gens: dict,
    n_periods: int,
) -> tuple[dict[str, list[float]], bool]:
    """
    Build renewable dispatch from pw solution values when available.

    Returns (renewable_dispatch, used_actual) where used_actual=True means
    the pw variables were found in the JSON (re-run of uc_model.py with pw saved).
    Falls back to pmax approximation if pw is missing (old JSON format).

    CBC pw[w,t] is bounded by [pmin[t], pmax[t]] and represents actual dispatch.
    """
    pw_data = sol.get("pw", {})
    used_actual = bool(pw_data)

    renewable_dispatch: dict[str, list[float]] = {}
    for name, gen in renewable_gens.items():
        pmax_s = gen.get("power_output_maximum", [])
        row: list[float] = []
        for t_display in range(1, n_periods + 1):   # CBC 1-indexed
            t_idx = t_display - 1
            if used_actual:
                key = f"{name},{t_display}"
                v   = pw_data.get(key)
                row.append(round(float(v), 4) if v is not None else 0.0)
            else:
                # Fallback: use pmax (approximation)
                row.append(float(pmax_s[t_idx]) if t_idx < len(pmax_s) else 0.0)
        renewable_dispatch[name] = row

    return renewable_dispatch, used_actual


# ── Startup cost ─────────────────────────────────────────────────────────────

def _compute_startup_cost_per_period(
    sol: dict,
    thermal_gens: dict,
    n_periods: int,
) -> list[float]:
    """
    Compute startup cost per period from CBC vg (startup indicator).

    For each vg[g,t] == 1 event, traces ug backwards to find the offline
    duration, then applies the generator's startup cost tier logic (same
    as the heuristic's _startup_cost):
      - tier with the highest lag <= offline_duration wins.
      - For startups at t=1, time_down_t0 from the instance is included.
    """
    vg_data = sol.get("vg", {})
    ug_data = sol.get("ug", {})
    startup_cost = [0.0] * n_periods

    for name, gen in thermal_gens.items():
        tiers = gen.get("startup", [])
        if not tiers:
            continue
        for t_display in range(1, n_periods + 1):
            t_idx = t_display - 1
            vg_v = vg_data.get(f"{name},{t_display}", 0.0)
            if round(float(vg_v)) < 1:
                continue

            # Count consecutive OFF periods immediately before this startup
            offline_periods = 0
            for t_back in range(t_display - 1, 0, -1):
                ug_back = ug_data.get(f"{name},{t_back}", None)
                if ug_back is not None and round(float(ug_back)) < 1:
                    offline_periods += 1
                else:
                    break

            # If we traced all the way back to t=0, add the t0 offline count
            if offline_periods == t_display - 1 and int(gen.get("unit_on_t0", 0)) == 0:
                offline_periods += int(gen.get("time_down_t0", 0))

            # Pick the most expensive qualifying tier
            cost = float(tiers[0]["cost"])
            for tier in tiers:
                if offline_periods >= int(tier["lag"]):
                    cost = float(tier["cost"])
                else:
                    break
            startup_cost[t_idx] += cost

    return startup_cost


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading instance: {INSTANCE_PATH}")
    data = _load_instance(INSTANCE_PATH)
    thermal_gens   = data["thermal_generators"]
    renewable_gens = data["renewable_generators"]
    n_periods      = int(data["time_periods"])

    print(f"Instance: {len(thermal_gens)} thermal, "
          f"{len(renewable_gens)} renewable, {n_periods} periods")

    print(f"Loading CBC solution: {SOLUTION_JSON}")
    with open(SOLUTION_JSON) as f:
        sol = json.load(f)

    obj = sol.get("objective")
    print(f"Objective: ${obj:,.2f}" if obj else "Objective: (not found)")
    print(f"Status: {sol.get('solver_status')}  "
          f"Termination: {sol.get('termination')}\n")

    # Demand
    total_demand = _compute_total_demand(data, n_periods)
    ren_min, ren_exp, ren_max = _compute_renewable_stats(renewable_gens, n_periods)

    # Cap ren_max at total demand
    ren_max_eff = [min(ren_max[t], total_demand[t]) for t in range(n_periods)]
    thermal_demand = [max(0.0, total_demand[t] - ren_exp[t]) for t in range(n_periods)]

    # Thermal dispatch
    thermal_dispatch, committed_per_period, committed_counts = \
        _parse_cbc_thermal_dispatch(sol, thermal_gens, n_periods)

    # Renewable dispatch (actual pw values if present, pmax approximation otherwise)
    renewable_dispatch, used_actual = _parse_cbc_renewable_dispatch(
        sol, renewable_gens, n_periods
    )
    if used_actual:
        print("Renewable dispatch: using actual CBC pw solution values.")
    else:
        print("Renewable dispatch: pw not found in JSON — using pmax approximation.")
        print("  Re-run uc_model.py to capture exact renewable dispatch.\n")

    # All reserve categories
    reserves = compute_all_reserves_per_period(
        generators=thermal_gens,
        thermal_dispatch=thermal_dispatch,
        committed_per_period=committed_per_period,
        n_periods=n_periods,
    )

    # ED cost per period (from piecewise_production interpolation)
    ed_cost = compute_dispatch_cost_per_period(
        generators=thermal_gens,
        thermal_dispatch=thermal_dispatch,
        n_periods=n_periods,
    )

    # Startup cost per period (from CBC vg startup indicator + instance tiers)
    startup_cost = _compute_startup_cost_per_period(sol, thermal_gens, n_periods)
    total_su = sum(startup_cost)
    n_events  = sum(1 for v in startup_cost if v > 0)
    print(f"Startup costs: {n_events} startup events, total ${total_su:,.2f}")

    print("Writing xlsx...")
    output = export_solution_xlsx(
        path=OUTPUT_XLSX,
        generators_thermal=thermal_gens,
        generators_renewable=renewable_gens,
        n_periods=n_periods,
        thermal_dispatch=thermal_dispatch,
        renewable_dispatch=renewable_dispatch,
        thermal_demand=thermal_demand,
        renewable_expected=ren_exp,
        renewable_min_vals=ren_min,
        renewable_max_vals=ren_max_eff,
        committed_thermal=committed_counts,
        reg_up_total=reserves["reg_up"],
        reg_down_total=reserves["reg_down"],
        spin_up_total=reserves["spin_up"],
        spin_down_total=reserves["spin_down"],
        flex_up_total=reserves["flex_up"],
        flex_down_total=reserves["flex_down"],
        ed_cost_per_period=ed_cost,
        startup_cost_per_period=startup_cost,
        period_labels=[f"t={t}" for t in range(1, n_periods + 1)],
        sheet_title=f"CBC Benchmark  obj=${obj:,.0f}" if obj else "CBC Benchmark",
    )
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
