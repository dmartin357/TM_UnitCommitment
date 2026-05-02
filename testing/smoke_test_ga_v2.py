"""
Smoke test for the GA v2 initial population generator.

Loads the instance specified in control_panel.CURRENT, generates the GA v2
initial population (n_population forward solutions), prints a summary of each
solution, and exports the best solution to xlsx.

Usage (from repo root, with power-systems conda env active):
    python testing/smoke_test_ga_v2.py
"""

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ga_v2.config import GAv2Config
from src.ga_v2.population import generate_initial_population
from src.io.xlsx_export import compute_reg_per_period, export_solution_xlsx
from testing.control_panel import CURRENT

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_FORMAT  = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"
_LOG_DATEFMT = "%H:%M:%S"
_formatter   = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_formatter)

_root = logging.getLogger()
_root.setLevel(logging.DEBUG)
_root.handlers.clear()
_root.addHandler(_console_handler)

# ── Paths ─────────────────────────────────────────────────────────────────────
INSTANCE_PATH = CURRENT.instance_path
RESULTS_DIR   = Path(__file__).parent / "results"

# ── GA v2 config ──────────────────────────────────────────────────────────────
GA_V2_CONFIG = GAv2Config(
    n_population=10,
    n_candidates_per_period=8,
    economics_weight=0.5,
    regulation_weight=0.5,
    solver="auto",
    renewable_cost_per_mwh=0.01,
    rng_seed=CURRENT.rng_seed,
)

# ── Instance loader ───────────────────────────────────────────────────────────

def load_instance(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_total_demand(data: dict) -> list[float]:
    demand_raw = data["demand"]
    if isinstance(demand_raw, list):
        return [float(d) for d in demand_raw]
    n = len(next(iter(demand_raw.values())))
    return [sum(bus[t] for bus in demand_raw.values()) for t in range(n)]


# ── Excel export ──────────────────────────────────────────────────────────────

def export_solution(solution, thermal_gens, renewable_gens, total_demand, n_periods: int, path: Path) -> None:
    """Export the best ForwardSolution to the shared xlsx format."""

    dispatch_by_period = {d.period: d for d in solution.decisions}

    # Thermal dispatch and committed sets
    t0_committed = [n for n, g in thermal_gens.items() if g.get("unit_on_t0", 0) == 1]
    committed_per_period = [t0_committed] + [
        list(dispatch_by_period[t].committed_names) if t in dispatch_by_period else []
        for t in range(1, n_periods)
    ]

    thermal_dispatch: dict[str, list[float]] = {}
    for name, gen in thermal_gens.items():
        t0_mw = float(gen.get("power_output_t0", 0.0)) if name in t0_committed else 0.0
        row   = [t0_mw] + [
            dispatch_by_period[t].dispatch_thermal.get(name, 0.0) if t in dispatch_by_period else 0.0
            for t in range(1, n_periods)
        ]
        thermal_dispatch[name] = row

    # Renewable dispatch: use ED result for variable gens; hydro = pmax
    renewable_dispatch: dict[str, list[float]] = {}
    for name, gen in renewable_gens.items():
        pmin_s   = gen.get("power_output_minimum", [])
        pmax_s   = gen.get("power_output_maximum", [])
        is_hydro = bool(re.search(r"HYDRO", name, re.IGNORECASE))
        row = []
        for t in range(n_periods):
            if t == 0:
                # Period 0: use midpoint (or pmax for hydro)
                hi = float(pmax_s[t]) if t < len(pmax_s) else 0.0
                lo = float(pmin_s[t]) if t < len(pmin_s) else 0.0
                row.append(hi if is_hydro else (lo + hi) / 2.0)
            elif t in dispatch_by_period and not is_hydro:
                row.append(dispatch_by_period[t].dispatch_renewable.get(name, 0.0))
            else:
                hi = float(pmax_s[t]) if t < len(pmax_s) else 0.0
                lo = float(pmin_s[t]) if t < len(pmin_s) else 0.0
                row.append(hi if is_hydro else (lo + hi) / 2.0)
        renewable_dispatch[name] = row

    # Renewable stats for summary rows
    ren_min_vals  = [0.0] * n_periods
    ren_max_vals  = [0.0] * n_periods
    ren_exp_vals  = [0.0] * n_periods
    for name, gen in renewable_gens.items():
        if re.search(r"HYDRO", name, re.IGNORECASE):
            continue
        pmin_s = gen.get("power_output_minimum", [])
        pmax_s = gen.get("power_output_maximum", [])
        for t in range(n_periods):
            lo = float(pmin_s[t]) if t < len(pmin_s) else 0.0
            hi = float(pmax_s[t]) if t < len(pmax_s) else 0.0
            hi = min(hi, total_demand[t])
            ren_min_vals[t] += lo
            ren_max_vals[t] += hi
            ren_exp_vals[t] += (lo + hi) / 2.0

    thermal_demand = [
        max(0.0, total_demand[t] - ren_exp_vals[t]) for t in range(n_periods)
    ]
    committed_counts = [len(c) for c in committed_per_period]

    reg_up, reg_down = compute_reg_per_period(
        generators=thermal_gens,
        thermal_dispatch=thermal_dispatch,
        committed_per_period=committed_per_period,
        n_periods=n_periods,
    )

    export_solution_xlsx(
        path=path,
        generators_thermal=thermal_gens,
        generators_renewable=renewable_gens,
        n_periods=n_periods,
        thermal_dispatch=thermal_dispatch,
        renewable_dispatch=renewable_dispatch,
        thermal_demand=thermal_demand,
        renewable_expected=ren_exp_vals,
        renewable_min_vals=ren_min_vals,
        renewable_max_vals=ren_max_vals,
        committed_thermal=committed_counts,
        reg_up_total=reg_up,
        reg_down_total=reg_down,
        period_labels=[f"t={t}" for t in range(n_periods)],
        sheet_title="GA v2 Best",
    )
    print(f"  Best solution xlsx → {path}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading instance: {INSTANCE_PATH}")
    data          = load_instance(INSTANCE_PATH)
    thermal_gens  = data["thermal_generators"]
    renewable_gens = data.get("renewable_generators", {})
    total_demand  = compute_total_demand(data)
    n_periods     = len(total_demand)

    print(
        f"Instance: {len(thermal_gens)} thermal generators, "
        f"{len(renewable_gens)} renewable generators, "
        f"{n_periods} periods\n",
        flush=True,
    )

    rng = np.random.default_rng(GA_V2_CONFIG.rng_seed)

    print(
        f"GA v2 config:  population={GA_V2_CONFIG.n_population}  "
        f"candidates/period={GA_V2_CONFIG.n_candidates_per_period}  "
        f"econ_weight={GA_V2_CONFIG.economics_weight}  "
        f"reg_weight={GA_V2_CONFIG.regulation_weight}\n",
        flush=True,
    )

    population = generate_initial_population(
        generators=thermal_gens,
        renewable_gens=renewable_gens,
        total_demand=total_demand,
        n_periods=n_periods,
        config=GA_V2_CONFIG,
        rng=rng,
    )

    # ── Print all solution summaries ──────────────────────────────────────────
    print("\n\n── Per-solution summaries ──────────────────────────────────────\n")
    for i, sol in enumerate(population):
        complete = sol.is_complete(n_periods)
        label = f"Solution {i+1}/{len(population)}  ({sol.periods_solved}/{n_periods-1} periods solved)"
        label += "" if complete else "  [INCOMPLETE — excluded from best]"
        print(label)
        sol.print_summary()

    # ── Best solution xlsx (complete solutions only) ──────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    complete_solutions = [s for s in population if s.is_complete(n_periods)]
    print(
        f"Complete solutions: {len(complete_solutions)}/{len(population)}  "
        f"({len(population) - len(complete_solutions)} excluded due to infeasible periods)"
    )

    if not complete_solutions:
        print("WARNING: no complete solutions found — exporting best incomplete solution.")
        best = min(population, key=lambda s: (-s.periods_solved, s.total_cost))
    else:
        best = min(complete_solutions, key=lambda s: s.total_cost)
    xlsx_path = RESULTS_DIR / f"ga_v2_best_{INSTANCE_PATH.stem}.xlsx"
    export_solution(best, thermal_gens, renewable_gens, total_demand, n_periods, xlsx_path)


if __name__ == "__main__":
    # File handler — DEBUG and above
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _log_path = RESULTS_DIR / f"ga_v2_{INSTANCE_PATH.stem}.log"
    _fh = logging.FileHandler(_log_path, mode="w", encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(_formatter)
    logging.getLogger().addHandler(_fh)
    print(f"  Detailed log → {_log_path}\n")

    main()
