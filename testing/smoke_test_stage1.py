"""
Smoke test for Stage 1 GA.

Modes (set MODE below):
  'single'  — one time period, verbose stats  (quick sanity check)
  'all'     — all time periods in parallel    (full Stage 1 timing run)

Usage (from repo root, with power-systems conda env active):
    python smoke_test_stage1.py
"""

import csv
import json
import logging
import math
import re
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stage1_ga.ga import run_stage1_ga
from src.stage1_ga.parallel import AllPeriodsResult, run_all_periods
from src.io.stage1_io import save_stage1_result
from src.pre_solve_stage.parallel import run_all_periods as run_pre_solve_all_periods
from testing.control_panel import CURRENT

# ── Logging setup ─────────────────────────────────────────────────────────────
# force=True clears any handlers that Pyomo (or other imports) may have
# installed before we get here, preventing duplicate log output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)

# ── Settings from control panel ───────────────────────────────────────────────
INSTANCE_PATH   = CURRENT.instance_path
MODE            = CURRENT.stage1_mode          # 'single' or 'all'
TIME_PERIOD_IDX = CURRENT.stage1_single_period # used only in 'single' mode
N_WORKERS       = CURRENT.stage1_n_workers     # None → all logical CPUs

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
CACHE_DIR   = Path(__file__).parent / "cache"


class InstanceData:
    """Parsed instance with thermal demand and per-period wind regulation requirements."""

    def __init__(
        self,
        thermal: dict,
        total_demand: list[float],
        renewable_min: list[float],
        renewable_max: list[float],
    ) -> None:
        self.thermal = thermal
        self.total_demand = total_demand
        self.renewable_min = renewable_min
        self.renewable_max = renewable_max
        n = len(total_demand)
        # Cap renewable max at total demand — thermal reg-down can never exceed demand itself
        renewable_max_eff = [min(renewable_max[t], total_demand[t]) for t in range(n)]
        # Expected renewable output = midpoint of effective forecast band (wind + PV)
        self.renewable_expected = [(renewable_min[t] + renewable_max_eff[t]) / 2.0 for t in range(n)]
        # Thermal demand = total demand less expected renewable output
        self.thermal_demand = [
            max(0.0, total_demand[t] - self.renewable_expected[t]) for t in range(n)
        ]
        # Regulation requirements driven by renewable forecast uncertainty
        self.renewable_reg_up   = [self.renewable_expected[t] - renewable_min[t] for t in range(n)]
        self.renewable_reg_down = [renewable_max_eff[t] - self.renewable_expected[t] for t in range(n)]

    def print_renewable_summary(self) -> None:
        n = len(self.total_demand)
        print(f"\n{'=' * 90}")
        print(f"  Renewable Adjustment Summary  ({n} periods)")
        print(f"{'=' * 90}")
        print(f"  {'t':<5}  {'Total Dem':>10}  {'Ren Min':>10}  {'Ren Exp':>10}"
              f"  {'Ren Max':>10}  {'Ren Max Cap':>11}  {'Therm Dem':>10}"
              f"  {'Reg Up Req':>11}  {'Reg Dn Req':>11}")
        print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}"
              f"  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*11}  {'-'*11}")
        for t in range(n):
            ren_max_cap = min(self.renewable_max[t], self.total_demand[t])
            print(f"  {t:<5}  {self.total_demand[t]:>10.1f}  {self.renewable_min[t]:>10.1f}"
                  f"  {self.renewable_expected[t]:>10.1f}  {self.renewable_max[t]:>10.1f}"
                  f"  {ren_max_cap:>11.1f}  {self.thermal_demand[t]:>10.1f}"
                  f"  {self.renewable_reg_up[t]:>11.1f}  {self.renewable_reg_down[t]:>11.1f}")
        print(f"{'=' * 90}\n", flush=True)


def load_instance(path: Path) -> InstanceData:
    """
    Parse a pglib-uc JSON file and return an InstanceData.

    Thermal demand per period = total demand − expected renewable output
    where expected renewable = (aggregate pmin + aggregate pmax) / 2
    aggregated across stochastic renewable generators (wind + PV) only.
    Hydro generators are excluded — they are controllable and tracked separately.

    Renewable max is capped at total demand before all calculations — thermal
    regulation can never exceed actual demand for that period.

    Renewable regulation requirements:
      reg_up_req   = expected renewable − min renewable  (output could drop → thermal covers)
      reg_down_req = min(max renewable, demand) − expected renewable  (output could surge)
    """
    with open(path) as f:
        data = json.load(f)

    thermal = data["thermal_generators"]

    demand_raw = data["demand"]
    if isinstance(demand_raw, list):
        total_demand: list[float] = [float(d) for d in demand_raw]
    elif isinstance(demand_raw, dict):
        n = len(next(iter(demand_raw.values())))
        total_demand = [sum(bus[t] for bus in demand_raw.values()) for t in range(n)]
    else:
        raise ValueError(f"Unexpected demand format: {type(demand_raw)}")

    n_periods = len(total_demand)
    renewable_min: list[float] = [0.0] * n_periods
    renewable_max: list[float] = [0.0] * n_periods

    for name, gen_data in data.get("renewable_generators", {}).items():
        if re.search(r"HYDRO", name, re.IGNORECASE):
            continue  # hydro is controllable — exclude from stochastic aggregation
        pmin_series = gen_data.get("power_output_minimum", [])
        pmax_series = gen_data.get("power_output_maximum", [])
        for t in range(n_periods):
            if t < len(pmin_series):
                renewable_min[t] += float(pmin_series[t])
            if t < len(pmax_series):
                renewable_max[t] += float(pmax_series[t])

    return InstanceData(
        thermal=thermal,
        total_demand=total_demand,
        renewable_min=renewable_min,
        renewable_max=renewable_max,
    )


def run_single(inst: InstanceData) -> None:
    n_thermal = len(inst.thermal)
    demand = inst.thermal_demand[TIME_PERIOD_IDX]
    total  = inst.total_demand[TIME_PERIOD_IDX]
    print(f"Running single period: t={TIME_PERIOD_IDX}"
          f"  total_demand={total:.1f} MW  thermal_demand={demand:.1f} MW\n",
          flush=True)

    rng = np.random.default_rng(seed=CURRENT.rng_seed)
    pop, stats = run_stage1_ga(
        generators=inst.thermal,
        demand=demand,
        config=CURRENT.stage1,
        rng=rng,
        time_period=TIME_PERIOD_IDX,
        reg_up_req=inst.renewable_reg_up[TIME_PERIOD_IDX],
        reg_down_req=inst.renewable_reg_down[TIME_PERIOD_IDX],
    )

    stats.print_summary(time_period=TIME_PERIOD_IDX)

    feasible = pop.feasible()
    print(f"Top {min(5, len(feasible))} feasible chromosomes:")
    print(f"  {'Rank':<6}  {'Fitness ($)':<18}  {'Committed':<12}  {'Hash (short)'}")
    print(f"  {'-'*6}  {'-'*18}  {'-'*12}  {'-'*12}")
    for rank, chrom in enumerate(feasible[:5], start=1):
        print(f"  {rank:<6}  {chrom.fitness:<18,.2f}  "
              f"{chrom.n_committed}/{n_thermal:<8}  {chrom.hash[:12]}")
    if not feasible:
        print("  No feasible chromosomes found.")


def export_csv(result: AllPeriodsResult, inst: InstanceData) -> Path:
    """
    Write one row per period.

    Columns include total demand, wind estimates, thermal demand,
    wind reg requirements, and the best chromosome's cost/committed/reg_up/reg_down.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    instance_stem = INSTANCE_PATH.stem
    out_path = RESULTS_DIR / f"stage1_{instance_stem}.csv"

    n_total = len(inst.thermal)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            # Period index
            "period",
            # Demand breakdown
            "total_demand_mw",
            "renewable_min_mw",
            "renewable_expected_mw",
            "renewable_max_mw",
            "thermal_demand_mw",
            # Renewable uncertainty → regulation requirements passed to GA
            "reg_up_req_mw",
            "reg_down_req_mw",
            # Best chromosome solution
            "best_cost_usd",
            "n_committed",
            "n_total",
            # Regulation capability provided by best chromosome
            "chrom_reg_up_mw",
            "chrom_reg_down_mw",
            # Headroom vs requirement (positive = requirement met)
            "reg_up_margin_mw",
            "reg_down_margin_mw",
            # Renewable curtailment due to thermal pmin floor
            "renewable_loss_mw",
            # GA run metadata — infeasibility breakdown
            "n_ed_feasible",
            "n_pmax_infeasible",   # under-committed: sum(pmax) < demand
            "n_pmin_augmented",    # over-committed: sum(pmin) > demand, demand raised
            "n_ed_solver_infeasible",  # passed pmax screen but solver still failed
            "n_reg_infeasible",
            "n_generations",
            "stop_reason",
        ])
        for t, (pop, s) in enumerate(zip(result.populations, result.period_stats)):
            best = pop.best
            rup_req = inst.renewable_reg_up[t]
            rdn_req = inst.renewable_reg_down[t]
            chrom_rup = best.reg_up   if (best and best.reg_up   is not None) else None
            chrom_rdn = best.reg_down if (best and best.reg_down is not None) else None
            writer.writerow([
                t,
                f"{inst.total_demand[t]:.4f}",
                f"{inst.renewable_min[t]:.4f}",
                f"{inst.renewable_expected[t]:.4f}",
                f"{inst.renewable_max[t]:.4f}",
                f"{inst.thermal_demand[t]:.4f}",
                f"{rup_req:.4f}",
                f"{rdn_req:.4f}",
                f"{s.best_fitness:.4f}" if math.isfinite(s.best_fitness) else "",
                best.n_committed if best else "",
                n_total,
                f"{chrom_rup:.4f}" if chrom_rup is not None else "",
                f"{chrom_rdn:.4f}" if chrom_rdn is not None else "",
                f"{chrom_rup - rup_req:.4f}" if chrom_rup is not None else "",
                f"{chrom_rdn - rdn_req:.4f}" if chrom_rdn is not None else "",
                f"{best.renewable_loss:.4f}" if (best and best.renewable_loss is not None) else "",
                s.n_ed_feasible,
                s.n_pmax_infeasible,
                s.n_pmin_augmented,
                s.n_ed_solver_infeasible,
                s.n_reg_infeasible,
                s.n_generations,
                s.stop_reason,
            ])

    print(f"  CSV exported → {out_path}\n")
    return out_path


def run_pre_solve(inst: InstanceData) -> list[int] | None:
    """Run the pre-solve stage and return per-period n_committed targets, or None if disabled."""
    if not CURRENT.stage1_use_presolve_targets:
        print("Pre-solve target guidance disabled — skipping pre-solve.\n", flush=True)
        return None

    n_periods = len(inst.thermal_demand)
    print(f"Running pre-solve ({CURRENT.pre_solve.n_samples:,} samples/period, "
          f"{n_periods} periods)…\n", flush=True)

    pre_result = run_pre_solve_all_periods(
        generators=inst.thermal,
        thermal_demand_values=inst.thermal_demand,
        reg_up_req_values=inst.renewable_reg_up,
        reg_down_req_values=inst.renewable_reg_down,
        config=CURRENT.pre_solve,
        n_workers=N_WORKERS,
        base_seed=CURRENT.rng_seed,
        show_progress=True,
    )
    pre_result.print_summary()
    targets = pre_result.target_n_committed()
    print(f"Pre-solve targets: min={min(targets)}  max={max(targets)}"
          f"  avg={sum(targets)/len(targets):.1f}\n", flush=True)
    return targets


def run_all(inst: InstanceData) -> None:
    n_periods = len(inst.thermal_demand)

    # ── Pre-solve (optional) ──────────────────────────────────────────────────
    targets = run_pre_solve(inst)

    print(f"Running all {n_periods} periods in parallel  "
          f"(n_workers={N_WORKERS or 'auto'})\n", flush=True)

    result = run_all_periods(
        generators=inst.thermal,
        demand_values=inst.thermal_demand,
        config=CURRENT.stage1,
        n_workers=N_WORKERS,
        base_seed=CURRENT.rng_seed,
        show_progress=True,
        reg_up_reqs=inst.renewable_reg_up,
        reg_down_reqs=inst.renewable_reg_down,
        target_n_committed=targets,
    )

    result.print_summary()
    export_csv(result, inst)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"stage1_{INSTANCE_PATH.stem}.json"
    save_stage1_result(result, cache_path)
    print(f"  Stage 1 result cached → {cache_path}\n")


def main() -> None:
    print(f"Loading instance: {INSTANCE_PATH}")
    inst = load_instance(INSTANCE_PATH)
    print(f"Instance: {len(inst.thermal)} thermal generators, "
          f"{len(inst.total_demand)} time periods\n", flush=True)
    inst.print_renewable_summary()

    if MODE == "single":
        run_single(inst)
    elif MODE == "all":
        run_all(inst)
    else:
        raise ValueError(f"Unknown MODE '{MODE}'. Use 'single' or 'all'.")


if __name__ == "__main__":
    main()
