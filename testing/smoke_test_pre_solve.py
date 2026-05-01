"""
Smoke test for the Pre-Solve Stage Monte Carlo Commitment Sampler.

Loads an RTS-GMLC instance, computes per-period thermal demand and regulation
requirements (same logic as Stage 1), then runs the Monte Carlo sampler across
all 48 time periods to build a distribution of n_committed per period.

The sampler uses the same stopping conditions as the Stage 1 GA:
  • new_pmax             >= thermal_demand + reg_up_req
  • new_reg_up_potential >= reg_up_req
  • new_reg_down_potential >= reg_down_req

No optimization (ED or otherwise) is performed — pure combinatorial sampling.
Completes in a few seconds on the RTS-GMLC instance.

Outputs
-------
1. Per-period distribution table (mean, P10, P25, median, P75, P90, target).
2. Startup/shutdown target table for each period transition.
3. Per-unit frequency table sorted by Pmax.
4. CSV export to testing/results/pre_solve_<instance_stem>.csv.

Usage (from repo root, with power-systems conda env active):
    python testing/smoke_test_pre_solve.py
"""

import csv
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pre_solve_stage.config import PreSolveConfig
from src.pre_solve_stage.parallel import PreSolveResult, run_all_periods

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PGLIB_UC_ROOT = Path("C:/gitrepos/power-grid-lib/pglib-uc")
INSTANCE_PATH = PGLIB_UC_ROOT / "rts_gmlc" / "2020-01-27.json"
RESULTS_DIR   = Path(__file__).parent / "results"

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = PreSolveConfig(
    n_samples=5_000,
    sort_attribute="power_output_maximum",
    sort_ascending=False,
    target_percentile=50.0,   # median as the commitment target
)
N_WORKERS = None   # None → all logical CPUs


# ── Instance loader ───────────────────────────────────────────────────────────

class InstanceData:
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
        renewable_max_eff = [min(renewable_max[t], total_demand[t]) for t in range(n)]
        self.renewable_expected = [
            (renewable_min[t] + renewable_max_eff[t]) / 2.0 for t in range(n)
        ]
        self.thermal_demand = [
            max(0.0, total_demand[t] - self.renewable_expected[t]) for t in range(n)
        ]
        self.renewable_reg_up   = [self.renewable_expected[t] - renewable_min[t] for t in range(n)]
        self.renewable_reg_down = [renewable_max_eff[t] - self.renewable_expected[t] for t in range(n)]


def load_instance(path: Path) -> InstanceData:
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
            continue
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


# ── Output helpers ────────────────────────────────────────────────────────────

def print_unit_frequency_table(result: PreSolveResult, generators: dict) -> None:
    mean_freq = result.mean_frequency()
    matrix    = result.commit_frequency_matrix()
    n_periods = len(result.period_results)

    sorted_names = sorted(
        result.gen_names,
        key=lambda n: generators[n].get("power_output_maximum", 0.0),
        reverse=True,
    )

    print(f"\n{'=' * 82}")
    print(f"  Pre-Solve Stage — Per-Unit Commitment Frequency  "
          f"({n_periods} periods, {result.period_results[0].total_samples:,} samples/period)")
    print(f"{'=' * 82}")
    print(f"  {'Unit':<20}  {'Pmax (MW)':>9}  {'Pmin (MW)':>9}  "
          f"{'Mean Freq':>9}  {'Min Freq':>9}  {'Max Freq':>9}")
    print(f"  {'-'*20}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")

    for name in sorted_names:
        pmax  = generators[name].get("power_output_maximum", 0.0)
        pmin  = generators[name].get("power_output_minimum", 0.0)
        freqs = matrix[name]
        print(f"  {name:<20}  {pmax:>9.1f}  {pmin:>9.1f}  "
              f"  {mean_freq[name]:>8.1%}  {min(freqs):>8.1%}  {max(freqs):>8.1%}")

    print(f"{'=' * 82}\n")


def export_csv(result: PreSolveResult, generators: dict, path: Path) -> None:
    """
    Export per-period distribution stats and per-unit frequencies to CSV.

    Section 1: period-level distribution summary + startup/shutdown targets.
    Section 2: unit-level frequency matrix (one row per unit).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    targets     = result.target_n_committed()
    transitions = result.startup_shutdown_targets()
    n_periods   = len(result.period_results)

    sorted_names = sorted(
        result.gen_names,
        key=lambda n: generators[n].get("power_output_maximum", 0.0),
        reverse=True,
    )

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # ── Period distribution summary ──────────────────────────────────────
        writer.writerow([
            "period", "thermal_demand_mw", "reg_up_req_mw", "reg_down_req_mw",
            "mean_n_committed", "p10_n_committed", "p25_n_committed",
            "median_n_committed", "p75_n_committed", "p90_n_committed",
            "target_n_committed", "startup_target", "shutdown_target",
            "wall_seconds",
        ])
        for t, pr in enumerate(result.period_results):
            su = transitions[t][0] if t < len(transitions) else ""
            sd = transitions[t][1] if t < len(transitions) else ""
            writer.writerow([
                t,
                f"{pr.thermal_demand:.4f}",
                f"{pr.reg_up_req:.4f}",
                f"{pr.reg_down_req:.4f}",
                f"{pr.mean_n_committed:.4f}",
                f"{pr.percentile_n_committed(10):.4f}",
                f"{pr.percentile_n_committed(25):.4f}",
                f"{pr.median_n_committed:.4f}",
                f"{pr.percentile_n_committed(75):.4f}",
                f"{pr.percentile_n_committed(90):.4f}",
                targets[t],
                su,
                sd,
                f"{pr.wall_seconds:.4f}",
            ])

        writer.writerow([])

        # ── Unit frequency matrix ────────────────────────────────────────────
        matrix    = result.commit_frequency_matrix()
        mean_freq = result.mean_frequency()
        writer.writerow(["unit", "pmax_mw", "pmin_mw", "mean_freq"]
                        + [f"t{t}_freq" for t in range(n_periods)])
        for name in sorted_names:
            pmax = generators[name].get("power_output_maximum", 0.0)
            pmin = generators[name].get("power_output_minimum", 0.0)
            writer.writerow(
                [name, f"{pmax:.2f}", f"{pmin:.2f}", f"{mean_freq[name]:.4f}"]
                + [f"{v:.4f}" for v in matrix[name]]
            )

    print(f"  CSV exported → {path}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading instance: {INSTANCE_PATH}")
    inst = load_instance(INSTANCE_PATH)
    n_thermal = len(inst.thermal)
    n_periods = len(inst.total_demand)
    print(f"Instance: {n_thermal} thermal generators, {n_periods} time periods\n",
          flush=True)

    print(f"Config: n_samples={CONFIG.n_samples:,}  "
          f"sort={CONFIG.sort_attribute}  "
          f"target=p{CONFIG.target_percentile:.0f}  "
          f"n_workers={N_WORKERS or 'auto'}\n", flush=True)

    result = run_all_periods(
        generators=inst.thermal,
        thermal_demand_values=inst.thermal_demand,
        reg_up_req_values=inst.renewable_reg_up,
        reg_down_req_values=inst.renewable_reg_down,
        config=CONFIG,
        n_workers=N_WORKERS,
        base_seed=42,
        show_progress=True,
    )

    result.print_summary()
    print_unit_frequency_table(result, inst.thermal)

    csv_path = RESULTS_DIR / f"pre_solve_{INSTANCE_PATH.stem}.csv"
    export_csv(result, inst.thermal, csv_path)


if __name__ == "__main__":
    main()
