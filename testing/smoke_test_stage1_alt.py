"""
Smoke test for the Alternative Stage 1 Monte Carlo Commitment Sampler.

Loads the RTS-GMLC instance, computes per-period non-renewable demand bounds
from the renewable forecast data, runs the sampler across all 48 time periods,
and prints per-unit commitment frequency statistics.

The sampler makes NO optimization calls — it is pure combinatorial sampling
and should complete in a few seconds on the RTS-GMLC instance.

Non-renewable demand bounds (passed to the sampler as stopping thresholds):
    demand_upper[t] = expected_demand[t] - min_renewable[t]
    demand_lower[t] = max(0, expected_demand[t] - max_renewable[t])

Stopping thresholds:
    upper_threshold[t] = demand_upper[t] × (1 + upper_tolerance)
    lower_threshold[t] = max(0, demand_lower[t] × (1 − lower_tolerance))

Outputs
-------
1. Per-period summary table (demand bounds, thresholds, avg committed, wall time).
2. Per-unit frequency table sorted by Pmax (mean freq + period min/max).
3. CSV export to results/stage1_alt_frequencies.csv (overwritten each run).

Usage (from repo root, with power-systems conda env active):
    python smoke_test_stage1_alt.py
"""

import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stage1_alt.config import AltStage1Config
from src.stage1_alt.parallel import AltStage1Result, run_all_periods_alt

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
CSV_OUTPUT    = Path(__file__).parent.parent / "output" / "stage1_alt_frequencies.csv"

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = AltStage1Config(
    n_samples=1_000,
    upper_tolerance=0.05,          # +5% above max non-renewable demand
    lower_tolerance=0.05,          # -5% below min non-renewable demand
    sort_attribute="power_output_maximum",
    sort_ascending=False,          # largest Pmax first
    location_dist_type="uniform",
    cut_size_min=1,
    cut_size_max=1,
)
N_WORKERS = None   # None → all logical CPUs


# ── Instance loader ───────────────────────────────────────────────────────────

def load_instance(path: Path) -> tuple[dict, list[float], list[float], list[float]]:
    """
    Return (thermal_generators, demand_values, demand_upper_values, demand_lower_values).

    demand_upper[t] = expected_demand[t] - min_renewable[t]
    demand_lower[t] = max(0, expected_demand[t] - max_renewable[t])
    """
    with open(path) as f:
        data = json.load(f)

    thermal = data["thermal_generators"]

    # ── Total expected demand ─────────────────────────────────────────────────
    demand_raw = data["demand"]
    if isinstance(demand_raw, list):
        demand_values: list[float] = [float(d) for d in demand_raw]
    elif isinstance(demand_raw, dict):
        n = len(next(iter(demand_raw.values())))
        demand_values = [
            sum(bus[t] for bus in demand_raw.values()) for t in range(n)
        ]
    else:
        raise ValueError(f"Unexpected demand format: {type(demand_raw)}")

    n_periods = len(demand_values)

    # ── Renewable generation bounds ───────────────────────────────────────────
    renewables = data.get("renewable_generators", {})

    # Aggregate min and max renewable output across all renewable units per period.
    # Each renewable unit has power_output_minimum and power_output_maximum as
    # per-period lists (pglib-uc standard format).
    min_renewable: list[float] = [0.0] * n_periods
    max_renewable: list[float] = [0.0] * n_periods

    for gen_data in renewables.values():
        pmin_series = gen_data.get("power_output_minimum", [])
        pmax_series = gen_data.get("power_output_maximum", [])
        for t in range(n_periods):
            if t < len(pmin_series):
                min_renewable[t] += float(pmin_series[t])
            if t < len(pmax_series):
                max_renewable[t] += float(pmax_series[t])

    # Non-renewable demand bounds
    demand_upper = [demand_values[t] - min_renewable[t] for t in range(n_periods)]
    demand_lower = [max(0.0, demand_values[t] - max_renewable[t]) for t in range(n_periods)]

    return thermal, demand_values, demand_upper, demand_lower


# ── Unit frequency table ──────────────────────────────────────────────────────

def print_unit_frequency_table(result: AltStage1Result, generators: dict) -> None:
    """Print per-unit commitment frequency statistics across all periods."""
    mean_freq = result.mean_frequency()
    matrix    = result.commit_frequency_matrix()
    n_periods = len(result.period_results)

    # Sort by Pmax descending for readability
    sorted_names = sorted(
        result.gen_names,
        key=lambda n: generators[n].get("power_output_maximum", 0.0),
        reverse=True,
    )

    print(f"\n{'=' * 82}")
    print(f"  Alt Stage 1 — Per-Unit Commitment Frequency  "
          f"({n_periods} periods, {result.period_results[0].total_samples:,} samples/period)")
    print(f"{'=' * 82}")
    print(f"  {'Unit':<20}  {'Pmax (MW)':>9}  {'Pmin (MW)':>9}  "
          f"{'Mean Freq':>9}  {'Min Freq':>9}  {'Max Freq':>9}")
    print(f"  {'-'*20}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")

    for name in sorted_names:
        pmax = generators[name].get("power_output_maximum", 0.0)
        pmin = generators[name].get("power_output_minimum", 0.0)
        freqs = matrix[name]
        mean_f = mean_freq[name]
        min_f  = min(freqs)
        max_f  = max(freqs)
        print(f"  {name:<20}  {pmax:>9.1f}  {pmin:>9.1f}  "
              f"  {mean_f:>8.1%}  {min_f:>8.1%}  {max_f:>8.1%}")

    print(f"{'=' * 82}\n")


def export_csv(result: AltStage1Result, generators: dict, path: Path) -> None:
    """
    Export per-unit per-period frequencies to a CSV file.
    Always overwrites the existing file if present.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = result.commit_frequency_matrix()
    n_periods = len(result.period_results)

    sorted_names = sorted(
        result.gen_names,
        key=lambda n: generators[n].get("power_output_maximum", 0.0),
        reverse=True,
    )

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["unit", "pmax_mw", "pmin_mw", "mean_freq"] + [
            f"t{t}_freq" for t in range(n_periods)
        ]
        writer.writerow(header)
        mean_freq = result.mean_frequency()
        for name in sorted_names:
            pmax = generators[name].get("power_output_maximum", 0.0)
            pmin = generators[name].get("power_output_minimum", 0.0)
            row = [name, f"{pmax:.2f}", f"{pmin:.2f}", f"{mean_freq[name]:.4f}"]
            row += [f"{v:.4f}" for v in matrix[name]]
            writer.writerow(row)

    print(f"  CSV exported → {path}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading instance: {INSTANCE_PATH}")
    thermal, demand_values, demand_upper, demand_lower = load_instance(INSTANCE_PATH)
    print(f"Instance: {len(thermal)} thermal generators, "
          f"{len(demand_values)} time periods\n", flush=True)

    print(f"Config: n_samples={CONFIG.n_samples:,}  "
          f"upper_tol=+{CONFIG.upper_tolerance:.0%}  "
          f"lower_tol=-{CONFIG.lower_tolerance:.0%}  "
          f"sort={CONFIG.sort_attribute}  "
          f"cut_size=[{CONFIG.cut_size_min},{CONFIG.cut_size_max}]  "
          f"location={CONFIG.location_dist_type}\n", flush=True)

    result = run_all_periods_alt(
        generators=thermal,
        demand_values=demand_values,
        demand_upper_values=demand_upper,
        demand_lower_values=demand_lower,
        config=CONFIG,
        n_workers=N_WORKERS,
        base_seed=42,
        show_progress=True,
    )

    result.print_summary()
    print_unit_frequency_table(result, thermal)
    export_csv(result, thermal, CSV_OUTPUT)


if __name__ == "__main__":
    main()
