"""
Smoke test for Stage 1 GA.

Modes (set MODE below):
  'single'  — one time period, verbose stats  (quick sanity check)
  'all'     — all time periods in parallel    (full Stage 1 timing run)

Usage (from repo root, with power-systems conda env active):
    python smoke_test_stage1.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from src.stage1_ga.config import GAConfig
from src.stage1_ga.ga import run_stage1_ga
from src.stage1_ga.parallel import run_all_periods

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

# ── Run mode ──────────────────────────────────────────────────────────────────
MODE = "all"          # 'single' or 'all'
TIME_PERIOD_IDX = 0   # used only in 'single' mode
N_WORKERS = None      # None → use all logical CPUs; set an int to cap

# ── Instance ──────────────────────────────────────────────────────────────────
PGLIB_UC_ROOT = Path("C:/gitrepos/power-grid-lib/pglib-uc")
INSTANCE_PATH = PGLIB_UC_ROOT / "rts_gmlc" / "2020-01-27.json"

# ── Config ────────────────────────────────────────────────────────────────────
config = GAConfig(
    population_size=50,
    initial_sample_size=30,
    sort_attribute="power_output_maximum",
    sort_ascending=False,
    location_dist_type="uniform",  # equal probability per position (random cuts)
    demand_tolerance=0.20,         # ±20% capacity margins around demand
    crossover_operator="single_point",
    mutation_rate=0.02,
    max_generations=50,
    max_wall_seconds=120.0,
    stagnation_limit=10,
    solver="auto",
)


def load_instance(path: Path) -> tuple[dict, list[float]]:
    """Return (thermal_generators, demand_values)."""
    with open(path) as f:
        data = json.load(f)
    thermal = data["thermal_generators"]
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
    return thermal, demand_values


def run_single(thermal: dict, demand_values: list[float]) -> None:
    n_thermal = len(thermal)
    demand = demand_values[TIME_PERIOD_IDX]
    print(f"Running single period: t={TIME_PERIOD_IDX}  demand={demand:.1f} MW\n",
          flush=True)

    rng = np.random.default_rng(seed=42)
    pop, stats = run_stage1_ga(
        generators=thermal,
        demand=demand,
        config=config,
        rng=rng,
        time_period=TIME_PERIOD_IDX,
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


def run_all(thermal: dict, demand_values: list[float]) -> None:
    n_periods = len(demand_values)
    print(f"Running all {n_periods} periods in parallel  "
          f"(n_workers={N_WORKERS or 'auto'})\n", flush=True)

    result = run_all_periods(
        generators=thermal,
        demand_values=demand_values,
        config=config,
        n_workers=N_WORKERS,
        base_seed=42,
        show_progress=True,
    )

    result.print_summary()


def main() -> None:
    print(f"Loading instance: {INSTANCE_PATH}")
    thermal, demand_values = load_instance(INSTANCE_PATH)
    print(f"Instance: {len(thermal)} thermal generators, "
          f"{len(demand_values)} time periods\n", flush=True)

    if MODE == "single":
        run_single(thermal, demand_values)
    elif MODE == "all":
        run_all(thermal, demand_values)
    else:
        raise ValueError(f"Unknown MODE '{MODE}'. Use 'single' or 'all'.")


if __name__ == "__main__":
    main()
