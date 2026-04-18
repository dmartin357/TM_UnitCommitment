"""
Smoke test for Stage 2 Graph Builder.

Loads a saved Stage 1 result from disk, then runs build_graph() and reports
graph statistics and per-period edge diagnostics.

STAGE1_RESULT_PATH controls where the Stage 1 JSON is read from.  If the file
does not exist and MODE is 'load_or_run', Stage 1 is run automatically and the
result is saved before proceeding to Stage 2.

Modes (set MODE below):
  'load_or_run'  — load saved Stage 1 result if it exists, otherwise run
                   Stage 1 first and save it, then run Stage 2.  (default)
  'run_and_save' — always re-run Stage 1, overwrite the saved file, then
                   run Stage 2.  Use this to refresh the saved result.
  'load_only'    — load saved result and fail fast if the file is missing.
                   Use this when you know Stage 1 is already cached.

Usage (from repo root, with power-systems conda env active):
    python smoke_test_stage2.py
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from src.io.stage1_io import load_stage1_result, save_stage1_result
from src.stage1_ga.config import GAConfig
from src.stage1_ga.parallel import AllPeriodsResult, run_all_periods
from src.stage2_graph.config import GraphBuilderConfig
from src.stage2_graph.graph_builder import build_graph

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PGLIB_UC_ROOT      = Path("C:/gitrepos/power-grid-lib/pglib-uc")
INSTANCE_PATH      = PGLIB_UC_ROOT / "rts_gmlc" / "2020-01-27.json"
STAGE1_RESULT_PATH = Path("results/stage1_rts_gmlc_2020-01-27.json")

# ── Mode ──────────────────────────────────────────────────────────────────────
MODE = "load_or_run"   # 'load_or_run' | 'run_and_save' | 'load_only'

# ── Stage 1 config (used only when Stage 1 is run) ───────────────────────────
# Keep in sync with smoke_test_stage1.py so the cached result is comparable.
STAGE1_CONFIG = GAConfig(
    population_size=50,
    initial_sample_size=30,
    sort_attribute="power_output_maximum",
    sort_ascending=False,
    location_dist_type="uniform",
    demand_tolerance=0.20,
    crossover_operator="single_point",
    mutation_rate=0.02,
    max_generations=50,
    max_wall_seconds=120.0,
    stagnation_limit=10,
    solver="auto",
)
STAGE1_N_WORKERS = None   # None → all logical CPUs

# ── Stage 2 config ────────────────────────────────────────────────────────────
STAGE2_CONFIG = GraphBuilderConfig(
    enable_unit_rectification=True,
    rectification_multiplier=2.0,
    enable_net_adjustment_check=False,   # disabled to observe its impact
    net_adjustment_tolerance=0.05,
)


# ── Instance loader ───────────────────────────────────────────────────────────

def load_instance(path: Path) -> tuple[dict, list[float]]:
    """Return (thermal_generators, demand_values) from a pglib-uc JSON file."""
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


# ── Stage 1 helpers ───────────────────────────────────────────────────────────

def run_and_save_stage1(thermal: dict, demand_values: list[float]) -> AllPeriodsResult:
    """Run Stage 1 for all periods, save result, and return AllPeriodsResult."""
    n_periods = len(demand_values)
    print(f"Running Stage 1: {n_periods} periods  "
          f"(n_workers={STAGE1_N_WORKERS or 'auto'})\n", flush=True)

    result = run_all_periods(
        generators=thermal,
        demand_values=demand_values,
        config=STAGE1_CONFIG,
        n_workers=STAGE1_N_WORKERS,
        base_seed=42,
        show_progress=True,
    )
    result.print_summary()

    STAGE1_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    saved = save_stage1_result(result, STAGE1_RESULT_PATH)
    print(f"Stage 1 result saved → {saved}\n", flush=True)
    return result


def get_stage1_result(thermal: dict, demand_values: list[float]) -> AllPeriodsResult:
    """Return AllPeriodsResult according to MODE."""
    if MODE == "run_and_save":
        return run_and_save_stage1(thermal, demand_values)

    if MODE == "load_only":
        if not STAGE1_RESULT_PATH.exists():
            raise FileNotFoundError(
                f"Stage 1 result not found: {STAGE1_RESULT_PATH}\n"
                "Run with MODE='load_or_run' or 'run_and_save' to generate it."
            )
        print(f"Loading Stage 1 result from {STAGE1_RESULT_PATH} …", flush=True)
        result = load_stage1_result(STAGE1_RESULT_PATH)
        print(f"Loaded: {len(result.populations)} periods\n", flush=True)
        return result

    # MODE == 'load_or_run'
    if STAGE1_RESULT_PATH.exists():
        print(f"Loading Stage 1 result from {STAGE1_RESULT_PATH} …", flush=True)
        result = load_stage1_result(STAGE1_RESULT_PATH)
        print(f"Loaded: {len(result.populations)} periods\n", flush=True)
        return result

    print(f"No saved Stage 1 result at {STAGE1_RESULT_PATH} — running Stage 1 first.\n",
          flush=True)
    return run_and_save_stage1(thermal, demand_values)


# ── Stage 2 diagnostics ───────────────────────────────────────────────────────

def print_graph_diagnostics(G, populations, demand_values) -> None:
    """Print per-period edge counts and flag any disconnected transitions."""
    n_periods = len(populations)

    # Count edges per period transition and track nodes with edges
    edges_per_transition: dict[int, int] = defaultdict(int)
    nodes_with_out_edge:  dict[int, set] = defaultdict(set)
    nodes_with_in_edge:   dict[int, set] = defaultdict(set)

    for u, v in G.edges():
        period_from = u[0]
        edges_per_transition[period_from] += 1
        nodes_with_out_edge[period_from].add(u)
        nodes_with_in_edge[period_from + 1].add(v)

    # Count feasible chromosomes per period
    feasible_per_period = [
        sum(1 for c in pop if c.is_feasible())
        for pop in populations
    ]

    print(f"\n{'=' * 76}")
    print(f"  Stage 2 — Per-Period Edge Diagnostics")
    print(f"{'=' * 76}")
    print(f"  {'Trans':<8}  {'Demand[i+1]':>11}  {'Edges':>7}  "
          f"{'Src w/ edge':>11}  {'Dst w/ edge':>11}  {'Feasible[i]':>11}  Note")
    print(f"  {'-'*8}  {'-'*11}  {'-'*7}  {'-'*11}  {'-'*11}  {'-'*11}  {'-'*15}")

    any_disconnected = False
    for i in range(n_periods - 1):
        n_edges    = edges_per_transition.get(i, 0)
        n_src      = len(nodes_with_out_edge.get(i, set()))
        n_dst      = len(nodes_with_in_edge.get(i + 1, set()))
        n_feasible = feasible_per_period[i]
        demand_i1  = demand_values[i + 1]
        note       = "  *** NO EDGES — Stage 3 will fail ***" if n_edges == 0 else ""
        if n_edges == 0:
            any_disconnected = True

        print(f"  {i}→{i+1:<5}  {demand_i1:>11.1f}  {n_edges:>7,}  "
              f"{n_src:>11}  {n_dst:>11}  {n_feasible:>11}  {note}")

    print(f"{'=' * 76}")

    if any_disconnected:
        print("\n  WARNING: One or more transitions have no edges.")
        print("  Stage 3 will be unable to find complete paths.")
        print("  Consider relaxing GraphBuilderConfig parameters.\n")
    else:
        print("\n  All transitions have at least one edge — graph is traversable.\n")

    # Best-case cost estimate: sum of best chromosome fitness per period
    best_costs = []
    for pop in populations:
        b = pop.best
        if b is not None and b.is_feasible():
            best_costs.append(b.fitness)

    if best_costs:
        print(f"  Best chromosome cost per period (no startup costs):")
        print(f"    Sum across all periods : ${sum(best_costs):>18,.2f}")
        print(f"    Min per-period cost    : ${min(best_costs):>18,.2f}")
        print(f"    Max per-period cost    : ${max(best_costs):>18,.2f}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading instance: {INSTANCE_PATH}")
    thermal, demand_values = load_instance(INSTANCE_PATH)
    print(f"Instance: {len(thermal)} thermal generators, "
          f"{len(demand_values)} time periods\n", flush=True)

    # Stage 1
    result = get_stage1_result(thermal, demand_values)

    # Stage 2
    print("Running Stage 2 Graph Builder …\n", flush=True)
    G, stats = build_graph(
        populations=result.populations,
        generators=thermal,
        demand_values=result.demand_values,
        config=STAGE2_CONFIG,
    )

    stats.print_summary()
    print_graph_diagnostics(G, result.populations, result.demand_values)


if __name__ == "__main__":
    main()
