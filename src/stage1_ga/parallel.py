"""
Parallel Stage 1 GA runner — one worker process per time period.

run_all_periods() distributes the per-period GA calls across a
ProcessPoolExecutor so all time periods run concurrently.

Worker isolation
----------------
Each worker gets its own NumPy Generator seeded from (base_seed + period_idx),
ensuring reproducible but independent random streams per period.  Pyomo solver
instances are created inside the worker process, so there are no shared
in-process resources between workers.

Logging in workers
------------------
Worker processes suppress logging output to avoid interleaved terminal noise.
The main process prints a single-line progress update as each period completes,
followed by a full per-period and aggregate summary at the end.
"""

from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from .config import GAConfig
from .ga import GAStats, run_stage1_ga
from .population import BoundedPopulation


# ── Worker (runs in a subprocess) ────────────────────────────────────────────

def _period_worker(
    args: tuple[int, dict, float, GAConfig, int],
) -> tuple[int, BoundedPopulation, GAStats]:
    """
    Worker entry point — must be a module-level function so it is picklable
    on Windows (which uses 'spawn' rather than 'fork').

    Parameters (packed into a single tuple for executor.map compatibility)
    ----------
    period_idx  : time period index (0-based)
    generators  : {name: gen_data} thermal generator dict
    demand      : demand (MW) for this period
    config      : GAConfig
    seed        : integer seed for this period's RNG
    """
    period_idx, generators, demand, config, seed = args

    # Silence all logging inside worker processes
    logging.disable(logging.CRITICAL)

    rng = np.random.default_rng(seed)
    pop, stats = run_stage1_ga(
        generators=generators,
        demand=demand,
        config=config,
        rng=rng,
        time_period=period_idx,
    )
    return period_idx, pop, stats


# ── Aggregate results ─────────────────────────────────────────────────────────

@dataclass
class AllPeriodsResult:
    """Collected output from a full run_all_periods() call."""
    populations: list[BoundedPopulation]   # indexed by time period
    period_stats: list[GAStats]            # indexed by time period
    demand_values: list[float]
    total_wall_seconds: float

    def print_summary(self) -> None:
        n = len(self.period_stats)
        feasible_periods = sum(
            1 for s in self.period_stats if math.isfinite(s.best_fitness)
        )
        total_ed = sum(s.n_ed_total for s in self.period_stats)
        total_infeas = sum(s.n_ed_infeasible for s in self.period_stats)
        best_fitnesses = [
            s.best_fitness for s in self.period_stats if math.isfinite(s.best_fitness)
        ]

        print(f"\n{'=' * 70}")
        print(f"  Stage 1 — All Periods Summary  ({n} periods)")
        print(f"{'=' * 70}")
        print(f"  Total wall time      : {self.total_wall_seconds:.1f}s")
        print(f"  Feasible periods     : {feasible_periods}/{n}")
        print(f"  ED solves (total)    : {total_ed}"
              f"   infeasible={total_infeas}"
              f"  ({100*total_infeas/total_ed:.1f}%)" if total_ed else "")
        if best_fitnesses:
            print(f"  Best fitness range   : "
                  f"{min(best_fitnesses):,.2f}  –  {max(best_fitnesses):,.2f}")
            print(f"  Total cost (sum)     : {sum(best_fitnesses):,.2f}")
        print()
        print(f"  {'Period':<8}  {'Demand (MW)':<13}  {'Best Cost ($)':<18}"
              f"  {'Committed':<12}  {'Gen':<6}  {'Wall (s)':<10}  Stop")
        print(f"  {'-'*8}  {'-'*13}  {'-'*18}  {'-'*12}  {'-'*6}  {'-'*10}  {'-'*20}")
        for t, (s, pop, demand) in enumerate(
            zip(self.period_stats, self.populations, self.demand_values)
        ):
            best = pop.best
            committed_str = (
                f"{best.n_committed}/{len(best.bits)}" if best else "N/A"
            )
            best_str = (
                f"{s.best_fitness:>18,.2f}" if math.isfinite(s.best_fitness)
                else f"{'No solution':>18}"
            )
            print(f"  {t:<8}  {demand:<13.1f}  {best_str}"
                  f"  {committed_str:<12}  {s.n_generations:<6}"
                  f"  {s.total_wall_seconds:<10.2f}  {s.stop_reason}")
        print(f"{'=' * 70}\n", flush=True)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_all_periods(
    generators: dict,
    demand_values: list[float],
    config: GAConfig,
    n_workers: int | None = None,
    base_seed: int = 42,
    show_progress: bool = True,
) -> AllPeriodsResult:
    """
    Run Stage 1 GA for every time period in parallel.

    Parameters
    ----------
    generators    : {name: gen_data} thermal generator dict.
    demand_values : demand (MW) for each time period, length = n_periods.
    config        : GAConfig shared across all periods.
    n_workers     : number of parallel worker processes.  Defaults to the
                    number of logical CPUs (os.cpu_count()).
    base_seed     : period t uses seed = base_seed + t for reproducibility.
    show_progress : print a one-liner to stdout as each period completes.

    Returns
    -------
    AllPeriodsResult
    """
    n_periods = len(demand_values)
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    worker_args = [
        (t, generators, demand_values[t], config, base_seed + t)
        for t in range(n_periods)
    ]

    # Pre-allocate result lists so we can insert by period index
    populations: list[BoundedPopulation | None] = [None] * n_periods
    period_stats: list[GAStats | None] = [None] * n_periods

    wall_start = time.monotonic()

    if show_progress:
        print(f"Launching {n_periods} periods across {n_workers} worker(s)…",
              flush=True)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_period_worker, args): args[0]
            for args in worker_args
        }
        completed = 0
        for future in as_completed(futures):
            period_idx, pop, stats = future.result()
            populations[period_idx] = pop
            period_stats[period_idx] = stats
            completed += 1
            if show_progress:
                best_str = (
                    f"{stats.best_fitness:,.2f}"
                    if math.isfinite(stats.best_fitness) else "infeasible"
                )
                elapsed = time.monotonic() - wall_start
                print(
                    f"  [{completed:>3}/{n_periods}]  t={period_idx:<4}"
                    f"  best={best_str:<18}  gen={stats.n_generations:<5}"
                    f"  {stats.total_wall_seconds:.1f}s worker"
                    f"  (wall {elapsed:.1f}s)",
                    flush=True,
                )

    total_wall = time.monotonic() - wall_start

    return AllPeriodsResult(
        populations=populations,        # type: ignore[arg-type]
        period_stats=period_stats,      # type: ignore[arg-type]
        demand_values=demand_values,
        total_wall_seconds=total_wall,
    )
