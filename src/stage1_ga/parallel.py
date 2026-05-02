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
    args: tuple[int, dict, float, GAConfig, int, float, float, int | None],
) -> tuple[int, BoundedPopulation, GAStats]:
    """
    Worker entry point — must be a module-level function so it is picklable
    on Windows (which uses 'spawn' rather than 'fork').

    Parameters (packed into a single tuple for executor.map compatibility)
    ----------
    period_idx         : time period index (0-based)
    generators         : {name: gen_data} thermal generator dict
    demand             : expected thermal demand (MW) for this period
    config             : GAConfig
    seed               : integer seed for this period's RNG
    reg_up_req         : MW of regulation-up the fleet must provide
    reg_down_req       : MW of regulation-down the fleet must provide
    target_n_committed : pre-solve commitment count target (None = disabled)
    """
    period_idx, generators, demand, config, seed, reg_up_req, reg_down_req, target_n_committed = args

    # Silence all logging inside worker processes
    logging.disable(logging.CRITICAL)

    rng = np.random.default_rng(seed)
    pop, stats = run_stage1_ga(
        generators=generators,
        demand=demand,
        config=config,
        rng=rng,
        time_period=period_idx,
        reg_up_req=reg_up_req,
        reg_down_req=reg_down_req,
        target_n_committed=target_n_committed,
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
        total_ed          = sum(s.n_ed_total         for s in self.period_stats)
        total_infeas      = sum(s.n_ed_infeasible    for s in self.period_stats)
        total_pmax        = sum(s.n_pmax_infeasible  for s in self.period_stats)
        total_pmin_aug    = sum(s.n_pmin_augmented   for s in self.period_stats)
        total_solver_inf  = sum(s.n_ed_solver_infeasible for s in self.period_stats)
        best_fitnesses = [
            s.best_fitness for s in self.period_stats if math.isfinite(s.best_fitness)
        ]

        print(f"\n{'=' * 90}")
        print(f"  Stage 1 — All Periods Summary  ({n} periods)")
        print(f"{'=' * 90}")
        print(f"  Total wall time      : {self.total_wall_seconds:.1f}s")
        print(f"  Feasible periods     : {feasible_periods}/{n}")
        if total_ed:
            print(f"  ED solves (total)    : {total_ed}"
                  f"  infeasible={total_infeas} ({100*total_infeas/total_ed:.1f}%)"
                  f"  pmin_augmented={total_pmin_aug}")
            if total_infeas:
                print(f"    infeasible detail  : pmax_ceiling={total_pmax}"
                      f"  solver={total_solver_inf}")
        if best_fitnesses:
            print(f"  Best fitness range   : "
                  f"{min(best_fitnesses):,.2f}  –  {max(best_fitnesses):,.2f}")
            print(f"  Total cost (sum)     : {sum(best_fitnesses):,.2f}")
        print()
        print(f"  {'t':<6}  {'Demand':>9}  {'Best Cost ($)':>18}"
              f"  {'Committed':>10}  {'pmax_inf':>9}  {'pmin_aug':>9}"
              f"  {'reg_inf':>8}  {'Gen':>5}  {'Wall':>6}  Stop")
        print(f"  {'-'*6}  {'-'*9}  {'-'*18}"
              f"  {'-'*10}  {'-'*9}  {'-'*9}"
              f"  {'-'*8}  {'-'*5}  {'-'*6}  {'-'*20}")
        for t, (s, pop, demand) in enumerate(
            zip(self.period_stats, self.populations, self.demand_values)
        ):
            best = pop.best
            committed_str = f"{best.n_committed}/{len(best.bits)}" if best else "N/A"
            best_str = (
                f"{s.best_fitness:>18,.2f}" if math.isfinite(s.best_fitness)
                else f"{'No solution':>18}"
            )
            print(f"  {t:<6}  {demand:>9.1f}  {best_str}"
                  f"  {committed_str:>10}  {s.n_pmax_infeasible:>9}"
                  f"  {s.n_pmin_augmented:>9}  {s.n_reg_infeasible:>8}"
                  f"  {s.n_generations:>5}  {s.total_wall_seconds:>6.1f}  {s.stop_reason}")
        print(f"{'=' * 90}\n", flush=True)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_all_periods(
    generators: dict,
    demand_values: list[float],
    config: GAConfig,
    n_workers: int | None = None,
    base_seed: int = 42,
    show_progress: bool = True,
    reg_up_reqs: list[float] | None = None,
    reg_down_reqs: list[float] | None = None,
    target_n_committed: list[int] | None = None,
) -> AllPeriodsResult:
    """
    Run Stage 1 GA for every time period in parallel.

    Parameters
    ----------
    generators         : {name: gen_data} thermal generator dict.
    demand_values      : expected thermal demand (MW) per time period.
    config             : GAConfig shared across all periods.
    n_workers          : number of parallel worker processes.  Defaults to all CPUs.
    base_seed          : period t uses seed = base_seed + t for reproducibility.
    show_progress      : print a one-liner to stdout as each period completes.
    reg_up_reqs        : MW of regulation-up required per period (renewable drop risk).
                         Defaults to zero for all periods if None.
    reg_down_reqs      : MW of regulation-down required per period (renewable surge risk).
                         Defaults to zero for all periods if None.
    target_n_committed : Pre-solve target commitment count per period.  When provided,
                         each period's GA stops early once its best feasible chromosome
                         is within config.target_n_committed_tolerance units of the target.
                         Pass None to disable this stopping criterion.

    Returns
    -------
    AllPeriodsResult
    """
    n_periods = len(demand_values)
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    _reg_up   = reg_up_reqs   if reg_up_reqs   is not None else [0.0] * n_periods
    _reg_down = reg_down_reqs if reg_down_reqs is not None else [0.0] * n_periods
    _targets  = target_n_committed if target_n_committed is not None else [None] * n_periods

    worker_args = [
        (t, generators, demand_values[t], config, base_seed + t, _reg_up[t], _reg_down[t], _targets[t])
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
