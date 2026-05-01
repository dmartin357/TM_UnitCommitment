"""
Pre-solve stage parallel runner — one worker process per time period.

run_all_periods() distributes per-period sampling calls across a
ProcessPoolExecutor.  Each period gets an independent RNG seeded from
(base_seed + period_idx) for reproducibility.

Because there is no optimization solve in this stage, the workload per
period is pure Python + NumPy.  Parallelism is still worthwhile for large
horizon lengths (e.g., 48 periods × 1 000+ samples each).
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from .config import PreSolveConfig
from .sampler import PreSolvePeriodResult, run_pre_solve_period


# ── Worker (module-level for Windows 'spawn' compatibility) ──────────────────

def _period_worker(
    args: tuple[int, dict, float, float, float, PreSolveConfig, int],
) -> tuple[int, PreSolvePeriodResult]:
    period_idx, generators, thermal_demand, reg_up_req, reg_down_req, config, seed = args
    logging.disable(logging.CRITICAL)
    result = run_pre_solve_period(
        generators=generators,
        thermal_demand=thermal_demand,
        reg_up_req=reg_up_req,
        reg_down_req=reg_down_req,
        config=config,
        seed=seed,
    )
    return period_idx, result


# ── Aggregate result ──────────────────────────────────────────────────────────

@dataclass
class PreSolveResult:
    """Collected output from a full run_all_periods() call."""
    period_results: list[PreSolvePeriodResult]    # indexed by time period
    gen_names: list[str]                           # canonical sorted order
    thermal_demand_values: list[float]
    reg_up_req_values: list[float]
    reg_down_req_values: list[float]
    total_wall_seconds: float
    target_percentile: float                       # percentile used for startup/shutdown targets

    # ── Derived helpers ───────────────────────────────────────────────────────

    def target_n_committed(self) -> list[int]:
        """Target n_committed per period at the configured percentile."""
        return [pr.target_n_committed(self.target_percentile) for pr in self.period_results]

    def startup_shutdown_targets(self) -> list[tuple[int, int]]:
        """
        Return (startup_target, shutdown_target) for each transition t → t+1.
        List length = n_periods - 1.
        """
        targets = self.target_n_committed()
        result = []
        for t in range(len(targets) - 1):
            delta = targets[t + 1] - targets[t]
            result.append((max(0, delta), max(0, -delta)))
        return result

    def commit_frequency_matrix(self) -> dict[str, list[float]]:
        """Return {unit_name: [freq_t0, freq_t1, ...]} for all time periods."""
        return {
            name: [pr.commit_frequency.get(name, 0.0) for pr in self.period_results]
            for name in self.gen_names
        }

    def mean_frequency(self) -> dict[str, float]:
        """Mean commitment frequency for each unit across all periods."""
        matrix = self.commit_frequency_matrix()
        n = len(self.period_results)
        if n == 0:
            return {name: 0.0 for name in self.gen_names}
        return {name: sum(freqs) / n for name, freqs in matrix.items()}

    # ── Printing ──────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        n = len(self.period_results)
        targets = self.target_n_committed()
        transitions = self.startup_shutdown_targets()

        print(f"\n{'=' * 100}")
        print(f"  Pre-Solve Stage — Distribution of n_committed  ({n} periods, "
              f"{self.period_results[0].total_samples:,} samples/period, "
              f"target=p{self.target_percentile:.0f})")
        print(f"{'=' * 100}")
        print(f"  {'t':<5}  {'TherDem':>8}  {'RupReq':>7}  {'RdnReq':>7}  "
              f"{'Mean':>6}  {'P10':>5}  {'P25':>5}  {'Med':>5}  {'P75':>5}  {'P90':>5}  "
              f"{'Target':>7}  {'Wall':>6}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*7}  "
              f"{'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  "
              f"{'-'*7}  {'-'*6}")

        for t, pr in enumerate(self.period_results):
            print(
                f"  {t:<5}  {pr.thermal_demand:>8.1f}  {pr.reg_up_req:>7.1f}  "
                f"{pr.reg_down_req:>7.1f}  "
                f"{pr.mean_n_committed:>6.1f}  "
                f"{pr.percentile_n_committed(10):>5.1f}  "
                f"{pr.percentile_n_committed(25):>5.1f}  "
                f"{pr.median_n_committed:>5.1f}  "
                f"{pr.percentile_n_committed(75):>5.1f}  "
                f"{pr.percentile_n_committed(90):>5.1f}  "
                f"{targets[t]:>7}  "
                f"{pr.wall_seconds:>6.3f}s"
            )

        print(f"{'=' * 100}")
        print(f"\n  Total wall time: {self.total_wall_seconds:.2f}s\n")

        print(f"  {'=' * 52}")
        print(f"  Startup / Shutdown Targets between Periods")
        print(f"  {'=' * 52}")
        print(f"  {'Transition':<14}  {'Target[t]':>9}  {'Target[t+1]':>11}  "
              f"{'Startups':>9}  {'Shutdowns':>10}")
        print(f"  {'-'*14}  {'-'*9}  {'-'*11}  {'-'*9}  {'-'*10}")
        for t, (su, sd) in enumerate(transitions):
            print(f"  {t}→{t+1:<12}  {targets[t]:>9}  {targets[t+1]:>11}  "
                  f"{su:>9}  {sd:>10}")
        print(f"  {'=' * 52}\n", flush=True)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_all_periods(
    generators: dict,
    thermal_demand_values: list[float],
    reg_up_req_values: list[float],
    reg_down_req_values: list[float],
    config: PreSolveConfig,
    n_workers: int | None = None,
    base_seed: int = 42,
    show_progress: bool = True,
) -> PreSolveResult:
    """
    Run the pre-solve stage sampler for every time period in parallel.

    Parameters
    ----------
    generators            : {name: gen_data} thermal generator dict.
    thermal_demand_values : expected thermal demand per period (MW).
    reg_up_req_values     : reg-up requirement per period (MW).
    reg_down_req_values   : reg-down requirement per period (MW).
    config                : PreSolveConfig shared across all periods.
    n_workers             : parallel workers.  None → all logical CPUs.
    base_seed             : period t uses seed = base_seed + t.
    show_progress         : print a one-liner to stdout as each period completes.

    Returns
    -------
    PreSolveResult
    """
    n_periods = len(thermal_demand_values)
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    worker_args = [
        (t, generators, thermal_demand_values[t],
         reg_up_req_values[t], reg_down_req_values[t],
         config, base_seed + t)
        for t in range(n_periods)
    ]

    period_results: list[PreSolvePeriodResult | None] = [None] * n_periods
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
            period_idx, result = future.result()
            period_results[period_idx] = result
            completed += 1
            if show_progress:
                elapsed = time.monotonic() - wall_start
                print(
                    f"  [{completed:>3}/{n_periods}]  t={period_idx:<4}"
                    f"  median_committed={result.median_n_committed:.1f}"
                    f"  mean_committed={result.mean_n_committed:.1f}"
                    f"  {result.wall_seconds:.3f}s worker"
                    f"  (wall {elapsed:.1f}s)",
                    flush=True,
                )

    total_wall = time.monotonic() - wall_start
    gen_names = list(period_results[0].commit_counts.keys())  # type: ignore[union-attr]

    return PreSolveResult(
        period_results=period_results,            # type: ignore[arg-type]
        gen_names=gen_names,
        thermal_demand_values=thermal_demand_values,
        reg_up_req_values=reg_up_req_values,
        reg_down_req_values=reg_down_req_values,
        total_wall_seconds=total_wall,
        target_percentile=config.target_percentile,
    )
