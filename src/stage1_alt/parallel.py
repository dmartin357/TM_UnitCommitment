"""
Alternative Stage 1 parallel runner — one worker process per time period.

run_all_periods_alt() distributes per-period sampling calls across a
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

from .config import AltStage1Config
from .sampler import AltStage1PeriodResult, run_alt_stage1_period


# ── Worker (module-level for Windows 'spawn' compatibility) ──────────────────

def _period_worker(
    args: tuple[int, dict, float, float, float, AltStage1Config, int],
) -> tuple[int, AltStage1PeriodResult]:
    period_idx, generators, demand, demand_upper, demand_lower, config, seed = args
    logging.disable(logging.CRITICAL)
    result = run_alt_stage1_period(
        generators=generators,
        demand=demand,
        demand_upper=demand_upper,
        demand_lower=demand_lower,
        config=config,
        seed=seed,
    )
    return period_idx, result


# ── Aggregate result ──────────────────────────────────────────────────────────

@dataclass
class AltStage1Result:
    """Collected output from a full run_all_periods_alt() call."""
    period_results: list[AltStage1PeriodResult]   # indexed by time period
    gen_names: list[str]                           # canonical sorted order
    demand_values: list[float]                     # expected total demand per period
    demand_upper_values: list[float]               # max non-renewable demand per period
    demand_lower_values: list[float]               # min non-renewable demand per period
    total_wall_seconds: float

    def commit_frequency_matrix(self) -> dict[str, list[float]]:
        """
        Return {unit_name: [freq_t0, freq_t1, ...]} for all time periods.
        Convenient for tabular display or CSV export.
        """
        return {
            name: [pr.commit_frequency.get(name, 0.0) for pr in self.period_results]
            for name in self.gen_names
        }

    def mean_frequency(self) -> dict[str, float]:
        """Return mean commitment frequency for each unit across all periods."""
        matrix = self.commit_frequency_matrix()
        n = len(self.period_results)
        if n == 0:
            return {name: 0.0 for name in self.gen_names}
        return {name: sum(freqs) / n for name, freqs in matrix.items()}

    def print_summary(self) -> None:
        n = len(self.period_results)
        total_samples = sum(pr.total_samples for pr in self.period_results)
        avg_committed = sum(pr.mean_committed for pr in self.period_results) / max(n, 1)

        print(f"\n{'=' * 90}")
        print(f"  Alt Stage 1 — All Periods Summary  ({n} periods)")
        print(f"{'=' * 90}")
        print(f"  Total wall time      : {self.total_wall_seconds:.2f}s")
        print(f"  Total samples        : {total_samples:,}  ({n} periods × "
              f"{self.period_results[0].total_samples if self.period_results else 0:,}/period)")
        print(f"  Avg committed/sample : {avg_committed:.1f} generators")
        print()
        print(f"  {'Period':<8}  {'Demand':>10}  {'DemUpper':>10}  {'DemLower':>10}  "
              f"{'UpperThr':>10}  {'LowerThr':>10}  {'AvgCommit':>10}  {'Wall (s)':<10}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  "
              f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        for t, pr in enumerate(self.period_results):
            print(f"  {t:<8}  {pr.demand:>10.1f}  {pr.demand_upper:>10.1f}  "
                  f"{pr.demand_lower:>10.1f}  {pr.upper_threshold:>10.1f}  "
                  f"{pr.lower_threshold:>10.1f}  {pr.mean_committed:>10.1f}  "
                  f"{pr.wall_seconds:<10.3f}")
        print(f"{'=' * 90}\n", flush=True)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_all_periods_alt(
    generators: dict,
    demand_values: list[float],
    demand_upper_values: list[float],
    demand_lower_values: list[float],
    config: AltStage1Config,
    n_workers: int | None = None,
    base_seed: int = 42,
    show_progress: bool = True,
) -> AltStage1Result:
    """
    Run the alternative Stage 1 sampler for every time period in parallel.

    Parameters
    ----------
    generators          : {name: gen_data} thermal generator dict.
    demand_values       : expected total demand (MW) per period.
    demand_upper_values : max non-renewable demand per period
                          (= expected_demand - min_renewable).
    demand_lower_values : min non-renewable demand per period
                          (= expected_demand - max_renewable).
    config              : AltStage1Config shared across all periods.
    n_workers           : parallel workers.  None → all logical CPUs.
    base_seed           : period t uses seed = base_seed + t.
    show_progress       : print a one-liner to stdout as each period completes.

    Returns
    -------
    AltStage1Result
    """
    n_periods = len(demand_values)
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    worker_args = [
        (t, generators, demand_values[t], demand_upper_values[t],
         demand_lower_values[t], config, base_seed + t)
        for t in range(n_periods)
    ]

    period_results: list[AltStage1PeriodResult | None] = [None] * n_periods
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
                    f"  avg_committed={result.mean_committed:.1f}"
                    f"  {result.wall_seconds:.3f}s worker"
                    f"  (wall {elapsed:.1f}s)",
                    flush=True,
                )

    total_wall = time.monotonic() - wall_start

    # Derive canonical gen_names from sort order used in period 0 result
    gen_names = list(period_results[0].commit_counts.keys())  # type: ignore[union-attr]

    return AltStage1Result(
        period_results=period_results,       # type: ignore[arg-type]
        gen_names=gen_names,
        demand_values=demand_values,
        demand_upper_values=demand_upper_values,
        demand_lower_values=demand_lower_values,
        total_wall_seconds=total_wall,
    )
