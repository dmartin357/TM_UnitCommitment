"""
Stage 1 — Genetic Algorithm (per time period).

run_stage1_ga() is the main entry point.  It:
  1. Generates an initial population via CDF/PDF-based sampling.
  2. Evaluates each chromosome with the Economic Dispatch (ED) subproblem.
  3. Iterates: select parents → crossover → mutate → evaluate → update population.
  4. Returns the final BoundedPopulation and a GAStats summary.

Stopping criteria (first to trigger wins):
  - max_generations reached
  - max_wall_seconds elapsed (0 = disabled)
  - stagnation_limit consecutive generations without improvement to best fitness

All three are currently placeholders with simple implementations; refine as needed.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .chromosome import Chromosome
from .config import GAConfig
from .ed.piecewise_linear import EDInfeasible, solve_ed_piecewise_linear
from .initial_population.generator import InitialPopulationGenerator
from .operators import CROSSOVER_REGISTRY
from .operators.mutation import bit_flip
from .population import BoundedPopulation

logger = logging.getLogger(__name__)


# ── Run statistics ────────────────────────────────────────────────────────────

@dataclass
class GAStats:
    """Collected metrics from a single run_stage1_ga call."""

    # Overall timing
    total_wall_seconds: float = 0.0

    # Generation counts
    n_generations: int = 0
    n_seed_chromosomes: int = 0     # generated in Phase 1 (CDF sampling)
    n_seed_evaluated: int = 0       # unique, actually solved in Phase 1
    n_offspring_evaluated: int = 0  # unique offspring actually solved in Phase 2
    n_duplicates_skipped: int = 0   # hash-matched, never re-evaluated

    # ED outcomes
    n_ed_feasible: int = 0          # ED returned a finite cost
    n_ed_infeasible: int = 0        # ED raised EDInfeasible
    n_reg_infeasible: int = 0       # ED feasible but failed reg up/down requirement

    # Per-generation timing (seconds)
    gen_wall_times: list[float] = field(default_factory=list)

    # Fitness trajectory: (generation, best_fitness) after each generation
    fitness_history: list[tuple[int, float]] = field(default_factory=list)

    # Stop reason
    stop_reason: str = ""

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def n_ed_total(self) -> int:
        return self.n_ed_feasible + self.n_ed_infeasible + self.n_reg_infeasible

    @property
    def mean_gen_wall_seconds(self) -> float:
        return float(np.mean(self.gen_wall_times)) if self.gen_wall_times else 0.0

    @property
    def median_gen_wall_seconds(self) -> float:
        return float(np.median(self.gen_wall_times)) if self.gen_wall_times else 0.0

    @property
    def best_fitness(self) -> float:
        if not self.fitness_history:
            return math.inf
        return self.fitness_history[-1][1]

    def print_summary(self, time_period: int | None = None) -> None:
        """Print a human-readable summary to stdout."""
        header = "Stage 1 GA Summary"
        if time_period is not None:
            header += f" — Time Period {time_period}"
        print(f"\n{'=' * 60}")
        print(f"  {header}")
        print(f"{'=' * 60}")
        print(f"  Stop reason         : {self.stop_reason}")
        print(f"  Total wall time     : {self.total_wall_seconds:.2f}s")
        print(f"  Generations         : {self.n_generations}")
        if self.gen_wall_times:
            print(f"  Time/generation     : mean={self.mean_gen_wall_seconds*1000:.1f}ms"
                  f"  median={self.median_gen_wall_seconds*1000:.1f}ms"
                  f"  max={max(self.gen_wall_times)*1000:.1f}ms")
        print(f"  Seed chromosomes    : {self.n_seed_evaluated} evaluated"
              f" ({self.n_seed_chromosomes} sampled)")
        print(f"  Offspring evaluated : {self.n_offspring_evaluated}")
        print(f"  Duplicates skipped  : {self.n_duplicates_skipped}")
        print(f"  ED solves (total)   : {self.n_ed_total}"
              f"  feasible={self.n_ed_feasible}"
              f"  ed_infeasible={self.n_ed_infeasible}"
              f"  reg_infeasible={self.n_reg_infeasible}")
        if self.n_ed_total > 0:
            infeas_pct = 100.0 * self.n_ed_infeasible / self.n_ed_total
            print(f"  Infeasibility rate  : {infeas_pct:.1f}%")
        best = self.best_fitness
        print(f"  Best fitness (cost) : {best:,.2f}" if math.isfinite(best) else
              f"  Best fitness (cost) : No feasible solution found")
        print(f"{'=' * 60}\n", flush=True)


# ── Public entry point ────────────────────────────────────────────────────────

def run_stage1_ga(
    generators: dict,
    demand: float,
    config: GAConfig,
    rng: np.random.Generator | None = None,
    time_period: int | None = None,
    reg_up_req: float = 0.0,
    reg_down_req: float = 0.0,
) -> tuple[BoundedPopulation, GAStats]:
    """
    Run the Stage 1 GA for a single time period.

    Parameters
    ----------
    generators  : {name: gen_data} from instance JSON (thermal generators only).
    demand      : expected thermal MW demand for this time period.
    config      : GAConfig instance.
    rng         : NumPy random Generator (a new one is created if None).
    time_period : Optional period index, used only for logging labels.
    reg_up_req  : MW of regulation-up the committed fleet must provide (renewable drop risk).
    reg_down_req: MW of regulation-down the committed fleet must provide (renewable surge risk).

    Returns
    -------
    (BoundedPopulation, GAStats)
    """
    if rng is None:
        rng = np.random.default_rng()

    stats = GAStats()
    run_start = time.monotonic()

    period_label = f"[t={time_period}] " if time_period is not None else ""

    # Pre-compute must-run mask using the stable generator ordering from
    # InitialPopulationGenerator.
    init_gen = InitialPopulationGenerator(
        generators=generators,
        sort_attribute=config.sort_attribute,
        sort_ascending=config.sort_ascending,
        location_dist_type=config.location_dist_type,
        demand=demand,
        reg_up_req=reg_up_req,
        reg_down_req=reg_down_req,
    )

    sorted_names = init_gen.sorted_names
    must_run_mask = np.array(
        [generators[n].get("must_run", 0) == 1 for n in sorted_names],
        dtype=bool,
    )

    crossover_fn = CROSSOVER_REGISTRY[config.crossover_operator]
    pop = BoundedPopulation(config.population_size)

    # ── Phase 1: seed population via CDF/PDF sampling ─────────────────────────
    logger.info("%sSeeding initial population (%d samples)…",
                period_label, config.initial_sample_size)

    for _ in range(config.initial_sample_size):
        chrom = init_gen.generate(rng)
        stats.n_seed_chromosomes += 1
        if pop.seen(chrom):
            stats.n_duplicates_skipped += 1
            continue
        _evaluate(chrom, generators, demand, config.solver, stats, reg_up_req, reg_down_req)
        stats.n_seed_evaluated += 1
        pop.add(chrom)

    seed_best = _pop_best_fitness(pop)
    logger.info(
        "%sInitial population: %d/%d chromosomes, best=%.2f",
        period_label, len(pop), config.population_size, seed_best,
    )

    # ── Phase 2: GA loop ──────────────────────────────────────────────────────
    generation = 0
    stagnation_count = 0
    best_fitness = seed_best
    loop_start = time.monotonic()

    # Record baseline fitness before any GA generations
    stats.fitness_history.append((0, best_fitness))

    while True:
        stop, reason = _check_stop(generation, stagnation_count, loop_start, config)
        if stop:
            stats.stop_reason = reason
            break

        gen_start = time.monotonic()
        generation += 1

        # Select two parents (tournament selection from feasible pool)
        feasible = pop.feasible()
        if len(feasible) < 2:
            # Not enough feasible chromosomes yet — keep seeding
            chrom = init_gen.generate(rng)
            stats.n_seed_chromosomes += 1
            if not pop.seen(chrom):
                _evaluate(chrom, generators, demand, config.solver, stats, reg_up_req, reg_down_req)
                stats.n_seed_evaluated += 1
                pop.add(chrom)
            else:
                stats.n_duplicates_skipped += 1
            stats.gen_wall_times.append(time.monotonic() - gen_start)
            continue

        parent_a, parent_b = _tournament_select(feasible, k=3, rng=rng)

        # Crossover → Mutation → Evaluate
        child_a, child_b = crossover_fn(parent_a, parent_b, must_run_mask, rng)
        child_a = bit_flip(child_a, config.mutation_rate, must_run_mask, rng)
        child_b = bit_flip(child_b, config.mutation_rate, must_run_mask, rng)

        for child in (child_a, child_b):
            if pop.seen(child):
                stats.n_duplicates_skipped += 1
            else:
                _evaluate(child, generators, demand, config.solver, stats, reg_up_req, reg_down_req)
                stats.n_offspring_evaluated += 1
                pop.add(child)

        # Track stagnation and fitness history
        current_best = _pop_best_fitness(pop)
        stats.fitness_history.append((generation, current_best))
        if current_best < best_fitness - 1e-6:
            best_fitness = current_best
            stagnation_count = 0
        else:
            stagnation_count += 1

        gen_elapsed = time.monotonic() - gen_start
        stats.gen_wall_times.append(gen_elapsed)

        if generation % 10 == 0:
            logger.debug(
                "%sGen %d | pop=%d/%d | best=%.2f | stagnation=%d | gen_time=%.1fms",
                period_label, generation, len(pop), config.population_size,
                best_fitness, stagnation_count, gen_elapsed * 1000,
            )

    stats.n_generations = generation
    stats.total_wall_seconds = time.monotonic() - run_start

    logger.info(
        "%sGA done: %d gen | best=%.2f | wall=%.2fs | stop=%s",
        period_label, generation, best_fitness,
        stats.total_wall_seconds, stats.stop_reason,
    )
    return pop, stats


# ── Internal helpers ──────────────────────────────────────────────────────────

def _pop_best_fitness(pop: BoundedPopulation) -> float:
    """Return the best fitness in pop as a float, or inf if empty or unevaluated."""
    b = pop.best
    return b.fitness if (b is not None and b.fitness is not None) else math.inf


def _evaluate(
    chrom: Chromosome,
    generators: dict,
    demand: float,
    solver: str,
    stats: GAStats,
    reg_up_req: float = 0.0,
    reg_down_req: float = 0.0,
) -> None:
    """
    Run ED and set chrom.fitness / chrom.dispatch / reg_up / reg_down in-place.

    After a feasible ED solve, reg-up and reg-down are computed and stored for
    reporting.  Shortfalls are counted in stats but do not mark the chromosome
    infeasible — the GA optimises on cost, and the CSV margin columns show any
    reserve shortfall.  Renewable curtailment covers reg-down shortfalls; reg-up
    shortfalls at the period level may be resolved by Stage 2 chromosome selection.

    If the committed fleet's aggregate pmin exceeds the expected thermal demand,
    the ED demand is augmented to sum(pmin) so the problem remains feasible.  The
    surplus (renewable_loss) represents renewable energy that must be curtailed to
    accommodate the thermal minimum-generation floor.
    """
    # Augment demand to committed pmin if necessary
    committed_pmin = sum(
        float(generators[name].get("power_output_minimum", 0.0))
        for name in chrom.committed
    )
    effective_demand = max(demand, committed_pmin)
    chrom.renewable_loss = effective_demand - demand   # 0 when no augmentation needed

    try:
        cost, dispatch = solve_ed_piecewise_linear(
            committed_names=chrom.committed,
            generators=generators,
            demand=effective_demand,
            solver=solver,
        )
        chrom.fitness  = cost
        chrom.dispatch = dispatch
        chrom.reg_up, chrom.reg_down = _compute_regulation(chrom, generators)
        if chrom.reg_up < reg_up_req:
            stats.n_reg_infeasible += 1  # shortfall noted; chromosome still competes on cost
        else:
            stats.n_ed_feasible += 1
    except EDInfeasible:
        chrom.fitness  = math.inf
        chrom.dispatch = None
        chrom.reg_up   = None
        chrom.reg_down = None
        stats.n_ed_infeasible += 1


def _compute_regulation(
    chrom: Chromosome,
    generators: dict,
) -> tuple[float, float]:
    """
    Return (total_reg_up, total_reg_down) in MW for the chromosome's committed set.

    Per unit:
      reg_up   = min(ramp_up_limit,   pmax - dispatch)
      reg_down = min(ramp_down_limit, dispatch - pmin)
    """
    total_up = total_down = 0.0
    dispatch = chrom.dispatch or {}
    for name in chrom.committed:
        gen   = generators[name]
        mw    = dispatch.get(name, 0.0)
        pmax  = float(gen.get("power_output_maximum", 0.0))
        pmin  = float(gen.get("power_output_minimum", 0.0))
        r_up  = float(gen.get("ramp_up_limit",   0.0))
        r_dn  = float(gen.get("ramp_down_limit", 0.0))
        total_up   += min(r_up, max(0.0, pmax - mw))
        total_down += min(r_dn, max(0.0, mw  - pmin))
    return total_up, total_down


def _tournament_select(
    feasible: list[Chromosome],
    k: int,
    rng: np.random.Generator,
) -> tuple[Chromosome, Chromosome]:
    """
    Tournament selection: sample k candidates, return the best.
    Repeated twice to get two independent parents.
    """
    def _pick() -> Chromosome:
        indices = rng.choice(len(feasible), size=min(k, len(feasible)), replace=False)
        return min((feasible[i] for i in indices),
                   key=lambda c: c.fitness if c.fitness is not None else math.inf)

    return _pick(), _pick()


def _check_stop(
    generation: int,
    stagnation_count: int,
    loop_start: float,
    config: GAConfig,
) -> tuple[bool, str]:
    """Return (should_stop, reason_string)."""
    if generation >= config.max_generations:
        return True, f"max_generations={config.max_generations}"

    if config.max_wall_seconds > 0:
        elapsed = time.monotonic() - loop_start
        if elapsed >= config.max_wall_seconds:
            return True, f"wall_time={elapsed:.1f}s >= {config.max_wall_seconds}s"

    if stagnation_count >= config.stagnation_limit:
        return True, f"stagnation={stagnation_count} generations"

    return False, ""
