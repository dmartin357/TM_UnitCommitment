"""
GAConfig — all tunable parameters for the Stage 1 genetic algorithm.

Each parameter has a sensible default; override on a per-instance basis via
dataclass field replacement or by passing keyword arguments to the constructor.
"""

from dataclasses import dataclass, field


@dataclass
class GAConfig:
    # ── Population ────────────────────────────────────────────────────────────
    population_size: int = 200
    # Number of chromosomes generated via CDF/PDF sampling before GA loop begins.
    # Remaining slots are filled by the GA loop itself.
    initial_sample_size: int = 100

    # ── CDF initial generation ────────────────────────────────────────────────
    # Generators are ranked by Pmax descending.  Two CDFs are computed over
    # the remaining committed generators in that shared rank order:
    #
    #   Pmax CDF[k] = sum(pmax[1..k])   — used as the location probability dist
    #   Pmin CDF[k] = sum(pmin[1..k])   — same rank order (not sorted by Pmin)
    #
    # The "highest" (last) CDF value of each = sum(pmax/pmin, all remaining),
    # which is what the stopping criteria checks against demand * (1 ± tol).
    #
    # sort_attribute / sort_ascending are kept as parameters for future
    # experimentation with alternative ranking dimensions, but the baseline
    # algorithm always uses Pmax descending.
    sort_attribute: str = "power_output_maximum"
    sort_ascending: bool = False   # False → descending (largest Pmax first)
    # 'uniform' → equal probability for every position (random cuts, no bias).
    # 'cdf'     → normalized cumulative Pmax values; biases cuts toward smaller
    #             generators, preserving large committed units.
    # 'pdf'     → weight proportional to individual Pmax value.
    location_dist_type: str = "uniform"

    # ── Cut group stopping criteria ───────────────────────────────────────────
    # Generators are cut one at a time (iteratively) until one of these margins
    # would be violated.  For a demand D and tolerance t:
    #   • Keep cutting while sum(pmax, committed) >= D * (1 + t)   [capacity buffer]
    #   • Keep cutting while sum(pmin, committed) <= D * (1 - t)   [over-commitment buffer]
    # The first cut that would breach either bound is undone; all prior cuts
    # form the cut group for that chromosome.
    demand_tolerance: float = 0.20  # 0.20 → ±20% margins around demand

    # ── Renewable uncertainty ─────────────────────────────────────────────────
    # Float in [0, 1].  0 = no uncertainty, 1 = maximum uncertainty.
    # Reserved for future use (e.g., tightening demand_tolerance or biasing
    # the location distribution under high renewable output variability).
    renewable_uncertainty: float = 0.0

    # ── GA operators ─────────────────────────────────────────────────────────
    # Name must match a key in operators.CROSSOVER_REGISTRY.
    crossover_operator: str = "single_point"
    # Probability of flipping each bit during mutation.
    mutation_rate: float = 0.01

    # ── Stopping criteria (all are enforced; whichever triggers first wins) ──
    # Maximum number of GA generations.
    max_generations: int = 100
    # Wall-clock time limit in seconds (0 = disabled).
    max_wall_seconds: float = 300.0
    # Stop if best fitness has not improved for this many consecutive generations.
    stagnation_limit: int = 20

    # ── ED solver ────────────────────────────────────────────────────────────
    # 'auto' → try Gurobi, then CPLEX, then CBC.
    # Override with 'gurobi', 'cplex', or 'cbc' to force a specific solver.
    solver: str = "auto"
