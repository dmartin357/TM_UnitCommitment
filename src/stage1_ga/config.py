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
    # Generators are ranked by the chosen sort_attribute and a discrete
    # probability distribution over cut positions is derived from it.
    #
    # sort_attribute / sort_ascending control both the ranking and the
    # probability weights.  The baseline uses Pmax descending.
    sort_attribute: str = "power_output_maximum"
    sort_ascending: bool = False   # False → descending (largest Pmax first)
    # 'uniform' → equal probability for every position (random cuts, no bias).
    # 'cdf'     → normalized cumulative attribute values; biases cuts toward
    #             smaller generators, preserving large committed units.
    # 'pdf'     → weight proportional to individual attribute value.
    location_dist_type: str = "uniform"

    # ── Cut group stopping criterion ──────────────────────────────────────────
    # Generators are cut one at a time until the proposed cut would reduce the
    # committed fleet's aggregate ramp capability below the regulation
    # requirements derived from renewable forecast uncertainty:
    #
    #   Keep cutting while sum(ramp_up_limit,   committed) >= reg_up_req
    #   Keep cutting while sum(ramp_down_limit, committed) >= reg_down_req
    #
    # reg_up_req and reg_down_req are computed per time period from the
    # renewable forecast band and passed directly to run_stage1_ga() —
    # they are not stored in GAConfig.

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
    # Stop early when the best feasible chromosome's n_committed is within this
    # many units of the pre-solve target.  Only active when a per-period
    # target_n_committed is passed to run_stage1_ga().  0 = exact match required.
    target_n_committed_tolerance: int = 2

    # ── ED solver ────────────────────────────────────────────────────────────
    # 'auto' → try Gurobi, then CPLEX, then CBC.
    # Override with 'gurobi', 'cplex', or 'cbc' to force a specific solver.
    solver: str = "auto"
