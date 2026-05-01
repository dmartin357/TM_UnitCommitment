"""Stage 2 GA — configuration."""

from dataclasses import dataclass


@dataclass
class Stage2Config:
    # LP solver for the per-transition ED (same options as Stage 1)
    solver: str = "auto"

    # Percentile of the pre-solve n_committed distribution used as the
    # startup/shutdown target for each period.
    target_percentile: float = 50.0

    # Selection distribution for Stage 1 candidates.
    # 'uniform' — each feasible chromosome equally likely.
    # More options (cost-weighted, rank-weighted) can be added later.
    selection_mode: str = "uniform"

    # Random seed for candidate selection and mutation.
    rng_seed: int = 42
