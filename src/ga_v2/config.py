"""GA v2 configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GAv2Config:
    # Population: one full T-period solution per t=1 candidate
    n_population: int = 20

    # Candidates generated at each period t > 1 within a forward pass
    n_candidates_per_period: int = 10

    # Weighted-rank selection weights (must sum to 1.0)
    economics_weight: float = 0.5
    regulation_weight: float = 0.5

    # LP solver for ED
    solver: str = "auto"

    # Marginal cost assigned to wind/solar in the ED ($0.01/MWh keeps them
    # at their upper bound without making them free/degenerate)
    renewable_cost_per_mwh: float = 0.01

    rng_seed: int = 42
