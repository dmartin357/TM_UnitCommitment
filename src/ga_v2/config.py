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

    # Fraction of renewable pmax expected to be dispatched when computing the
    # thermal demand target for the cutting feasibility check.
    # 1.0 = assume full forecast (matches CBC dispatch behaviour for zero-cost
    # wind/solar); 0.5 = old midpoint heuristic; lower values are more
    # conservative and result in committing more thermal units.
    renewable_fraction: float = 1.0

    # Minimum regulation-up reserve requirement expressed as a fraction of
    # total period demand.  Acts as a floor on the thermal demand target so
    # that the cutting algorithm never commits so few thermal units that the
    # fleet cannot provide the required reg-up, even when renewables alone
    # could theoretically meet all demand.
    # Typical ISO value: 0.05 (5 % of demand).
    reg_up_req_fraction: float = 0.05

    rng_seed: int = 42
