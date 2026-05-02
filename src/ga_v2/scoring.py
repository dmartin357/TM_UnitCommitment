"""
Weighted-rank candidate selection for GA v2.

Candidates are ranked independently on two criteria:
  1. total_cost  (lower is better → rank 1 = cheapest)
  2. reg_range   (higher is better → rank 1 = most flexible)

The composite score for candidate i is:
  score_i = economics_weight * cost_rank_i + regulation_weight * reg_rank_i

The candidate with the lowest composite score wins.  Ties in either ranking
are broken by original list order (stable sort).
"""

from __future__ import annotations

from .candidate import PeriodCandidate


def select_winner(
    candidates: list[PeriodCandidate],
    economics_weight: float = 0.5,
    regulation_weight: float = 0.5,
) -> PeriodCandidate:
    """
    Return the candidate with the best weighted-rank composite score.

    Parameters
    ----------
    candidates         : non-empty list of feasible PeriodCandidates.
    economics_weight   : weight applied to cost rank.
    regulation_weight  : weight applied to regulation-range rank.
    """
    if len(candidates) == 1:
        return candidates[0]

    n = len(candidates)

    # Cost rank: ascending (lower cost → lower rank number = better)
    cost_order = sorted(range(n), key=lambda i: candidates[i].total_cost)
    cost_rank = [0] * n
    for rank, idx in enumerate(cost_order):
        cost_rank[idx] = rank + 1

    # Regulation rank: descending (higher reg range → lower rank number = better)
    reg_order = sorted(range(n), key=lambda i: -candidates[i].reg_range)
    reg_rank = [0] * n
    for rank, idx in enumerate(reg_order):
        reg_rank[idx] = rank + 1

    scores = [
        economics_weight * cost_rank[i] + regulation_weight * reg_rank[i]
        for i in range(n)
    ]
    best = min(range(n), key=lambda i: scores[i])
    return candidates[best]
