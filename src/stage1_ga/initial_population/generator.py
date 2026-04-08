"""
InitialPopulationGenerator — produces seed chromosomes via iterative CDF cutting.

Algorithm (per chromosome)
--------------------------
1. Start with all generators committed (bits all 1).  Must-run generators are
   permanently committed and excluded from the cut pool.

2. Repeat:
   a. Sort the currently committed may-run generators by Pmax descending.
      Compute two CDFs in that shared rank order:
        Pmax CDF[k] = cumsum(pmax[1..k])   ← used as the location probability dist
        Pmin CDF[k] = cumsum(pmin[1..k])   ← same rank order, not sorted by Pmin
      Normalize the Pmax CDF to get a discrete probability distribution over
      cut positions.  Higher-ranked (lower-Pmax) positions get higher probability,
      so the algorithm preferentially cuts smaller generators first.

   b. Sample one generator from the Pmax CDF distribution (the "cut candidate").

   c. Project the highest CDF values AFTER removing that generator:
        new_pmax_total = last value of new Pmax CDF = sum(pmax, remaining)
        new_pmin_total = last value of new Pmin CDF = sum(pmin, remaining)
      Check:
        upper margin:  new_pmax_total >= demand * (1 + tolerance)
        lower margin:  new_pmin_total >= demand * (1 - tolerance)

   d. If BOTH margins still hold → accept the cut and continue.
      If EITHER margin would be violated → undo (don't apply the cut) and stop.

3. The resulting bit vector is the chromosome.

Rationale for the stopping criteria
-------------------------------------
  upper margin (capacity buffer):
    Pmax CDF total >= demand * (1 + tol).  Prevents cutting so many generators
    that the fleet can't cover demand plus an upside uncertainty buffer.

  lower margin (min-generation floor):
    Pmin CDF total >= demand * (1 - tol).  Both totals decrease monotonically as
    cuts are accepted; the lower margin fires when the committed fleet's minimum
    generation has been reduced far enough.  Stopping here ensures the ED still
    has flexibility: sum(pmin) is close to but above demand*(1-tol), well below
    demand itself.

The Pmax CDF (and therefore the probability distribution) is recomputed after
every accepted cut because removing a generator changes the relative weights of
the remaining pool.
"""

from __future__ import annotations

import numpy as np

from ..chromosome import Chromosome
from .cdf_utils import build_static_ordering, compute_location_probs


class InitialPopulationGenerator:
    """
    Pre-computes the stable generator ordering once, then cheaply generates
    individual chromosomes on demand via the iterative cut algorithm.

    Parameters
    ----------
    generators         : {name: gen_data} from instance JSON.
    sort_attribute     : attribute to sort generators by (and weight the CDF).
    sort_ascending     : True → smallest first in the ranked list.
    location_dist_type : 'pdf' or 'cdf' — how attribute values become probabilities.
    demand_tolerance   : fraction of demand used as the feasibility margin (e.g., 0.20).
    """

    def __init__(
        self,
        generators: dict,
        sort_attribute: str,
        sort_ascending: bool,
        location_dist_type: str,
        demand_tolerance: float,
    ) -> None:
        self._generators = generators
        self._sort_attribute = sort_attribute
        self._sort_ascending = sort_ascending
        self._location_dist_type = location_dist_type
        self._demand_tolerance = demand_tolerance

        # Stable bit-vector ordering: must-run first, then may-run
        self._sorted_names, self._must_run_sorted, self._may_run_sorted = (
            build_static_ordering(generators, sort_attribute, sort_ascending)
        )

        # Pre-cache pmax and pmin for every generator (avoids repeated dict lookups)
        self._pmax: dict[str, float] = {
            n: float(g.get("power_output_maximum", 0.0))
            for n, g in generators.items()
        }
        self._pmin: dict[str, float] = {
            n: float(g.get("power_output_minimum", 0.0))
            for n, g in generators.items()
        }

        # Must-run capacity sums are fixed across all chromosomes
        self._must_run_pmax: float = sum(self._pmax[n] for n in self._must_run_sorted)
        self._must_run_pmin: float = sum(self._pmin[n] for n in self._must_run_sorted)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, rng: np.random.Generator, demand: float) -> Chromosome:
        """
        Sample one chromosome for the given demand level.

        The cut group is built iteratively: one generator is cut per step,
        the CDF is recomputed, and the step is accepted only if both
        capacity margins remain satisfied.
        """
        # Track currently committed may-run generators as a list (for CDF
        # computation) and a set (for O(1) membership and removal).
        committed_list = list(self._may_run_sorted)
        committed_set  = set(self._may_run_sorted)

        current_pmax = self._must_run_pmax + sum(self._pmax[n] for n in committed_list)
        current_pmin = self._must_run_pmin + sum(self._pmin[n] for n in committed_list)

        upper_limit = demand * (1.0 + self._demand_tolerance)
        lower_limit = demand * (1.0 - self._demand_tolerance)

        while committed_list:
            # Recompute CDF/PDF for currently committed may-run generators
            sorted_subset, probs = compute_location_probs(
                committed_list,
                self._generators,
                self._sort_attribute,
                self._sort_ascending,
                self._location_dist_type,
            )

            # Sample one candidate to cut
            cut_idx = int(rng.choice(len(sorted_subset), p=probs))
            cut_gen = sorted_subset[cut_idx]

            # Project capacity after this cut
            new_pmax = current_pmax - self._pmax[cut_gen]
            new_pmin = current_pmin - self._pmin[cut_gen]

            # Reject the cut if either margin would be violated:
            #   new_pmax < upper_limit → too little capacity remaining
            #   new_pmin < lower_limit → min-generation floor dropped too low
            # Both pmax and pmin decrease monotonically as cuts are accepted.
            # We cut TOWARD lower_limit from above; stop when we'd overshoot it.
            if new_pmax < upper_limit or new_pmin < lower_limit:
                break

            # Accept the cut
            committed_set.discard(cut_gen)
            committed_list = [n for n in committed_list if n != cut_gen]
            current_pmax = new_pmax
            current_pmin = new_pmin

        # Build the bit vector using the stable sorted_names ordering
        bits = np.array(
            [
                1 if (n in self._must_run_sorted or n in committed_set) else 0
                for n in self._sorted_names
            ],
            dtype=np.uint8,
        )
        return Chromosome(gen_names=self._sorted_names, bits=bits)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def sorted_names(self) -> list[str]:
        """Generator names in the bit-vector order shared by all chromosomes."""
        return self._sorted_names

    @property
    def n_generators(self) -> int:
        return len(self._sorted_names)

    @property
    def n_must_run(self) -> int:
        return len(self._must_run_sorted)

    @property
    def n_may_run(self) -> int:
        return len(self._may_run_sorted)
