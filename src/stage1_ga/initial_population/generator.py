"""
InitialPopulationGenerator — produces seed chromosomes via iterative CDF cutting.

Algorithm (per chromosome)
--------------------------
1. Start with all generators committed (bits all 1).  Must-run generators are
   permanently committed and excluded from the cut pool.

2. Repeat:
   a. Sort the currently committed may-run generators by the chosen attribute
      and compute a discrete probability distribution over cut positions.

   b. Sample one generator from that distribution (the "cut candidate").

   c. Project the aggregate ramp capability AFTER removing that generator:
        new_ramp_up   = sum(ramp_up_limit,   remaining committed)
        new_ramp_down = sum(ramp_down_limit,  remaining committed)
      Check:
        new_ramp_up   >= reg_up_req    (fleet can still cover a renewable drop)
        new_ramp_down >= reg_down_req  (fleet can still absorb a renewable surge)

   d. If BOTH checks pass → accept the cut and continue.
      If EITHER would be violated → undo (don't apply the cut) and stop.

3. The resulting bit vector is the chromosome.

Rationale for the four stopping checks
----------------------------------------
  Capacity upper (new_pmax >= thermal_max_demand):
    Ensures the fleet can still meet demand in the worst case — renewables at
    their minimum, so thermal must cover the full gap.
    thermal_max_demand = demand + reg_up_req = total_demand − renewable_min.

  Reg-up check (new_ramp_up >= reg_up_req):
    Ensures the fleet retains enough upward ramp capacity to compensate if
    aggregate renewable output falls to its minimum.

  Reg-down check (new_ramp_down >= reg_down_req):
    Ensures the fleet can reduce output fast enough if renewables surge to their
    maximum, preventing over-generation.

The location-probability distribution is recomputed after every accepted cut
because removing a generator changes the relative weights of the remaining pool.
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
    location_dist_type : 'uniform', 'pdf', or 'cdf'.
    demand             : expected thermal demand (MW) for this period.
    reg_up_req         : MW of upward ramp the committed fleet must retain
                         (= renewable_expected − renewable_min).
    reg_down_req       : MW of downward ramp the committed fleet must retain
                         (= renewable_max − renewable_expected).
    """

    def __init__(
        self,
        generators: dict,
        sort_attribute: str,
        sort_ascending: bool,
        location_dist_type: str,
        demand: float = 0.0,
        reg_up_req: float = 0.0,
        reg_down_req: float = 0.0,
    ) -> None:
        self._generators = generators
        self._sort_attribute = sort_attribute
        self._sort_ascending = sort_ascending
        self._location_dist_type = location_dist_type
        self._reg_up_req = reg_up_req
        self._reg_down_req = reg_down_req

        # Worst-case thermal demand = demand when renewables are at their minimum
        #   thermal_max = demand + reg_up_req  = total_demand − renewable_min
        self._thermal_max_demand: float = demand + reg_up_req

        # Stable bit-vector ordering: must-run first, then may-run
        self._sorted_names, self._must_run_sorted, self._may_run_sorted = (
            build_static_ordering(generators, sort_attribute, sort_ascending)
        )

        # Pre-cache pmax and effective regulation potentials (avoids repeated dict lookups).
        # reg_up_potential   = min(ramp_up_limit,   pmax - pmin)  — actual achievable reg-up at pmin dispatch
        # reg_down_potential = min(ramp_down_limit, pmax - pmin)  — actual achievable reg-down at pmax dispatch
        # Using raw ramp limits overestimates headroom-constrained generators (ramp > pmax-pmin),
        # causing the cutting criterion to pass when the actual reg capacity would fail.
        self._pmax: dict[str, float] = {
            n: float(g.get("power_output_maximum", 0.0))
            for n, g in generators.items()
        }
        self._reg_up_potential: dict[str, float] = {
            n: min(
                float(g.get("ramp_up_limit", 0.0)),
                float(g.get("power_output_maximum", 0.0)) - float(g.get("power_output_minimum", 0.0)),
            )
            for n, g in generators.items()
        }
        self._reg_down_potential: dict[str, float] = {
            n: min(
                float(g.get("ramp_down_limit", 0.0)),
                float(g.get("power_output_maximum", 0.0)) - float(g.get("power_output_minimum", 0.0)),
            )
            for n, g in generators.items()
        }

        # Must-run sums are fixed across all chromosomes
        self._must_run_pmax:           float = sum(self._pmax[n]              for n in self._must_run_sorted)
        self._must_run_ramp_up:        float = sum(self._reg_up_potential[n]  for n in self._must_run_sorted)
        self._must_run_ramp_down:      float = sum(self._reg_down_potential[n] for n in self._must_run_sorted)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, rng: np.random.Generator) -> Chromosome:
        """
        Sample one chromosome.

        Generators are cut one at a time.  After each proposed cut the
        aggregate ramp-up and ramp-down capability of the remaining committed
        fleet is checked against the regulation requirements.  The first cut
        that would violate either requirement is rejected; the chromosome is
        built from all prior accepted cuts.
        """
        committed_list = list(self._may_run_sorted)
        committed_set  = set(self._may_run_sorted)

        current_pmax      = self._must_run_pmax      + sum(self._pmax[n]               for n in committed_list)
        current_ramp_up   = self._must_run_ramp_up   + sum(self._reg_up_potential[n]   for n in committed_list)
        current_ramp_down = self._must_run_ramp_down + sum(self._reg_down_potential[n] for n in committed_list)

        while committed_list:
            sorted_subset, probs = compute_location_probs(
                committed_list,
                self._generators,
                self._sort_attribute,
                self._sort_ascending,
                self._location_dist_type,
            )

            cut_idx = int(rng.choice(len(sorted_subset), p=probs))
            cut_gen = sorted_subset[cut_idx]

            new_pmax      = current_pmax      - self._pmax[cut_gen]
            new_ramp_up   = current_ramp_up   - self._reg_up_potential[cut_gen]
            new_ramp_down = current_ramp_down - self._reg_down_potential[cut_gen]

            # Reject cut if any of the three feasibility checks would be violated
            if (
                new_pmax      < self._thermal_max_demand  # can't cover worst-case demand
                or new_ramp_up   < self._reg_up_req       # insufficient reg-up reserve
                or new_ramp_down < self._reg_down_req     # insufficient reg-down reserve
            ):
                break

            committed_set.discard(cut_gen)
            committed_list    = [n for n in committed_list if n != cut_gen]
            current_pmax      = new_pmax
            current_ramp_up   = new_ramp_up
            current_ramp_down = new_ramp_down


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
