"""
Target-seeking mutation for Stage 2.

After repair, compare the working chromosome's n_committed to the pre-solve
target for this period.  Randomly flip eligible units to close the gap:

  Need more startups (target > current):
    Eligible to flip 0→1: units currently OFF in working copy that were also
    OFF in t and have satisfied time_down_minimum (allowed to start up).

  Need more shutdowns (target < current):
    Eligible to flip 1→0: units currently ON in working copy that were also
    ON in t and have satisfied time_up_minimum (allowed to shut down).

Selection within the eligible pool is uniform.  Stops when the target is
reached or the eligible pool is exhausted — whichever comes first.

Returns a new bit array (never modifies the input).
"""

from __future__ import annotations

import numpy as np

from .unit_state import FleetState


def mutate_toward_target(
    bits: np.ndarray,
    gen_names: list[str],
    fleet_state: FleetState,
    generators: dict,
    target_n_committed: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return a copy of bits mutated toward target_n_committed.

    Parameters
    ----------
    bits               : repaired commitment vector for t+1 (not modified).
    gen_names          : generator names in bit-vector order.
    fleet_state        : UnitState for each generator at end of period t.
    generators         : full generator dict from instance JSON.
    target_n_committed : desired number of committed generators.
    rng                : NumPy random Generator.

    Returns
    -------
    Mutated bit vector (np.ndarray, dtype=uint8).
    """
    bits = bits.copy()
    current_n = int(bits.sum())
    delta = target_n_committed - current_n

    if delta == 0:
        return bits

    if delta > 0:
        # Need more units on — find units eligible for startup
        eligible = [
            i for i, name in enumerate(gen_names)
            if bits[i] == 0
            and not fleet_state[name].committed          # was OFF in t
            and fleet_state[name].time_in_state >= int(
                generators[name].get("time_down_minimum", 0)
            )
        ]
        if eligible:
            chosen = rng.choice(eligible, size=min(delta, len(eligible)), replace=False)
            for i in chosen:
                bits[i] = 1

    else:
        # Need fewer units on — find units eligible for shutdown
        eligible = [
            i for i, name in enumerate(gen_names)
            if bits[i] == 1
            and fleet_state[name].committed              # was ON in t
            and fleet_state[name].time_in_state >= int(
                generators[name].get("time_up_minimum", 0)
            )
        ]
        if eligible:
            chosen = rng.choice(eligible, size=min(-delta, len(eligible)), replace=False)
            for i in chosen:
                bits[i] = 0

    return bits
