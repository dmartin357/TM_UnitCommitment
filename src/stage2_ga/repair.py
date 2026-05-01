"""
Min up/down time repair for Stage 2.

Given a candidate chromosome (bit vector) for period t+1 and the fleet state
from period t, force any commitment violations:

  - Unit was ON in t and has not yet satisfied time_up_minimum
      → must remain ON in t+1 (cannot shut down yet)

  - Unit was OFF in t and has not yet satisfied time_down_minimum
      → must remain OFF in t+1 (cannot start up yet)

Returns a new bit array (never modifies the input).
"""

from __future__ import annotations

import numpy as np

from .unit_state import FleetState


def repair_min_updown(
    bits: np.ndarray,
    gen_names: list[str],
    fleet_state: FleetState,
    generators: dict,
) -> np.ndarray:
    """
    Return a copy of bits with min up/down time violations corrected.

    Parameters
    ----------
    bits        : binary commitment vector for period t+1 (not modified).
    gen_names   : generator names in bit-vector order.
    fleet_state : UnitState for each generator at end of period t.
    generators  : full generator dict from instance JSON.

    Returns
    -------
    Corrected bit vector (np.ndarray, dtype=uint8).
    """
    bits = bits.copy()
    for i, name in enumerate(gen_names):
        state = fleet_state[name]
        gen   = generators[name]
        min_up = int(gen.get("time_up_minimum",   0))
        min_dn = int(gen.get("time_down_minimum", 0))

        if state.committed and state.time_in_state < min_up:
            bits[i] = 1   # must stay on — min up time not yet met
        elif not state.committed and state.time_in_state < min_dn:
            bits[i] = 0   # must stay off — min down time not yet met

    return bits
