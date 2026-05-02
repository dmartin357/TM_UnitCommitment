"""
Classify thermal units into constrained_ON, constrained_OFF, and free
for the next time period, given the current fleet state.

Rules (same logic as Stage 2 repair):
  constrained_ON  : must_run=1, OR (was ON and min_up not satisfied),
                    OR (was ON and dispatch > ramp_shutdown_limit — unit
                    cannot drop to zero in one period)
  constrained_OFF : was OFF and min_down not satisfied
  free            : everything else (eligible for cutting pool)
"""

from __future__ import annotations

from ..stage2_ga.unit_state import FleetState


def classify_thermal_units(
    fleet_state: FleetState,
    generators: dict,
) -> tuple[set[str], set[str], set[str]]:
    """
    Return (constrained_ON, constrained_OFF, free) for the next period.

    Parameters
    ----------
    fleet_state : current UnitState for every thermal generator.
    generators  : full thermal generator dict from instance JSON.
    """
    constrained_on: set[str] = set()
    constrained_off: set[str] = set()
    free: set[str] = set()

    for name, gen in generators.items():
        state   = fleet_state[name]
        min_up  = int(gen.get("time_up_minimum",   0))
        min_dn  = int(gen.get("time_down_minimum", 0))
        must_run = int(gen.get("must_run", 0))

        if must_run:
            constrained_on.add(name)
        elif state.committed:
            if state.time_in_state < min_up:
                constrained_on.add(name)
            else:
                sd_ramp = float(gen.get("ramp_shutdown_limit", float("inf")))
                if state.dispatch > sd_ramp + 1e-6:
                    # Dispatch is too high to shut down in one period
                    constrained_on.add(name)
                else:
                    free.add(name)
        else:
            if state.time_in_state < min_dn:
                constrained_off.add(name)
            else:
                free.add(name)

    return constrained_on, constrained_off, free
