"""
UnitState — per-generator state carried forward through the Stage 2 pass.

For each generator we track:
  committed     : whether the unit is currently on (True) or off (False)
  dispatch      : MW output in the current period (0.0 if offline)
  time_in_state : consecutive periods the unit has been in its current state

The initial state at t=0 is read directly from the instance JSON fields:
  unit_on_t0    → committed
  power_output_t0 → dispatch
  time_up_t0    → time_in_state when committed=True
  time_down_t0  → time_in_state when committed=False
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UnitState:
    committed: bool
    dispatch: float       # MW; 0.0 when offline
    time_in_state: int    # consecutive periods in current committed/offline state

    @classmethod
    def from_t0(cls, gen_data: dict) -> "UnitState":
        committed = bool(gen_data.get("unit_on_t0", 0))
        time_in_state = int(
            gen_data.get("time_up_t0", 0) if committed
            else gen_data.get("time_down_t0", 0)
        )
        return cls(
            committed=committed,
            dispatch=float(gen_data.get("power_output_t0", 0.0)),
            time_in_state=time_in_state,
        )

    def advance(self, committed: bool, dispatch: float) -> "UnitState":
        """Return the state for the next period given this period's decision."""
        if committed == self.committed:
            new_time = self.time_in_state + 1
        else:
            new_time = 1
        return UnitState(committed=committed, dispatch=dispatch, time_in_state=new_time)


# Alias for the full per-period fleet state
FleetState = dict[str, UnitState]   # {gen_name: UnitState}


def fleet_state_from_t0(generators: dict) -> FleetState:
    """Build the initial FleetState from instance t0 fields."""
    return {name: UnitState.from_t0(gen_data) for name, gen_data in generators.items()}


def advance_fleet_state(
    fleet_state: FleetState,
    committed_names: set[str],
    dispatch: dict[str, float],
) -> FleetState:
    """
    Produce the next period's FleetState given the winning commitment + dispatch.

    Units not in committed_names are treated as offline (dispatch = 0.0).
    """
    return {
        name: state.advance(
            committed=name in committed_names,
            dispatch=dispatch.get(name, 0.0),
        )
        for name, state in fleet_state.items()
    }
