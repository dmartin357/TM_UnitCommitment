"""PeriodCandidate and ForwardSolution dataclasses for GA v2."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PeriodCandidate:
    """Commitment + dispatch decision for one time period."""

    period: int
    committed_names: list[str]            # thermal units that are ON
    dispatch_thermal: dict[str, float]    # {name: MW} for committed thermals
    dispatch_renewable: dict[str, float]  # {name: MW} for variable renewables
    thermal_cost: float                   # ED production cost (thermals only)
    startup_cost: float                   # sum of startup costs for new ON units
    reg_up: float                         # sum(ramp_ub - dispatch) committed thermals
    reg_down: float                       # sum(dispatch - ramp_lb) committed thermals
    n_committed: int
    n_startups: int
    n_shutdowns: int

    @property
    def total_cost(self) -> float:
        return self.thermal_cost + self.startup_cost

    @property
    def reg_range(self) -> float:
        return self.reg_up + self.reg_down


@dataclass
class ForwardSolution:
    """A complete T-period UC solution produced by one GA v2 forward pass."""

    decisions: list[PeriodCandidate]   # one per period 1 .. n_periods-1

    @property
    def periods_solved(self) -> int:
        return len(self.decisions)

    def is_complete(self, n_periods: int) -> bool:
        """True only if every period 1..n_periods-1 has a solved decision."""
        return len(self.decisions) == n_periods - 1

    @property
    def total_cost(self) -> float:
        return sum(d.total_cost for d in self.decisions)

    @property
    def total_thermal_cost(self) -> float:
        return sum(d.thermal_cost for d in self.decisions)

    @property
    def total_startup_cost(self) -> float:
        return sum(d.startup_cost for d in self.decisions)

    def print_summary(self) -> None:
        n = len(self.decisions)
        w = 110
        print(f"\n{'=' * w}")
        print(f"  GA v2 Forward Solution Summary  ({n} periods)")
        print(f"{'=' * w}")
        print(
            f"  {'t':>4}  {'Commit':>7}  {'Start':>6}  {'Shut':>5}  "
            f"{'ED Cost ($)':>14}  {'SU Cost ($)':>12}  "
            f"{'Reg Up (MW)':>12}  {'Reg Dn (MW)':>12}"
        )
        print(
            f"  {'-'*4}  {'-'*7}  {'-'*6}  {'-'*5}  "
            f"{'-'*14}  {'-'*12}  {'-'*12}  {'-'*12}"
        )
        for d in self.decisions:
            print(
                f"  {d.period:>4}  {d.n_committed:>7}  {d.n_startups:>6}  "
                f"{d.n_shutdowns:>5}  "
                f"{d.thermal_cost:>14,.2f}  {d.startup_cost:>12,.2f}  "
                f"{d.reg_up:>12,.1f}  {d.reg_down:>12,.1f}"
            )
        print(f"{'=' * w}")
        print(f"  Total thermal cost : ${self.total_thermal_cost:>18,.2f}")
        print(f"  Total startup cost : ${self.total_startup_cost:>18,.2f}")
        print(f"  Total cost         : ${self.total_cost:>18,.2f}")
        print(f"{'=' * w}\n", flush=True)
