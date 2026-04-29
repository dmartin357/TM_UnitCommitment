"""
Chromosome representation for Stage 1.

A chromosome encodes the commitment decision for all generators in a single
time period as a binary NumPy array:
    bits[i] = 1  →  generator i is committed (online)
    bits[i] = 0  →  generator i is offline

Must-run generators are always set to 1 and are never modified by the GA.

The SHA-256 hash of the bit vector is used for O(1) duplicate detection
across the population and the seen-chromosome cache.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Chromosome:
    # Ordered list of generator names (defines the index mapping).
    gen_names: list[str]
    # Binary commitment vector; dtype=np.uint8, shape=(n_generators,).
    bits: np.ndarray
    # SHA-256 hex digest of the bit vector (set on construction).
    hash: str = field(init=False)
    # Total production cost from the ED subproblem; None until evaluated.
    fitness: float | None = None
    # Per-generator dispatch levels (MW); None until evaluated.
    dispatch: dict[str, float] | None = None
    # Regulation capability (MW); computed alongside dispatch, None until evaluated.
    # reg_up  = sum over committed units of min(ramp_up_limit,  pmax - dispatch)
    # reg_down = sum over committed units of min(ramp_down_limit, dispatch - pmin)
    reg_up:   float | None = None
    reg_down: float | None = None
    # Renewable energy lost to thermal minimum-generation constraints (MW).
    # > 0 when sum(pmin, committed) > expected thermal demand; the ED demand is
    # augmented to sum(pmin) in that case, forcing curtailment of renewable output.
    renewable_loss: float | None = None

    def __post_init__(self) -> None:
        self.hash = _hash_bits(self.bits)

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def committed(self) -> list[str]:
        """Names of committed (online) generators."""
        return [name for name, b in zip(self.gen_names, self.bits) if b == 1]

    @property
    def n_committed(self) -> int:
        return int(self.bits.sum())

    def is_feasible(self) -> bool:
        """True if an ED solution was found (fitness is finite)."""
        return self.fitness is not None and np.isfinite(self.fitness)

    # ── Factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def all_on(cls, gen_names: list[str]) -> "Chromosome":
        """All generators committed."""
        return cls(gen_names=gen_names, bits=np.ones(len(gen_names), dtype=np.uint8))

    @classmethod
    def from_bits(cls, gen_names: list[str], bits: np.ndarray) -> "Chromosome":
        return cls(gen_names=gen_names, bits=bits.astype(np.uint8).copy())

    # ── Mutation support ──────────────────────────────────────────────────────

    def copy(self) -> "Chromosome":
        """Return a new Chromosome with the same bits (fitness cleared)."""
        return Chromosome(gen_names=self.gen_names, bits=self.bits.copy())

    # ── Comparison (by fitness) ───────────────────────────────────────────────

    def __lt__(self, other: "Chromosome") -> bool:
        """Lower fitness (cost) is better."""
        if self.fitness is None:
            return False
        if other.fitness is None:
            return True
        return self.fitness < other.fitness

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chromosome):
            return NotImplemented
        return self.hash == other.hash

    def __hash__(self) -> int:
        return hash(self.hash)

    def __repr__(self) -> str:
        fit = f"{self.fitness:.2f}"         if self.fitness        is not None else "None"
        ru  = f"{self.reg_up:.1f}"          if self.reg_up         is not None else "None"
        rd  = f"{self.reg_down:.1f}"        if self.reg_down       is not None else "None"
        rl  = f"{self.renewable_loss:.1f}"  if self.renewable_loss is not None else "None"
        return (f"Chromosome(n={len(self.bits)}, committed={self.n_committed}, "
                f"fitness={fit}, reg_up={ru}, reg_down={rd}, renewable_loss={rl})")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _hash_bits(bits: np.ndarray) -> str:
    return hashlib.sha256(bits.tobytes()).hexdigest()
