"""
BoundedPopulation — fixed-capacity chromosome pool for one time period.

Invariants maintained at all times:
  - At most `max_size` chromosomes are stored.
  - No two chromosomes with the same SHA-256 hash coexist.
  - The internal list is kept sorted by fitness (ascending = best first).
    Chromosomes with fitness=None sort to the end.

When the population is full and a new chromosome is added:
  - If the new chromosome is better than the current worst, the worst is
    evicted and the new one inserted.
  - Otherwise the new chromosome is discarded.
"""

from __future__ import annotations

import bisect
from typing import Iterator

from .chromosome import Chromosome


class BoundedPopulation:
    def __init__(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self.max_size = max_size
        self._chroms: list[Chromosome] = []   # sorted best → worst
        self._seen: set[str] = set()           # hashes of all chromosomes ever added

    # ── Core API ──────────────────────────────────────────────────────────────

    def add(self, chrom: Chromosome) -> bool:
        """
        Attempt to add a chromosome.

        Returns True if the chromosome was accepted, False if it was rejected
        (duplicate hash, or worse than the current worst when full).
        """
        if chrom.hash in self._seen:
            return False

        self._seen.add(chrom.hash)

        if len(self._chroms) < self.max_size:
            _sorted_insert(self._chroms, chrom)
            return True

        # Full: only accept if better than current worst.
        worst = self._chroms[-1]
        if _is_better(chrom, worst):
            self._chroms.pop()
            _sorted_insert(self._chroms, chrom)
            return True

        return False

    def __len__(self) -> int:
        return len(self._chroms)

    def __iter__(self) -> Iterator[Chromosome]:
        return iter(self._chroms)

    def __getitem__(self, idx: int) -> Chromosome:
        return self._chroms[idx]

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def best(self) -> Chromosome | None:
        return self._chroms[0] if self._chroms else None

    @property
    def worst(self) -> Chromosome | None:
        return self._chroms[-1] if self._chroms else None

    @property
    def is_full(self) -> bool:
        return len(self._chroms) >= self.max_size

    def seen(self, chrom: Chromosome) -> bool:
        """True if this hash has already been evaluated (duplicate detection)."""
        return chrom.hash in self._seen

    def feasible(self) -> list[Chromosome]:
        """Return all chromosomes with a finite fitness value."""
        return [c for c in self._chroms if c.is_feasible()]

    def __repr__(self) -> str:
        best_fit = f"{self.best.fitness:.2f}" if self.best and self.best.fitness else "N/A"
        return (
            f"BoundedPopulation(size={len(self)}/{self.max_size}, "
            f"best_fitness={best_fit})"
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

class _FitnessKey:
    """Wrap a Chromosome so bisect can sort on fitness (None → +inf)."""

    __slots__ = ("chrom",)

    def __init__(self, chrom: Chromosome) -> None:
        self.chrom = chrom

    def _key(self) -> float:
        return self.chrom.fitness if self.chrom.fitness is not None else float("inf")

    def __lt__(self, other: "_FitnessKey") -> bool:
        return self._key() < other._key()

    def __le__(self, other: "_FitnessKey") -> bool:
        return self._key() <= other._key()

    def __gt__(self, other: "_FitnessKey") -> bool:
        return self._key() > other._key()

    def __ge__(self, other: "_FitnessKey") -> bool:
        return self._key() >= other._key()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _FitnessKey):
            return NotImplemented
        return self._key() == other._key()


def _sorted_insert(lst: list[Chromosome], chrom: Chromosome) -> None:
    """Insert chrom into lst maintaining ascending fitness order."""
    keys = [_FitnessKey(c) for c in lst]
    idx = bisect.bisect_left(keys, _FitnessKey(chrom))
    lst.insert(idx, chrom)


def _is_better(a: Chromosome, b: Chromosome) -> bool:
    """Return True if a has strictly lower cost than b."""
    fa = a.fitness if a.fitness is not None else float("inf")
    fb = b.fitness if b.fitness is not None else float("inf")
    return fa < fb
