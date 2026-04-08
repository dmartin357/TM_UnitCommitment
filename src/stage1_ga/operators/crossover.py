"""
Crossover operator registry (stub / baseline).

Single-point crossover
----------------------
Split both parent chromosomes at a uniformly random point and exchange the
tails:
    child_a = parent_a[:k] + parent_b[k:]
    child_b = parent_b[:k] + parent_a[k:]

Must-run generators are protected: bits at must-run positions are forced to 1
after crossover to preserve the invariant that must-run generators are always
committed.

To add a new crossover operator:
  1. Create the operator as a function with signature:
       fn(parent_a, parent_b, must_run_mask, rng) -> tuple[Chromosome, Chromosome]
  2. Import it here and add it to REGISTRY with a string key.
  3. Pass that key as GAConfig.crossover_operator.
"""

from __future__ import annotations

import numpy as np

from ..chromosome import Chromosome


def single_point(
    parent_a: Chromosome,
    parent_b: Chromosome,
    must_run_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[Chromosome, Chromosome]:
    """
    Single-point crossover.

    Parameters
    ----------
    parent_a, parent_b : parent chromosomes (must share the same gen_names).
    must_run_mask      : boolean array, True at must-run generator indices.
    rng                : NumPy random Generator.

    Returns
    -------
    Two child Chromosomes (fitness=None, not yet evaluated).
    """
    n = len(parent_a.bits)
    # Crossover point in (0, n) so each child gets at least one bit from each parent
    k = int(rng.integers(low=1, high=n))

    bits_a = np.concatenate([parent_a.bits[:k], parent_b.bits[k:]])
    bits_b = np.concatenate([parent_b.bits[:k], parent_a.bits[k:]])

    # Enforce must-run
    bits_a[must_run_mask] = 1
    bits_b[must_run_mask] = 1

    gen_names = parent_a.gen_names
    return (
        Chromosome(gen_names=gen_names, bits=bits_a),
        Chromosome(gen_names=gen_names, bits=bits_b),
    )
