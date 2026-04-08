"""
Bit-flip mutation.

Each bit in the chromosome is independently flipped with probability
`mutation_rate`.  Must-run generator bits are protected and never flipped.
"""

from __future__ import annotations

import numpy as np

from ..chromosome import Chromosome


def bit_flip(
    chrom: Chromosome,
    mutation_rate: float,
    must_run_mask: np.ndarray,
    rng: np.random.Generator,
) -> Chromosome:
    """
    Return a new Chromosome with bits randomly flipped (fitness cleared).

    Parameters
    ----------
    chrom          : chromosome to mutate.
    mutation_rate  : probability of flipping each bit.
    must_run_mask  : boolean array, True at must-run generator indices.
    rng            : NumPy random Generator.
    """
    flip_mask = rng.random(len(chrom.bits)) < mutation_rate
    # Do not flip must-run bits
    flip_mask &= ~must_run_mask

    new_bits = chrom.bits.copy()
    new_bits[flip_mask] ^= 1  # XOR flip: 0→1, 1→0

    return Chromosome(gen_names=chrom.gen_names, bits=new_bits)
