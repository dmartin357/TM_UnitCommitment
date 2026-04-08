"""
Uniform cut-size distribution (stub / baseline).

Samples the cut group size uniformly from [1, n_cuttable // 2].

Higher renewable uncertainty shifts the upper bound downward so that,
on average, fewer generators are cut (more units remain online to provide
implicit reserves).

    max_cut = max(1, int(n_cuttable * 0.5 * (1 - renewable_uncertainty)))

This is intentionally simple.  Replace or supplement with a more
sophisticated distribution (e.g., Beta, truncated Normal, or demand-relative)
by adding a new module to this package and registering it in __init__.py.
"""

from __future__ import annotations

import numpy as np


def sample(
    n_cuttable: int,
    renewable_uncertainty: float,
    rng: np.random.Generator,
) -> int:
    """
    Return a sampled cut-group size.

    Parameters
    ----------
    n_cuttable            : number of may-run (cuttable) generators.
    renewable_uncertainty : float in [0, 1].
    rng                   : NumPy random Generator.
    """
    if n_cuttable <= 0:
        return 0
    max_cut = max(1, int(n_cuttable * 0.5 * (1.0 - renewable_uncertainty)))
    return int(rng.integers(low=1, high=max_cut + 1))
