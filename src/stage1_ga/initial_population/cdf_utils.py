"""
CDF utility for initial population generation.

Provides two public functions:

  build_static_ordering()
      Called once at InitialPopulationGenerator construction time.
      Returns the stable must-run-first generator ordering used as the bit-vector
      index throughout all chromosomes produced by that generator instance.

  compute_location_probs()
      Called inside the iterative cut loop on every iteration.
      Given the current set of committed may-run generators, sorts them by
      Pmax descending and returns a discrete probability distribution over
      cut positions derived from the Pmax CDF.

Dual-CDF structure
------------------
Generators are ranked by Pmax descending.  Two CDFs share this rank order:

  Pmax CDF[k] = sum(pmax[rank 1 .. k])
  Pmin CDF[k] = sum(pmin[rank 1 .. k])   ← same rank order, NOT sorted by Pmin

The Pmax CDF values (normalized) serve as the cut-location probability
distribution: higher-ranked (smaller-Pmax) positions receive more probability
mass, biasing cuts toward smaller generators.

The last (highest) value of each CDF equals the total committed Pmax/Pmin,
which is what the stopping criteria compares against demand thresholds.

Sort attribute note
-------------------
The baseline algorithm always sorts by 'power_output_maximum' descending.
Alternative sort attributes are supported for experimentation:
  'power_output_minimum', 'ramp_up_limit', 'ramp_down_limit', 'mc_min', 'mc_max'

Location distribution types
----------------------------
  'cdf'  p[k] = cumsum(pmax)[k] / sum(cumsum)   (default; larger k → higher prob)
  'pdf'  p[k] = pmax[k] / sum(pmax)             (weight ∝ individual pmax value)
"""

from __future__ import annotations

import numpy as np


# ── Derived attribute helpers ─────────────────────────────────────────────────

def _mc_range(gen_data: dict) -> tuple[float, float]:
    pts = gen_data.get("piecewise_production", [])
    mcs = [
        (pts[i + 1]["cost"] - pts[i]["cost"]) / (pts[i + 1]["mw"] - pts[i]["mw"])
        for i in range(len(pts) - 1)
        if pts[i + 1]["mw"] - pts[i]["mw"] > 0
    ]
    return (min(mcs), max(mcs)) if mcs else (np.nan, np.nan)


_SCALAR_ATTRS = {
    "power_output_maximum",
    "power_output_minimum",
    "ramp_up_limit",
    "ramp_down_limit",
}

_DERIVED_ATTRS: dict = {
    "mc_min": lambda g: _mc_range(g)[0],
    "mc_max": lambda g: _mc_range(g)[1],
}


def _get_attr_value(gen_data: dict, attr: str) -> float:
    if attr in _SCALAR_ATTRS:
        val = gen_data.get(attr, np.nan)
        return float(val) if not isinstance(val, list) else float(val[0])
    if attr in _DERIVED_ATTRS:
        return _DERIVED_ATTRS[attr](gen_data)
    raise ValueError(
        f"Unknown sort attribute '{attr}'. "
        f"Valid options: {sorted(_SCALAR_ATTRS | _DERIVED_ATTRS.keys())}"
    )


# ── Static ordering (called once at construction) ─────────────────────────────

def build_static_ordering(
    generators: dict,
    sort_attribute: str,
    sort_ascending: bool,
) -> tuple[list[str], list[str], list[str]]:
    """
    Partition generators into must-run and may-run groups, sort each by the
    chosen attribute, and return:
        (sorted_names, must_run_sorted, may_run_sorted)

    sorted_names = must_run_sorted + may_run_sorted
    This ordering defines the bit-vector index for all produced chromosomes.
    """
    must_run = [n for n, g in generators.items() if g.get("must_run", 0) == 1]
    may_run  = [n for n, g in generators.items() if g.get("must_run", 0) != 1]

    def _sort_key(n: str) -> tuple:
        v = _get_attr_value(generators[n], sort_attribute)
        # NaN sorts last regardless of direction
        return (np.isnan(v), v if not np.isnan(v) else 0.0)

    must_run_sorted = sorted(must_run, key=_sort_key, reverse=not sort_ascending)
    may_run_sorted  = sorted(may_run,  key=_sort_key, reverse=not sort_ascending)
    sorted_names    = must_run_sorted + may_run_sorted

    return sorted_names, must_run_sorted, may_run_sorted


# ── Per-iteration CDF/PDF (called inside the cut loop) ───────────────────────

def compute_location_probs(
    committed_names: list[str],
    generators: dict,
    sort_attribute: str,
    sort_ascending: bool,
    location_dist_type: str,
) -> tuple[list[str], np.ndarray]:
    """
    Given the current set of committed may-run generator names, sort them by
    the chosen attribute and return a discrete probability distribution over
    their positions.

    Parameters
    ----------
    committed_names    : names of currently committed may-run generators.
    generators         : full {name: gen_data} dict.
    sort_attribute     : attribute to sort and weight by.
    sort_ascending     : True → smallest first.
    location_dist_type : 'pdf' or 'cdf'.

    Returns
    -------
    (sorted_names, probs)
      sorted_names : list[str] — names in sort order.
      probs        : np.ndarray — probability mass at each position.
    """
    attr_vals = {n: _get_attr_value(generators[n], sort_attribute) for n in committed_names}

    sorted_names = sorted(
        committed_names,
        key=lambda n: (np.isnan(attr_vals[n]), attr_vals[n] if not np.isnan(attr_vals[n]) else 0.0),
        reverse=not sort_ascending,
    )

    vals = np.array(
        [max(attr_vals[n], 0.0) if not np.isnan(attr_vals[n]) else 0.0 for n in sorted_names],
        dtype=float,
    )

    if vals.sum() == 0:
        vals = np.ones(len(sorted_names), dtype=float)

    if location_dist_type == "uniform":
        probs = np.ones(len(sorted_names), dtype=float) / len(sorted_names)
    elif location_dist_type == "pdf":
        probs = vals / vals.sum()
    elif location_dist_type == "cdf":
        cdf = np.cumsum(vals)
        probs = cdf / cdf.sum()
    else:
        raise ValueError(
            f"Unknown location_dist_type '{location_dist_type}'. "
            "Use 'uniform', 'pdf', or 'cdf'."
        )

    return sorted_names, probs
