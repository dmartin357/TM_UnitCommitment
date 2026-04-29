"""
Stage 1 I/O — serialize and deserialize AllPeriodsResult to/from JSON.

Intended use
------------
After a Stage 1 run, call save_stage1_result() to persist the populations and
stats to disk.  In a later session (or a Stage 2 smoke test), call
load_stage1_result() to restore them without re-running the GA.

File format
-----------
JSON, with format_version for forward-compatibility.  gen_names are stored
once at the top level (shared across all chromosomes and periods).  Each
chromosome stores bits as a list of ints, fitness as a float (or null for
infeasible), and dispatch as a {name: MW} dict (or null).

    {
      "format_version": "1.1",
      "saved_at": "<ISO-8601 UTC timestamp>",
      "gen_names": ["GEN421", "GEN53", ...],
      "n_periods": 48,
      "demand_values": [14821.6, ...],
      "total_wall_seconds": 420.5,
      "populations": [
        {
          "max_size": 200,
          "chromosomes": [
            {"bits": [1, 0, ...], "fitness": 874321.5, "dispatch": {"GEN53": 156.0, ...},
             "reg_up": 412.3, "reg_down": 198.7},
            ...
          ]
        },
        ...
      ],
      "period_stats": [
        {
          "total_wall_seconds": 9.2,
          "n_generations": 45,
          ...
        },
        ...
      ]
    }
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..stage1_ga.chromosome import Chromosome
from ..stage1_ga.ga import GAStats
from ..stage1_ga.parallel import AllPeriodsResult
from ..stage1_ga.population import BoundedPopulation

FORMAT_VERSION = "1.2"
_SUPPORTED_VERSIONS = {"1.0", "1.1", "1.2"}


# ── Public API ────────────────────────────────────────────────────────────────

def save_stage1_result(result: AllPeriodsResult, path: str | Path) -> Path:
    """
    Serialize AllPeriodsResult to a JSON file.

    Parameters
    ----------
    result : AllPeriodsResult returned by run_all_periods().
    path   : destination file path (created or overwritten).

    Returns
    -------
    The resolved Path that was written.
    """
    path = Path(path)

    # Extract gen_names from the first non-empty population.
    gen_names: list[str] | None = None
    for pop in result.populations:
        if len(pop) > 0:
            gen_names = pop[0].gen_names
            break

    data = {
        "format_version": FORMAT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "gen_names": gen_names,
        "n_periods": len(result.populations),
        "demand_values": result.demand_values,
        "total_wall_seconds": result.total_wall_seconds,
        "populations": [_serialize_population(pop) for pop in result.populations],
        "period_stats": [_serialize_stats(s) for s in result.period_stats],
    }

    path.write_text(json.dumps(data, indent=2))
    return path


def load_stage1_result(path: str | Path) -> AllPeriodsResult:
    """
    Deserialize AllPeriodsResult from a JSON file produced by save_stage1_result().

    Parameters
    ----------
    path : path to the JSON file.

    Returns
    -------
    AllPeriodsResult with fully reconstructed BoundedPopulation objects.
    Chromosome hashes are recomputed from bits on construction, so duplicate
    detection remains correct.

    Raises
    ------
    ValueError  if the file's format_version is not supported.
    FileNotFoundError if the file does not exist.
    """
    path = Path(path)
    data = json.loads(path.read_text())

    version = data.get("format_version")
    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unsupported Stage 1 file format version: {version!r}. "
            f"Supported: {sorted(_SUPPORTED_VERSIONS)}."
        )

    gen_names: list[str] = data["gen_names"]

    populations = [
        _deserialize_population(p_data, gen_names)
        for p_data in data["populations"]
    ]
    period_stats = [
        _deserialize_stats(s_data)
        for s_data in data["period_stats"]
    ]

    return AllPeriodsResult(
        populations=populations,
        period_stats=period_stats,
        demand_values=data["demand_values"],
        total_wall_seconds=data["total_wall_seconds"],
    )


# ── Serialization helpers ─────────────────────────────────────────────────────

def _serialize_population(pop: BoundedPopulation) -> dict:
    return {
        "max_size": pop.max_size,
        "chromosomes": [_serialize_chromosome(c) for c in pop],
    }


def _serialize_chromosome(c: Chromosome) -> dict:
    return {
        "bits":           c.bits.tolist(),
        "fitness":        c.fitness,           # float or None
        "dispatch":       c.dispatch,          # {name: MW} or None
        "reg_up":         c.reg_up,            # float or None
        "reg_down":       c.reg_down,          # float or None
        "renewable_loss": c.renewable_loss,    # float or None
    }


def _serialize_stats(s: GAStats) -> dict:
    # dataclasses.asdict converts tuples → lists (e.g. fitness_history entries).
    # That is fine for JSON; we restore tuples on deserialization.
    return dataclasses.asdict(s)


# ── Deserialization helpers ───────────────────────────────────────────────────

def _deserialize_population(
    data: dict,
    gen_names: list[str],
) -> BoundedPopulation:
    chromosomes = [
        _deserialize_chromosome(c_data, gen_names)
        for c_data in data["chromosomes"]
    ]
    # Chromosomes are stored best→worst; from_chromosomes preserves that order.
    return BoundedPopulation.from_chromosomes(
        max_size=data["max_size"],
        chromosomes=chromosomes,
    )


def _deserialize_chromosome(data: dict, gen_names: list[str]) -> Chromosome:
    bits = np.array(data["bits"], dtype=np.uint8)
    c = Chromosome(gen_names=gen_names, bits=bits)
    c.fitness  = data["fitness"]              # float or None
    c.dispatch = data["dispatch"]             # dict or None
    c.reg_up         = data.get("reg_up")            # None for v1.0/1.1 files
    c.reg_down       = data.get("reg_down")          # None for v1.0/1.1 files
    c.renewable_loss = data.get("renewable_loss")    # None for v1.0/1.1 files
    return c


def _deserialize_stats(data: dict) -> GAStats:
    data = dict(data)
    # fitness_history was serialized as [[gen, fitness], ...]; restore tuples.
    data["fitness_history"] = [
        (int(entry[0]), float(entry[1]))
        for entry in data.get("fitness_history", [])
    ]
    return GAStats(**data)
