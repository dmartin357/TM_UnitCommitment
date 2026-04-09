"""
Stage 2 — Graph Builder.

build_graph() takes the per-period chromosome populations from Stage 1 and
constructs a directed NetworkX graph connecting chromosomes between adjacent
time periods.

Graph structure
---------------
  Nodes : (period_idx, chrom_idx)
            period_idx  — 0-based time period index
            chrom_idx   — position in BoundedPopulation._chroms (sorted best→worst)
  Edges : (period_i, j) → (period_i+1, k)
            weight              — cost of chromosome k + startup/shutdown costs
            startup_shutdown_cost — total transition cost on this edge

Ramp constraint handling (per unit, per candidate edge)
-------------------------------------------------------
OFF → OFF   No check needed.

OFF → ON    Startup.
  1. Add startup cost for unit (first-tier stub — see _preprocess_generators).
  2. Clamp p_{i+1} to ramp_startup_limit if exceeded.
  3. Check ramp_up from effective p_i = 0 to (possibly clamped) p_{i+1}.
     If violated and within rectification_multiplier × ramp_up_limit,
     reduce p_{i+1} to p_i + ramp_up_limit (clamped to [pmin, pmax]).
     If still violated → discard edge.

ON  → OFF   Shutdown.
  1. Clamp p_i to ramp_shutdown_limit if exceeded.
  2. Check ramp_down from (possibly clamped) p_i to effective p_{i+1} = 0.
     If violated and within rectification_multiplier × ramp_down_limit,
     reduce p_i to ramp_down_limit (clamped to [pmin, pmax]).
     If still violated → discard edge.

ON  → ON    Normal transition.
  Ramp-up violation  (p_{i+1} - p_i > ramp_up_limit):
    If within multiplier × limit → reduce p_{i+1} to p_i + ramp_up_limit
    (clamped to [pmin, pmax]).  If pmin pushes new p_{i+1} above the limit
    → discard.
  Ramp-down violation (p_i - p_{i+1} > ramp_down_limit):
    If within multiplier × limit → raise p_{i+1} to p_i - ramp_down_limit
    (clamped to [pmin, pmax]).  If pmax pushes new p_{i+1} below the floor
    is impossible but pmin might block → discard.

Net adjustment tolerance
------------------------
After all units are processed, the total signed MW adjustment applied to
period i's dispatch and to period i+1's dispatch are each checked against
net_adjustment_tolerance × demand for that period.  Exceeding either bound
discards the edge.

Startup cost stub
-----------------
The startup[] list in the instance data supports multiple lag tiers (different
costs depending on how long the unit has been offline).  The current stub
always uses the first (and in the FERC instance, typically only) tier.
Full lag-dependent lookup requires tracking offline duration across paths,
which is a Stage 2 enhancement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import networkx as nx

from ..stage1_ga.chromosome import Chromosome
from ..stage1_ga.population import BoundedPopulation
from .config import GraphBuilderConfig

logger = logging.getLogger(__name__)


# ── Statistics ────────────────────────────────────────────────────────────────

@dataclass
class GraphBuilderStats:
    """Collected metrics from a build_graph() call."""

    total_wall_seconds: float = 0.0

    # Edge counts
    n_edges_considered: int = 0
    n_edges_added: int = 0
    n_edges_discarded_ramp: int = 0        # violation > rectification threshold
    n_edges_discarded_adjustment: int = 0  # net MW adjustment exceeded tolerance

    # Rectification
    n_rectifications_attempted: int = 0
    n_rectifications_succeeded: int = 0

    # Startup / shutdown detections (per unit per edge)
    n_startup_events: int = 0
    n_shutdown_events: int = 0

    # ── Derived ──────────────────────────────────────────────────────────────

    @property
    def n_edges_discarded(self) -> int:
        return self.n_edges_discarded_ramp + self.n_edges_discarded_adjustment

    @property
    def edge_acceptance_rate(self) -> float:
        if self.n_edges_considered == 0:
            return 0.0
        return self.n_edges_added / self.n_edges_considered

    def print_summary(self) -> None:
        print(f"\n{'=' * 60}")
        print(f"  Stage 2 — Graph Builder Summary")
        print(f"{'=' * 60}")
        print(f"  Total wall time       : {self.total_wall_seconds:.2f}s")
        print(f"  Edges considered      : {self.n_edges_considered:,}")
        print(f"  Edges added           : {self.n_edges_added:,}"
              f"  ({100 * self.edge_acceptance_rate:.1f}%)")
        print(f"  Discarded (ramp)      : {self.n_edges_discarded_ramp:,}")
        print(f"  Discarded (adj limit) : {self.n_edges_discarded_adjustment:,}")
        print(f"  Rectifications        : {self.n_rectifications_succeeded:,}"
              f" / {self.n_rectifications_attempted:,} succeeded")
        print(f"  Startup events        : {self.n_startup_events:,}")
        print(f"  Shutdown events       : {self.n_shutdown_events:,}")
        print(f"{'=' * 60}\n", flush=True)


# ── Public entry point ────────────────────────────────────────────────────────

def build_graph(
    populations: list[BoundedPopulation],
    generators: dict,
    demand_values: list[float],
    config: GraphBuilderConfig,
) -> tuple[nx.DiGraph, GraphBuilderStats]:
    """
    Build the Stage 2 directed graph from Stage 1 chromosome populations.

    Parameters
    ----------
    populations   : one BoundedPopulation per time period (Stage 1 output).
    generators    : {name: gen_data} thermal generator dict.
    demand_values : demand (MW) for each time period.
    config        : GraphBuilderConfig.

    Returns
    -------
    (DiGraph, GraphBuilderStats)
    """
    stats = GraphBuilderStats()
    t_start = time.monotonic()

    gen_meta = _preprocess_generators(generators)

    # Print population summary before edge-building begins
    n_periods = len(populations)
    total_feasible = sum(len(pop.feasible()) for pop in populations)
    total_all      = sum(len(pop) for pop in populations)
    print(f"\n{'=' * 60}")
    print(f"  Stage 2 — Input Population Summary  ({n_periods} periods)")
    print(f"{'=' * 60}")
    print(f"  {'Period':<8}  {'Demand (MW)':<13}  {'Total':>7}  {'Feasible':>9}  {'Infeasible':>11}")
    print(f"  {'-'*8}  {'-'*13}  {'-'*7}  {'-'*9}  {'-'*11}")
    for t, (pop, demand) in enumerate(zip(populations, demand_values)):
        n_feasible   = len(pop.feasible())
        n_infeasible = len(pop) - n_feasible
        print(f"  {t:<8}  {demand:<13.1f}  {len(pop):>7}  {n_feasible:>9}  {n_infeasible:>11}")
    print(f"  {'-'*8}  {'-'*13}  {'-'*7}  {'-'*9}  {'-'*11}")
    print(f"  {'Total':<8}  {'':13}  {total_all:>7}  {total_feasible:>9}")
    print(f"{'=' * 60}\n", flush=True)

    G = nx.DiGraph()

    for i in range(n_periods - 1):
        pop_i  = populations[i]
        pop_i1 = populations[i + 1]
        tol_i  = config.net_adjustment_tolerance * demand_values[i]
        tol_i1 = config.net_adjustment_tolerance * demand_values[i + 1]

        n_added_this_period = 0

        for j_idx, chrom_j in enumerate(pop_i):
            if not chrom_j.is_feasible() or chrom_j.dispatch is None:
                continue

            for k_idx, chrom_k in enumerate(pop_i1):
                if not chrom_k.is_feasible() or chrom_k.dispatch is None:
                    continue

                stats.n_edges_considered += 1

                result = _evaluate_edge(
                    chrom_j, chrom_k,
                    gen_meta,
                    tol_i, tol_i1,
                    enable_unit_rectification=config.enable_unit_rectification,
                    rectification_multiplier=config.rectification_multiplier,
                    enable_net_adjustment_check=config.enable_net_adjustment_check,
                    stats=stats,
                )

                if result is None:
                    continue

                edge_weight, startup_shutdown_cost = result

                node_j = (i,     j_idx)
                node_k = (i + 1, k_idx)
                G.add_node(node_j, period=i,     chrom_idx=j_idx)
                G.add_node(node_k, period=i + 1, chrom_idx=k_idx)
                G.add_edge(
                    node_j, node_k,
                    weight=edge_weight,
                    startup_shutdown_cost=startup_shutdown_cost,
                )
                n_added_this_period += 1
                stats.n_edges_added += 1

        logger.debug(
            "Period %d→%d: %d edges added (cumulative: %d)",
            i, i + 1, n_added_this_period, stats.n_edges_added,
        )

    stats.total_wall_seconds = time.monotonic() - t_start
    logger.info(
        "Stage 2 graph: %d nodes, %d edges, acceptance=%.1f%%, wall=%.2fs",
        G.number_of_nodes(), G.number_of_edges(),
        100 * stats.edge_acceptance_rate, stats.total_wall_seconds,
    )
    return G, stats


# ── Internal helpers ──────────────────────────────────────────────────────────

def _preprocess_generators(generators: dict) -> dict[str, dict]:
    """
    Pre-extract per-generator metadata needed for ramp checks and costs.

    Startup cost stub: uses the first entry in the startup[] list.  The FERC
    instance has a single tier per generator, so this is exact for that case.
    Multi-tier lag-dependent lookup is a future enhancement.
    """
    meta: dict[str, dict] = {}
    for name, g in generators.items():
        pts  = g["piecewise_production"]
        pmin = float(pts[0]["mw"])
        pmax = float(pts[-1]["mw"])

        startup_entries = g.get("startup", [])
        startup_cost = float(startup_entries[0]["cost"]) if startup_entries else 0.0

        # ramp_startup_limit / ramp_shutdown_limit of 0 means no separate
        # startup/shutdown ramp constraint (treat as unconstrained).
        raw_start  = float(g.get("ramp_startup_limit",  0.0))
        raw_stop   = float(g.get("ramp_shutdown_limit", 0.0))

        meta[name] = {
            "pmin":               pmin,
            "pmax":               pmax,
            "ramp_up_limit":      float(g.get("ramp_up_limit",   1e9)),
            "ramp_down_limit":    float(g.get("ramp_down_limit", 1e9)),
            "ramp_startup_limit": raw_start if raw_start > 0 else pmax,
            "ramp_shutdown_limit":raw_stop  if raw_stop  > 0 else pmax,
            "startup_cost":       startup_cost,
        }
    return meta


def _evaluate_edge(
    chrom_j: Chromosome,
    chrom_k: Chromosome,
    gen_meta: dict[str, dict],
    tol_i: float,
    tol_i1: float,
    enable_unit_rectification: bool,
    rectification_multiplier: float,
    enable_net_adjustment_check: bool,
    stats: GraphBuilderStats,
) -> tuple[float, float] | None:
    """
    Check ramp feasibility and compute the edge cost for the (j → k) pair.

    Returns (edge_weight, startup_shutdown_cost) if accepted, else None.
    edge_weight = chrom_k.fitness + startup_shutdown_cost.

    Works on local copies of dispatch — original Chromosome objects are never
    mutated.
    """
    gen_names = chrom_j.gen_names
    bits_j    = chrom_j.bits
    bits_k    = chrom_k.bits

    # Local dispatch copies (may be adjusted during rectification)
    dispatch_j = dict(chrom_j.dispatch)
    dispatch_k = dict(chrom_k.dispatch)

    net_adj_i  = 0.0   # cumulative MW adjustment to period-i dispatch
    net_adj_i1 = 0.0   # cumulative MW adjustment to period-(i+1) dispatch
    startup_shutdown_cost = 0.0

    for idx, name in enumerate(gen_names):
        on_j = bool(bits_j[idx])
        on_k = bool(bits_k[idx])

        if not on_j and not on_k:
            continue  # OFF→OFF: no ramp check

        meta   = gen_meta[name]
        pmin   = meta["pmin"]
        pmax   = meta["pmax"]
        p_j    = dispatch_j[name] if on_j else 0.0
        p_k    = dispatch_k[name] if on_k else 0.0
        adj_j  = 0.0
        adj_k  = 0.0

        if not on_j and on_k:
            # ── STARTUP ──────────────────────────────────────────────────────
            stats.n_startup_events += 1
            startup_shutdown_cost += meta["startup_cost"]

            # Step 1: clamp p_k to ramp_startup_limit
            startup_limit = meta["ramp_startup_limit"]
            if p_k > startup_limit + 1e-6:
                new_p_k = max(min(startup_limit, pmax), pmin)
                adj_k  += new_p_k - p_k
                p_k     = new_p_k

            # Step 2: ramp_up check from effective p_j = 0
            rectified = _rectify_ramp_up(
                p_from=0.0, p_to=p_k,
                pmin_to=pmin, pmax_to=pmax,
                ramp_up_limit=meta["ramp_up_limit"],
                multiplier=rectification_multiplier if enable_unit_rectification else 1.0,
                stats=stats,
            )
            if rectified is None:
                stats.n_edges_discarded_ramp += 1
                return None
            new_p_k, extra = rectified
            adj_k += extra
            p_k    = new_p_k

        elif on_j and not on_k:
            # ── SHUTDOWN ─────────────────────────────────────────────────────
            stats.n_shutdown_events += 1

            # Step 1: clamp p_j to ramp_shutdown_limit
            shutdown_limit = meta["ramp_shutdown_limit"]
            if p_j > shutdown_limit + 1e-6:
                new_p_j = max(min(shutdown_limit, pmax), pmin)
                adj_j  += new_p_j - p_j
                p_j     = new_p_j

            # Step 2: ramp_down check from p_j to effective p_k = 0
            rectified = _rectify_ramp_down_from(
                p_from=p_j,
                pmin_from=pmin, pmax_from=pmax,
                ramp_down_limit=meta["ramp_down_limit"],
                multiplier=rectification_multiplier if enable_unit_rectification else 1.0,
                stats=stats,
            )
            if rectified is None:
                stats.n_edges_discarded_ramp += 1
                return None
            new_p_j, extra = rectified
            adj_j += extra
            p_j    = new_p_j

        else:
            # ── ON → ON ──────────────────────────────────────────────────────
            if p_k - p_j > meta["ramp_up_limit"] + 1e-6:
                rectified = _rectify_ramp_up(
                    p_from=p_j, p_to=p_k,
                    pmin_to=pmin, pmax_to=pmax,
                    ramp_up_limit=meta["ramp_up_limit"],
                    multiplier=rectification_multiplier if enable_unit_rectification else 1.0,
                    stats=stats,
                )
                if rectified is None:
                    stats.n_edges_discarded_ramp += 1
                    return None
                new_p_k, extra = rectified
                adj_k += extra
                p_k    = new_p_k

            elif p_j - p_k > meta["ramp_down_limit"] + 1e-6:
                rectified = _rectify_ramp_up_to(
                    p_from=p_j, p_to=p_k,
                    pmin_to=pmin, pmax_to=pmax,
                    ramp_down_limit=meta["ramp_down_limit"],
                    multiplier=rectification_multiplier if enable_unit_rectification else 1.0,
                    stats=stats,
                )
                if rectified is None:
                    stats.n_edges_discarded_ramp += 1
                    return None
                new_p_k, extra = rectified
                adj_k += extra
                p_k    = new_p_k

        # Write back adjusted values into local copies
        if on_j:
            dispatch_j[name] = p_j
        if on_k:
            dispatch_k[name] = p_k

        net_adj_i  += adj_j
        net_adj_i1 += adj_k

    # Net adjustment tolerance check
    if enable_net_adjustment_check:
        if abs(net_adj_i) > tol_i or abs(net_adj_i1) > tol_i1:
            stats.n_edges_discarded_adjustment += 1
            return None

    edge_weight = chrom_k.fitness + startup_shutdown_cost
    return edge_weight, startup_shutdown_cost


def _rectify_ramp_up(
    p_from: float,
    p_to: float,
    pmin_to: float,
    pmax_to: float,
    ramp_up_limit: float,
    multiplier: float,
    stats: GraphBuilderStats,
) -> tuple[float, float] | None:
    """
    Check ramp-up constraint (p_to - p_from <= ramp_up_limit).
    If violated and within multiplier × limit, reduce p_to toward the limit.

    Returns (new_p_to, adjustment) where adjustment = new_p_to - p_to (≤ 0).
    Returns None if the violation cannot be rectified.
    """
    delta = p_to - p_from
    if delta <= ramp_up_limit + 1e-6:
        return p_to, 0.0  # no violation

    stats.n_rectifications_attempted += 1

    if delta > multiplier * ramp_up_limit + 1e-6:
        return None  # violation too large for rectification

    target = p_from + ramp_up_limit
    target = max(min(target, pmax_to), pmin_to)

    if target - p_from > ramp_up_limit + 1e-6:
        # pmin forces target above the ramp limit — irrecoverable
        return None

    stats.n_rectifications_succeeded += 1
    return target, target - p_to


def _rectify_ramp_down_from(
    p_from: float,
    pmin_from: float,
    pmax_from: float,
    ramp_down_limit: float,
    multiplier: float,
    stats: GraphBuilderStats,
) -> tuple[float, float] | None:
    """
    Check ramp-down constraint from p_from to an effective p_to = 0 (shutdown).
    Constraint: p_from <= ramp_down_limit.
    If violated and within multiplier × limit, reduce p_from toward the limit.

    Returns (new_p_from, adjustment) where adjustment = new_p_from - p_from (≤ 0).
    Returns None if irrecoverable.
    """
    if p_from <= ramp_down_limit + 1e-6:
        return p_from, 0.0  # no violation

    stats.n_rectifications_attempted += 1

    if p_from > multiplier * ramp_down_limit + 1e-6:
        return None  # too large

    target = ramp_down_limit
    target = max(min(target, pmax_from), pmin_from)

    if target > ramp_down_limit + 1e-6:
        # pmin forces target above the ramp limit — irrecoverable
        return None

    stats.n_rectifications_succeeded += 1
    return target, target - p_from


def _rectify_ramp_up_to(
    p_from: float,
    p_to: float,
    pmin_to: float,
    pmax_to: float,
    ramp_down_limit: float,
    multiplier: float,
    stats: GraphBuilderStats,
) -> tuple[float, float] | None:
    """
    Check ramp-down constraint for ON→ON (p_from - p_to <= ramp_down_limit).
    If violated and within multiplier × limit, raise p_to toward the floor
    p_from - ramp_down_limit.

    Returns (new_p_to, adjustment) where adjustment = new_p_to - p_to (≥ 0).
    Returns None if irrecoverable.
    """
    delta = p_from - p_to
    if delta <= ramp_down_limit + 1e-6:
        return p_to, 0.0  # no violation

    stats.n_rectifications_attempted += 1

    if delta > multiplier * ramp_down_limit + 1e-6:
        return None

    target = p_from - ramp_down_limit
    target = max(min(target, pmax_to), pmin_to)

    if p_from - target > ramp_down_limit + 1e-6:
        # pmin forces target too low — irrecoverable
        return None

    stats.n_rectifications_succeeded += 1
    return target, target - p_to
