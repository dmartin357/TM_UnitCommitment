"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              EXPERIMENT CONTROL PANEL — Unit Commitment Heuristic           ║
╚══════════════════════════════════════════════════════════════════════════════╝

This is the single file you edit to change experiment parameters.  All smoke
tests import the `CURRENT` instance defined at the bottom of this file; you
should never need to touch the smoke test scripts themselves when tuning the
algorithm.

LAYOUT
------
1. Per-stage config dataclasses (imported from src/)
2. TargetGuidanceConfig  — controls pre-solve target guiding in Stage 2
3. ExperimentConfig      — top-level aggregator (one field per stage/concern)
4. CURRENT               — the active experiment (edit this block)

STUBS & FUTURE OPTIONS (documented here for visibility)
-------------------------------------------------------
Stage 1 — location_dist_type
    "uniform"   ← current default  (equal probability per cut position)
    "cdf"       future: normalize cumulative Pmax; bias cuts toward smaller units
    "pdf"       future: weight proportional to individual Pmax value

Stage 1 — crossover_operator
    "single_point"  ← current default  (one crossover point)
    future: "two_point", "uniform" crossover operators

Stage 2 GA — selection_mode
    "uniform"   ← current default  (each feasible candidate equally likely)
    future: "cost_weighted"  (lower cost → higher selection probability)
            "rank_weighted"  (rank-ordered probability)

Stage 2 GA — target_guidance.smoothing
    "none"           ← pass-through (raw sampled targets)
    "moving_average" ← centered moving average over `smoothing_window` periods

Stage 2 Graph — startup cost tiers
    Currently always uses the first (hottest/cheapest) startup cost tier.
    Future: lag-dependent multi-tier lookup based on offline duration.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow importing from repo root when running smoke tests directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pre_solve_stage.config import PreSolveConfig
from src.stage1_ga.config import GAConfig
from src.stage2_ga.config import Stage2Config
from src.stage2_graph.config import GraphBuilderConfig


# ── Pre-solve target guidance ─────────────────────────────────────────────────

@dataclass
class TargetGuidanceConfig:
    """
    Controls how Stage 1 pre-solve targets guide Stage 2 candidate ordering.

    When enabled, Stage 2 sorts the Stage 1 candidate pool for each period by
    proximity to `target_n_committed[t]` before trying them sequentially.  This
    is a *soft* guide — constraints (min up/down time, ramp limits) still
    override the ordering, so the actual committed count may differ from target.

    Fields
    ------
    enabled
        False → ignore pre-solve targets entirely; use random candidate order.
    percentile
        Which percentile of the Stage 1 population's n_committed distribution
        to use as the target for each period.  50 = median.
    smoothing
        Smoothing method applied to raw per-period targets before passing to
        Stage 2.  See src/pre_solve_stage/smoothing.py for implementations.
        "none"           — pass-through (raw sampled targets may bounce)
        "moving_average" — centered moving average; reduces period-to-period
                           noise at the cost of tracking sharp ramps less well
    smoothing_window
        Window size in periods for methods that require it (e.g. moving_average).
        Odd values give a symmetric window; even values are rounded up.
    """
    enabled:          bool  = True
    percentile:       float = 50.0
    smoothing:        str   = "none"    # "none" | "moving_average"
    smoothing_window: int   = 3


# ── Top-level experiment config ───────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """
    Single source of truth for all algorithm parameters in one run.

    Compose this once in the CURRENT block below; all smoke tests import it.

    Instance
    --------
    pglib_uc_root    : root of the power-grid-lib pglib-uc clone on disk.
    instance_subpath : path relative to pglib_uc_root to the target JSON file.

    Stage 1
    -------
    stage1           : GAConfig — GA hyperparameters and solver choice.
                       target_n_committed_tolerance controls how close (in units
                       committed) the best chromosome must be to the pre-solve
                       target before the GA stops early.
    stage1_n_workers : worker processes for parallel per-period GA runs.
                       None → all logical CPUs.
    stage1_mode      : "all"    — run all periods in parallel (timing run).
                       "single" — one period, verbose stats (sanity check).
    stage1_single_period       : time-period index used only in "single" mode.
    stage1_use_presolve_targets: True  — run pre-solve before Stage 1 and use
                                         per-period n_committed targets to stop
                                         the GA early.
                                 False — skip pre-solve; Stage 1 runs without
                                         commitment count guidance.

    Stage 2 GA Forward Pass
    -----------------------
    stage2_ga        : Stage2Config — solver and selection mode.
    target_guidance  : TargetGuidanceConfig — pre-solve target guiding and
                       smoothing (see that dataclass for details).

    Stage 2 Graph Builder
    ---------------------
    stage2_graph      : GraphBuilderConfig — rectification and net-adj tolerance.
    stage2_graph_mode : "load_or_run" — load cached Stage 1 result if present,
                                        else run Stage 1 first.
                        "run_and_save" — always re-run Stage 1.
                        "load_only"    — fail fast if cache is missing.

    Pre-solve
    ---------
    pre_solve        : PreSolveConfig — Monte Carlo sampler parameters.
    """

    # ── Instance ──────────────────────────────────────────────────────────────
    pglib_uc_root:    Path = field(default_factory=lambda: Path("C:/gitrepos/power-grid-lib/pglib-uc"))
    instance_subpath: str  = "rts_gmlc/2020-01-27.json"

    # ── Pre-solve ─────────────────────────────────────────────────────────────
    pre_solve: PreSolveConfig = field(default_factory=PreSolveConfig)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    stage1:                      GAConfig   = field(default_factory=GAConfig)
    stage1_n_workers:            int | None = None   # None → all logical CPUs
    stage1_mode:                 str        = "all"  # "single" | "all"
    stage1_single_period:        int        = 0      # used only in "single" mode
    # When True, pre-solve runs before Stage 1 and supplies per-period n_committed
    # targets.  The GA stops early when the best feasible chromosome is within
    # stage1.target_n_committed_tolerance units of the target.
    stage1_use_presolve_targets: bool       = True

    # ── Stage 2 GA forward pass ───────────────────────────────────────────────
    stage2_ga:       Stage2Config        = field(default_factory=Stage2Config)
    target_guidance: TargetGuidanceConfig = field(default_factory=TargetGuidanceConfig)

    # ── Stage 2 graph builder ─────────────────────────────────────────────────
    stage2_graph:      GraphBuilderConfig = field(default_factory=GraphBuilderConfig)
    stage2_graph_mode: str                = "load_or_run"

    # ── Global random seed ────────────────────────────────────────────────────
    # Used as the base seed for Stage 1 (period t → seed + t) and the fixed
    # seed for Stage 2 GA.  Override stage2_ga.rng_seed separately if you want
    # Stage 2 to use a different seed from Stage 1.
    rng_seed: int = 42

    # ── Derived paths (read-only) ─────────────────────────────────────────────
    @property
    def instance_path(self) -> Path:
        return self.pglib_uc_root / self.instance_subpath


# ══════════════════════════════════════════════════════════════════════════════
#  CURRENT EXPERIMENT — edit this block to change what the smoke tests run
# ══════════════════════════════════════════════════════════════════════════════

CURRENT = ExperimentConfig(

    # ── Instance ──────────────────────────────────────────────────────────────
    instance_subpath="rts_gmlc/2020-01-27.json",

    # ── Pre-solve sampler ─────────────────────────────────────────────────────
    pre_solve=PreSolveConfig(
        n_samples=1_000,
        sort_attribute="power_output_maximum",
        sort_ascending=False,
        target_percentile=50.0,
    ),

    # ── Stage 1 GA ────────────────────────────────────────────────────────────
    stage1=GAConfig(
        population_size=100,
        initial_sample_size=50,
        sort_attribute="power_output_maximum",
        sort_ascending=False,
        location_dist_type="uniform",      # stub: "cdf" / "pdf" are future options
        crossover_operator="single_point", # stub: "two_point" is a future option
        mutation_rate=0.02,
        max_generations=100,
        max_wall_seconds=360.0,
        stagnation_limit=20,
        target_n_committed_tolerance=2,    # ±2 units from pre-solve target triggers early stop
        solver="auto",
    ),
    stage1_n_workers=None,              # None → all logical CPUs
    stage1_mode="all",                  # "single" for quick sanity check
    stage1_single_period=0,
    stage1_use_presolve_targets=True,   # False → skip pre-solve, no early stopping

    # ── Stage 2 GA forward pass ───────────────────────────────────────────────
    stage2_ga=Stage2Config(
        solver="auto",
        target_percentile=50.0,
        selection_mode="uniform",  # stub: "cost_weighted" / "rank_weighted" future
        rng_seed=42,
    ),

    # ── Pre-solve target guidance (Stage 2) ───────────────────────────────────
    target_guidance=TargetGuidanceConfig(
        enabled=True,
        percentile=50.0,
        smoothing="none",          # try "moving_average" to reduce period bounce
        smoothing_window=3,
    ),

    # ── Stage 2 graph builder ─────────────────────────────────────────────────
    stage2_graph=GraphBuilderConfig(
        enable_unit_rectification=True,
        rectification_multiplier=2.0,
        enable_net_adjustment_check=False,  # disable to observe its impact alone
        net_adjustment_tolerance=0.05,
    ),
    stage2_graph_mode="load_or_run",

    # ── Global random seed ────────────────────────────────────────────────────
    rng_seed=42,
)
