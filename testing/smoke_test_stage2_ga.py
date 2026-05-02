"""
Smoke test for Stage 2 GA forward pass.

Loads saved Stage 1 results (JSON) and pre-solve targets, then runs the
Stage 2 sequential forward pass and prints the per-period decision table.

Requires a Stage 1 result file produced by smoke_test_stage1.py
(set STAGE1_RESULT_PATH below).

Usage (from repo root, with power-systems conda env active):
    python testing/smoke_test_stage2_ga.py
"""

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io.stage1_io import load_stage1_result
from src.io.xlsx_export import classify_fuel, compute_reg_per_period, export_solution_xlsx
from src.pre_solve_stage.smoothing import smooth_targets
from src.stage2_ga.forward_pass import run_stage2_forward_pass
from testing.control_panel import CURRENT

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_FORMAT  = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"
_LOG_DATEFMT = "%H:%M:%S"
_formatter   = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)

# Console handler — INFO and above only
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_formatter)

# Root logger at DEBUG so the file handler (added in __main__) captures everything;
# the console handler's own level keeps the terminal clean.
_root = logging.getLogger()
_root.setLevel(logging.DEBUG)
_root.handlers.clear()
_root.addHandler(_console_handler)

# ── Paths (derived from control panel) ────────────────────────────────────────
INSTANCE_PATH      = CURRENT.instance_path
CACHE_DIR          = Path(__file__).parent / "cache"
STAGE1_RESULT_PATH = CACHE_DIR / f"stage1_{INSTANCE_PATH.stem}.json"

# ── Instance loader (mirrors smoke_test_stage1.py) ────────────────────────────

class InstanceData:
    def __init__(self, thermal, total_demand, renewable_min, renewable_max):
        self.thermal = thermal
        self.total_demand = total_demand
        n = len(total_demand)
        renewable_max_eff = [min(renewable_max[t], total_demand[t]) for t in range(n)]
        self.renewable_expected = [
            (renewable_min[t] + renewable_max_eff[t]) / 2.0 for t in range(n)
        ]
        self.thermal_demand = [
            max(0.0, total_demand[t] - self.renewable_expected[t]) for t in range(n)
        ]
        self.renewable_min      = renewable_min
        self.renewable_max      = renewable_max_eff
        self.renewable_reg_up   = [self.renewable_expected[t] - renewable_min[t] for t in range(n)]
        self.renewable_reg_down = [renewable_max_eff[t] - self.renewable_expected[t] for t in range(n)]


def load_instance(path: Path) -> InstanceData:
    with open(path) as f:
        data = json.load(f)
    thermal = data["thermal_generators"]
    demand_raw = data["demand"]
    if isinstance(demand_raw, list):
        total_demand = [float(d) for d in demand_raw]
    elif isinstance(demand_raw, dict):
        n = len(next(iter(demand_raw.values())))
        total_demand = [sum(bus[t] for bus in demand_raw.values()) for t in range(n)]
    else:
        raise ValueError(f"Unexpected demand format: {type(demand_raw)}")
    n_periods = len(total_demand)
    renewable_min = [0.0] * n_periods
    renewable_max = [0.0] * n_periods
    for name, gen_data in data.get("renewable_generators", {}).items():
        if re.search(r"HYDRO", name, re.IGNORECASE):
            continue
        for t in range(n_periods):
            pmin_s = gen_data.get("power_output_minimum", [])
            pmax_s = gen_data.get("power_output_maximum", [])
            if t < len(pmin_s):
                renewable_min[t] += float(pmin_s[t])
            if t < len(pmax_s):
                renewable_max[t] += float(pmax_s[t])
    return InstanceData(thermal, total_demand, renewable_min, renewable_max)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading instance: {INSTANCE_PATH}")
    inst = load_instance(INSTANCE_PATH)
    n_periods = len(inst.total_demand)
    print(f"Instance: {len(inst.thermal)} thermal generators, {n_periods} periods\n",
          flush=True)

    # ── Load Stage 1 results ──────────────────────────────────────────────────
    print(f"Loading Stage 1 results: {STAGE1_RESULT_PATH}")
    stage1_result = load_stage1_result(STAGE1_RESULT_PATH)
    populations   = stage1_result.populations
    print(f"Stage 1: {len(populations)} populations loaded\n", flush=True)

    # ── Run Stage 2 forward pass ──────────────────────────────────────────────
    print("Running Stage 2 forward pass…\n", flush=True)
    rng = np.random.default_rng(CURRENT.rng_seed)

    # Pre-solve target guidance
    tg = CURRENT.target_guidance
    target_n_committed = None
    if tg.enabled:
        raw_targets = [
            round(sum(c.n_committed for c in pop) / len(pop)) if len(pop) > 0 else 0
            for pop in populations
        ]
        target_n_committed = smooth_targets(
            raw_targets,
            method=tg.smoothing,
            window=tg.smoothing_window,
        )
        print(
            f"Pre-solve targets ({tg.smoothing} smoothing"
            + (f", window={tg.smoothing_window}" if tg.smoothing != "none" else "")
            + f"):  min={min(target_n_committed)}"
            f"  max={max(target_n_committed)}"
            f"  avg={sum(target_n_committed)/len(target_n_committed):.1f}\n",
            flush=True,
        )
    else:
        print("Pre-solve target guidance disabled — random candidate order.\n",
              flush=True)

    result = run_stage2_forward_pass(
        populations=populations,
        generators=inst.thermal,
        thermal_demand_values=inst.thermal_demand,
        config=CURRENT.stage2_ga,
        rng=rng,
        target_n_committed=target_n_committed,
    )

    result.print_summary()

    xlsx_path = Path(__file__).parent / "results" / f"stage2_dispatch_{INSTANCE_PATH.stem}.xlsx"
    _export_heuristic_xlsx(result, inst, xlsx_path)


# ── Excel export ──────────────────────────────────────────────────────────────

def _export_heuristic_xlsx(result, inst: InstanceData, path: Path) -> None:
    """
    Build thermal + renewable dispatch arrays from the Stage 2 result and
    call the shared export_solution_xlsx utility.
    """
    thermal_gens = inst.thermal

    with open(INSTANCE_PATH) as f:
        renewable_gens = json.load(f).get("renewable_generators", {})

    # t=0 = initial state (power_output_t0); t=1..N-1 = Stage 2 decisions
    n_periods = max((d.period for d in result.decisions), default=0) + 1
    dispatch_by_period: dict[int, dict[str, float]] = {
        d.period: d.dispatch for d in result.decisions
    }

    # ── Thermal dispatch & committed sets ─────────────────────────────────────
    t0_committed = [n for n, g in thermal_gens.items() if g.get("unit_on_t0", 0) == 1]
    committed_per_period: list[list[str]] = [t0_committed]
    for t in range(1, n_periods):
        committed_per_period.append(list(dispatch_by_period.get(t, {}).keys()))

    thermal_dispatch: dict[str, list[float]] = {}
    for name, gen in thermal_gens.items():
        t0_mw = float(gen.get("power_output_t0", 0.0)) if name in t0_committed else 0.0
        row = [t0_mw] + [
            dispatch_by_period.get(t, {}).get(name, 0.0)
            for t in range(1, n_periods)
        ]
        thermal_dispatch[name] = row

    committed_counts = [len(c) for c in committed_per_period]

    # ── Renewable dispatch ────────────────────────────────────────────────────
    # Hydro: pmax[t]  (must-run, pmin ≈ pmax — accurate)
    # Wind/Solar/CSP: (pmin + pmax) / 2  (consistent with Stage 1 thermal demand calc)
    renewable_dispatch: dict[str, list[float]] = {}
    for name, gen in renewable_gens.items():
        pmin_s   = gen.get("power_output_minimum", [])
        pmax_s   = gen.get("power_output_maximum", [])
        is_hydro = bool(re.search(r"HYDRO", name, re.IGNORECASE))
        row = []
        for t in range(n_periods):
            lo = float(pmin_s[t]) if t < len(pmin_s) else 0.0
            hi = float(pmax_s[t]) if t < len(pmax_s) else 0.0
            row.append(hi if is_hydro else (lo + hi) / 2.0)
        renewable_dispatch[name] = row

    # ── Regulation per period ─────────────────────────────────────────────────
    reg_up, reg_down = compute_reg_per_period(
        generators=thermal_gens,
        thermal_dispatch=thermal_dispatch,
        committed_per_period=committed_per_period,
        n_periods=n_periods,
    )

    export_solution_xlsx(
        path=path,
        generators_thermal=thermal_gens,
        generators_renewable=renewable_gens,
        n_periods=n_periods,
        thermal_dispatch=thermal_dispatch,
        renewable_dispatch=renewable_dispatch,
        thermal_demand=inst.thermal_demand[:n_periods],
        renewable_expected=inst.renewable_expected[:n_periods],
        renewable_min_vals=inst.renewable_min[:n_periods],
        renewable_max_vals=inst.renewable_max[:n_periods],
        committed_thermal=committed_counts,
        reg_up_total=reg_up,
        reg_down_total=reg_down,
        period_labels=[f"t={t}" for t in range(n_periods)],
        sheet_title="Heuristic",
    )
    print(f"  Dispatch xlsx exported → {path}\n")


if __name__ == "__main__":
    # File handler — DEBUG and above
    _results_dir = Path(__file__).parent / "results"
    _results_dir.mkdir(parents=True, exist_ok=True)
    _log_path = _results_dir / f"stage2_{INSTANCE_PATH.stem}.log"
    _fh = logging.FileHandler(_log_path, mode="w", encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(_formatter)
    logging.getLogger().addHandler(_fh)
    print(f"  Detailed log → {_log_path}\n")

    main()
