# CLAUDE.md — Unit Commitment Heuristic Algorithm (Thesis Project)

## Project Overview

This is a master's thesis project in electrical engineering developing a novel **multi-stage heuristic algorithm for Unit Commitment (UC)**. The goal is to produce a computationally efficient heuristic that achieves competitive solution quality compared to full MILP solvers (CBC, Gurobi, CPLEX), targeting a **2–5% optimality gap in under 30 minutes** on industrial-scale instances.

The benchmark reference is a CBC 2.10.12 solution on the **power-grid-lib RTS-GMLC UC test case** (instance `rts_gmlc/2020-01-27.json`, 48 periods, 24-hour horizon), which achieved a **0.66% optimality gap ($41,519,481 objective) after 35.2 hours**. That is the bar we are building toward, not replicating — the point is to get close much faster.

This is a **research script/notebook project**, not a production package. Prioritize clarity, modularity, and reproducibility over software engineering polish.

---

## Current Implementation Status

| Component | Status | Notes |
|---|---|---|
| Stage 1 — GA (per time period) | **Complete** | Smoke-tested on RTS-GMLC 2020-01-27 (48 periods) |
| Stage 2 — GA Forward Pass | **Complete** | Greedy forward pass using Stage 1 populations |
| Stage 2 — Graph Builder | **Stub** | Rectification=2×, net adj tolerance=±5% demand |
| Stage 3 — Shortest Path | Not started | Depends on Graph Builder output |
| Stage 4 — Min Up/Down Time | Not started | Depends on Stage 3 output |
| **GA v2 — Initial Population** | **In progress** | New forward-pass architecture (see below) |

### GA v2 is the active development branch

The project has pivoted from the original 4-stage graph-based architecture toward a more classical genetic algorithm (GA v2) that operates as a **sequential forward-pass population generator**. The original Stage 1/2 code is preserved and still works; GA v2 is a parallel development path intended to replace it as the primary solution method.

---

## Algorithm Architecture

### Original Architecture (Stages 1–4) — Preserved, not actively developed

The original design decomposes UC constraints across four sequential stages:

**Stage 1** runs an independent GA per time period to build diverse chromosome populations (binary commitment vectors), evaluated by piecewise-linear Economic Dispatch. No cross-period constraints.

**Stage 2 (GA forward pass)** runs a greedy sequential forward pass: for each period, takes the Stage 1 population as candidates, repairs min up/down time violations, runs single-period ED with ramp-adjusted bounds, and selects the first feasible candidate as the winner.

**Stage 2 (Graph Builder)** builds a directed graph connecting Stage 1 chromosomes between adjacent periods using rectification logic to handle near-feasible ramp violations.

**Stage 3** finds lowest-cost paths through the Stage 2 graph using Dijkstra/NetworkX.

**Stage 4** post-processes paths to enforce min up/down time constraints via repair/re-routing.

---

### GA v2 — Active Development: Forward-Pass Population Generator

GA v2 is a **classical-style GA** that generates an initial population of **complete UC solutions** (full T-period schedules), rather than per-period chromosome pools. Each solution is produced by a greedy sequential forward pass that respects temporal constraints from the start. The population then feeds into a mating/crossover stage (not yet implemented).

#### GA v2 Initial Population Generation

**Overall structure:** Generate N complete T-period UC solutions. Each solution is seeded by a distinct period-1 commitment set; the forward pass then greedily builds the remaining periods.

**Per-period algorithm (applied at each t = 1 … T−1):**

1. **Identify deterministic must-runs.** Hydro generators have pmin = pmax and are treated as fixed dispatch (subtracted from demand before ED). Wind and solar are variable — included in the ED as optimization variables with time-varying bounds `[pmin[t], pmax[t]]` and a nominal cost of $0.01/MWh to keep them at the upper bound without degeneracy.

2. **Compute thermal demand target.** `thermal_demand_target = total_demand[t] − hydro_fixed[t] − renewable_expected[t]` where `renewable_expected[t] = (pmin[t] + pmax[t]) / 2` summed over wind/solar. This target is used only for the pmax violation check during cutting.

3. **Classify thermal units** (based on previous period's committed state + generator data):
   - `constrained_ON`: must stay ON — unit was ON and min up time not yet satisfied, OR dispatch was above ramp_shutdown_limit (cannot drop to zero in one period), OR must_run = 1
   - `constrained_OFF`: must stay OFF — unit was OFF and min down time not yet satisfied
   - `free`: all other units — eligible for the cutting pool

4. **Generate candidates via uniform random cutting.** Start with all free units committed (plus constrained_ON). Uniformly randomly shuffle the free pool and cut units one at a time. Stop when the next cut would cause `sum(pmax of remaining committed) < thermal_demand_target`. Each call to this procedure produces one candidate committed set.

5. **Run single-period ED** for each candidate:
   - Thermal units: piecewise-linear cost, bounds ramp-adjusted from previous period's dispatch (same ramp logic as Stage 2 — startup ramp for newly started units, up/down ramp limits for continuing units)
   - Variable renewables: linear cost at $0.01/MWh, bounded by `[pmin[t], pmax[t]]`
   - Hydro: subtracted from demand as fixed; not an optimization variable
   - Demand balance: `sum(thermal dispatch) + sum(renewable dispatch) = total_demand[t] − hydro_fixed[t]`

6. **Score candidates** on a weighted rank of two criteria:
   - Economics rank: lower total cost = better (rank 1 = cheapest)
   - Regulation range rank: higher `reg_up + reg_down` = better (rank 1 = most flexible); reg range computed from ramp-adjusted thermal bounds
   - Composite score: `econ_weight × cost_rank + reg_weight × reg_rank` — lower is better

7. **Select winner.** Candidate with best composite score advances. The winner's committed set and dispatch become the fleet state for the next period.

**Building the initial population:**
- Generate N t=1 candidates (cutting from t=0 initial state, respecting min up/down constraints from `unit_on_t0`, `time_up_t0`, `time_down_t0`)
- Each t=1 candidate seeds one complete forward pass through all T periods
- Only **complete solutions** (all periods successfully solved by ED) are eligible to be selected as the best; incomplete solutions are flagged in output

**Next step (not yet implemented):** Crossover and mutation operators on the population of complete solutions to generate successive generations.

---

## Key Design Decisions & Rationale

**Why pivot to GA v2 from the graph-based architecture?**
The original Stage 1 + Stage 2 graph approach separates temporal and capacity constraints across stages, but this makes it hard to generate diverse, temporally-feasible solutions efficiently. GA v2 generates complete, ramp-feasible, min-up/down-feasible solutions from the start, which are much more useful as a GA population — crossover and mutation on complete solutions is well-defined, whereas crossover on per-period chromosome pools is not.

**Why include renewables in the ED rather than subtracting them as fixed?**
The RTS-GMLC instance has wind/solar with `pmin[t] = 0` and `pmax[t] = time-varying forecast`. The CBC benchmark dispatches them as decision variables within those bounds, naturally pushing them to their upper bound since they have zero cost in the objective. GA v2 mirrors this: renewables enter the ED at $0.01/MWh, which makes them always fully dispatched up to forecast while keeping the LP numerically well-conditioned (no free variables).

**Why use weighted rank scoring rather than just picking the cheapest candidate?**
Pure cost minimization at each greedy step may produce schedules that are cheap locally but leave the fleet with no regulation headroom, making subsequent periods harder to solve. The weighted rank balances economics against flexibility, producing more robust solutions.

**Why min up/down constraints from t=0?**
The instance JSON provides `unit_on_t0`, `time_up_t0`, and `time_down_t0` for every thermal generator. These encode the initial fleet state and must be respected at t=1 — a unit that has been on for fewer periods than its minimum up time cannot shut down. GA v2 enforces this identically to Stage 2's repair logic.

**Why hash-based duplicate detection in Stage 1?**
The combinatorial chromosome space has significant overlap, especially in later GA generations. Avoiding re-evaluation of identical chromosomes saves ED/OPF subproblem solves, which are the computational bottleneck. (This applies to Stage 1; GA v2 uses random cutting, which has low duplication probability.)

---

## Benchmark Context

**Solver:** CBC 2.10.12 via Pyomo
**Instance:** power-grid-lib RTS-GMLC UC test case (`rts_gmlc/2020-01-27.json`), 24-hour horizon, 48 half-hour periods
**Problem scale:** ~330k rows, ~312k columns, ~147k binary variables after presolve
**Key results:**

| Milestone | Objective Value | Wall Clock |
|---|---|---|
| LP relaxation lower bound | $40,756,600 | ~207s |
| First feasible solution | $41,965,500 | ~7.8h |
| Best solution at termination | $41,519,481 | ~35.2h |
| Lower bound at termination | $41,245,691 | — |

**Final gap:** 0.66% (~$273k absolute)
**Integrality gap (LP vs best integer):** ~1.87%

**Key findings that inform heuristic design:**
- TwoMir (15,728) and Gomory (8,042) cuts dominated → min up/down time and ramp constraints are the primary combinatorial difficulty
- Feasibility pump needed 32 passes / 3.5h → naive LP rounding is a poor initialization strategy
- Presolve fixed 807 binaries outright → some commitment decisions are structurally forced
- Late improvement at node 1,552 ($41.88M → $41.52M) → need diversification, not just local refinement
- LP relaxation solved in ~207s → target heuristic solve time is 1–10x this (~200s–2000s)

**Renewable dispatch in CBC benchmark:** Wind and solar are dispatched as decision variables bounded by `[pmin[t], pmax[t]]` where `pmin[t] = 0` and `pmax[t]` is the time-varying forecast. Since renewables have no cost in the CBC objective, CBC dispatches them at their upper bound (full forecast) for all periods. Hydro has `pmin[t] = pmax[t]` (fixed output).

**Heuristic target:** 2–5% optimality gap in under 30 minutes on instances of this scale.

---

## Directory Structure

```
TM_UnitCommitment/                    # Repo root
├── CLAUDE.md                         # This file
├── uc_model.py                       # CBC MILP benchmark solver (Pyomo)
├── benchmarks/
│   ├── convert_cbc_to_xlsx.py        # Converts CBC solution JSON → xlsx
│   ├── cbc_solution.json             # CBC optimal solution (obj=$41,519,481)
│   └── cbc_solution.xlsx             # CBC solution in expanded comparison format
├── design_documents/                 # Drawio flowcharts for algorithm stages
├── PG_Lib_FERC_Instance/             # RTS-GMLC instance EDA notebook + data
├── src/
│   ├── ga_v2/                        # GA v2: forward-pass population generator (active)
│   │   ├── config.py                 # GAv2Config dataclass
│   │   ├── candidate.py              # PeriodCandidate, ForwardSolution
│   │   ├── constraint_check.py       # classify_thermal_units (ON/OFF/free)
│   │   ├── cutting.py                # classify_renewables, generate_cut_candidate
│   │   ├── ed_single_period.py       # Pyomo ED: thermals + renewables
│   │   ├── scoring.py                # weighted rank candidate selection
│   │   ├── forward_pass.py           # build_forward_solution (one T-period solution)
│   │   └── population.py             # generate_initial_population (N solutions)
│   ├── stage1_ga/                    # Stage 1: per-period GA (complete, preserved)
│   │   ├── ga.py                     # run_stage1_ga() single-period entry point
│   │   ├── parallel.py               # run_all_periods() ProcessPoolExecutor runner
│   │   ├── ed/piecewise_linear.py    # Pyomo piecewise-linear ED (fitness function)
│   │   ├── initial_population/       # CDF-cut seed sampling
│   │   ├── population.py             # BoundedPopulation (sorted, SHA-256 dedup)
│   │   ├── chromosome.py             # binary commitment vector
│   │   └── operators/                # crossover, mutation
│   ├── stage2_ga/                    # Stage 2 GA forward pass (complete, preserved)
│   │   ├── forward_pass.py           # run_stage2_forward_pass()
│   │   ├── unit_state.py             # FleetState, UnitState, advance_fleet_state
│   │   ├── repair.py                 # repair_min_updown (also used by GA v2)
│   │   └── config.py
│   ├── stage2_graph/                 # Stage 2 graph builder (stub, preserved)
│   ├── pre_solve_stage/              # Monte Carlo pre-solve (n_committed targets)
│   └── io/
│       ├── stage1_io.py              # Stage 1 result JSON serialization
│       └── xlsx_export.py            # Shared xlsx export (CBC + heuristic)
├── testing/
│   ├── control_panel.py              # Single source of truth for experiment params
│   ├── smoke_test_ga_v2.py           # GA v2 smoke test (active)
│   ├── smoke_test_stage1.py          # Stage 1 smoke test (preserved)
│   ├── smoke_test_stage2_ga.py       # Stage 2 GA forward pass smoke test (preserved)
│   ├── cache/                        # Cached Stage 1 JSON results
│   └── results/                      # xlsx outputs, logs
└── testing/control_panel.py          # Experiment config (instance, seeds, weights)
```

---

## Solver Preferences

**For ED subproblem (all stages):**
1. **Gurobi** (preferred — fastest)
2. **CPLEX** (fallback)
3. **CBC** (fallback if commercial solvers unavailable)

**For OPF (if network constraints active):**
1. **IPOPT** (available in environment, suitable for nonlinear AC OPF)

**Pyomo** is the modeling layer for all optimization subproblems.

---

## Environment

- **Conda environment:** `power-systems` (Python 3.11.10)
- **Key packages:** Pyomo 6.9.4, NetworkX 3.5, NumPy 1.26.4, SciPy 1.12.0, Pandas 2.3.1, Matplotlib 3.10.8, Plotly 6.6.0, openpyxl (xlsx export)
- **Solvers available:** CBC 2.10.12, IPOPT 3.14.19, Gurobi (external), CPLEX (external)
- **IDE:** VSCode with Claude Code extension
- **Notable packages:** EGRET 0.5.6 (Sandia power systems toolkit with UC/ED formulations built on Pyomo — worth exploring for subproblem formulations)

---

## Development Approach

- **Format:** Research scripts and Jupyter notebooks (not a production package)
- **Test incrementally:** Each stage should be testable independently before integration; each has its own smoke test
- **Separate test paths:** GA v2 (`smoke_test_ga_v2.py`) is completely independent from Stage 1/2 tests — the two development paths can coexist
- **Control panel:** `testing/control_panel.py` is the single file to edit for experiment parameters; smoke tests import from it
- **Commit frequently:** GitHub is used to sync state between VSCode development sessions

---

## Design Documents

Drawio flowchart diagrams live in `design_documents/`:
- `TreyMartin_UC_Visual.drawio` — overall algorithm overview (original 4-stage architecture)
- `TreyMartin_Stage_1_GA_ver_1.drawio` — Stage 1 GA flowchart
- `TreyMartin_Stage_2_GraphBuilder_ver_1.drawio` — Stage 2 graph builder flowchart
- `GA_v2_InitialPopulation_Reqs.txt` — written requirements for GA v2 initial population generation

---

## Related Resources

- **power-grid-lib UC instances:** https://github.com/power-grid-lib/pglib-uc
- **Kazarlis et al. (1996):** Foundational GA for UC paper — IEEE Trans. Power Systems, Vol. 11, No. 1, pp. 83–92
- **UC_GA repo (pwdemars):** Python implementation of Kazarlis (1996) — useful reference for GA operators
- **EGRET docs:** https://github.com/grid-parity-exchange/Egret
