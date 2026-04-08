# CLAUDE.md — Unit Commitment Heuristic Algorithm (Thesis Project)

## Project Overview

This is a master's thesis project in electrical engineering developing a novel **multi-stage heuristic algorithm for Unit Commitment (UC)**. The goal is to produce a computationally efficient heuristic that achieves competitive solution quality compared to full MILP solvers (CBC, Gurobi, CPLEX), targeting a **2–5% optimality gap in under 30 minutes** on industrial-scale instances.

The benchmark reference is a CBC 2.10.12 solution on the **power-grid-lib FERC UC test case** (RTO scale), which achieved a **0.66% optimality gap ($41,519,481 objective) after 35.2 hours**. That is the bar we are building toward, not replicating — the point is to get close much faster.

This is a **research script/notebook project**, not a production package. Prioritize clarity, modularity, and reproducibility over software engineering polish.

---

## Algorithm Architecture

The algorithm consists of four stages. Each stage decomposes a distinct class of UC constraints:

### Stage 1 — Genetic Algorithm (Per Time Period)
- Runs **independently for each time period** in the UC horizon
- Handles **network constraints** (if any) and **min/max generation limits**
- Runs **Economic Dispatch (ED)** as the fitness evaluation subproblem (no network), or **OPF** (AC/DC) if network constraints are active
- Produces a **diverse, bounded population of chromosomes** per time period
- Uses **multiple CDF/PDF proxies** (sorted lists across dimensions) for initial generation
  - Cut group size distribution parameterized by renewable uncertainty (more uncertainty → smaller cuts → more units online → implicit reserves)
  - Cut group location distribution parameterized by renewable uncertainty and CDF type (e.g., Max Power Output CDF biased toward beginning/middle under high uncertainty)
- Uses **SHA hash-based duplicate detection** to avoid re-evaluating chromosomes
- Tracks best and worst fitness per time period; bounded population discards worst when full
- **No cross-period feedback** — temporal feasibility is entirely handled by Stage 2

### Stage 2 — Graph Builder
- Takes Stage 1 chromosome populations and builds a **directed graph connecting chromosomes between adjacent time periods**
- Handles **ramping constraints** via a 4-level nested loop: time periods → chromosomes i → chromosomes i+1 → units
- **Rectification logic**: if a ramp violation is within ~2x the unit ramp limit, attempts to adjust power levels rather than immediately discarding the edge
- Tracks **net power adjustments** across all units; discards edge if total net adjustment exceeds predetermined limits
- Detects and adds **startup/shutdown costs** to edge costs (attributed to the later time period)
- **Edge cost = cost of target chromosome + startup/shutdown costs**
- Goal: collapse N^M potential paths to a manageable subset by eliminating ramp-infeasible edges

### Stage 3 — Shortest Path
- Takes the Stage 2 directed graph and finds **lowest-cost complete paths** from first to last time period
- Algorithm candidates: **Dijkstra** (non-negative edge weights, standard case) or **Bellman-Ford** (if negative weights arise)
- Implemented using **NetworkX**
- Should return a **ranked list of top-K paths** (not just single best) to support Stage 4

### Stage 4 — Minimum Up/Down Time Enforcement (NOT YET IMPLEMENTED)
- Post-processes paths from Stage 3 to handle **minimum up time and minimum down time constraints**
- These are the **primary source of combinatorial difficulty** in UC (confirmed by CBC run: TwoMir and Gomory cuts dominated, pointing to min up/down time and ramp constraints)
- Planned approach: hybrid of path filtering and local repair
  - Min down time violations: local look-back, handle via targeted edge re-costing or repair
  - Min up time violations: forward-looking, handle via local re-routing through Stage 2 graph over violated window
- Do NOT expect a single repair pass to be sufficient — may require iteration across top-K paths from Stage 3

---

## Key Design Decisions & Rationale

**Why decompose across stages?**
Each stage handles a distinct constraint class, reducing the complexity that any single optimization step must handle simultaneously. This mirrors how MILP solvers decompose via LP relaxation + cuts + branching, but in a more explicit and controllable way.

**Why run Stage 1 independently per time period?**
Simplifies the GA significantly. The GA only needs to find diverse, low-cost, network-feasible dispatch configurations — it does not need to worry about temporal transitions. Stage 2 handles all temporal reasoning.

**Why hash-based duplicate detection in Stage 1?**
The combinatorial chromosome space has significant overlap, especially in later GA generations. Avoiding re-evaluation of identical chromosomes saves ED/OPF subproblem solves, which are the computational bottleneck.

**Why the rectification logic in Stage 2?**
Purely discarding ramp-violating chromosome pairs would make the graph too sparse, potentially leaving Stage 3 with no feasible paths. Rectification preserves edges that are "close" to feasible, at the cost of small power adjustments.

**Why top-K paths from Stage 3?**
The CBC run showed a late improvement at node 1,552 from $41.88M to $41.52M — a qualitatively different commitment pattern, not a marginal tweak. This means the greedy shortest path may not survive Stage 4's min up/down time repair. Having K candidates gives Stage 4 fallback options.

---

## Benchmark Context

**Solver:** CBC 2.10.12 via Pyomo  
**Instance:** power-grid-lib FERC UC test case (RTO scale, 24-hour horizon)  
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

**Heuristic target:** 2–5% optimality gap in under 30 minutes on instances of this scale.

---

## Directory Structure

```
thesis-dev/                          # Primary thesis development directory
├── CLAUDE.md                        # This file
├── data/                            # Local data (gitignored if large)
├── eda/                             # Exploratory Data Analysis notebooks
│   └── ferc_eda.ipynb               # EDA for FERC benchmark instance
├── src/                             # Algorithm implementation
│   ├── stage1_ga/                   # Stage 1: Genetic Algorithm
│   ├── stage2_graph/                # Stage 2: Graph Builder
│   ├── stage3_shortest_path/        # Stage 3: Shortest Path
│   └── stage4_updown/               # Stage 4: Min Up/Down Time (TBD)
├── tests/                           # Unit tests
├── results/                         # Solver output, logs, result CSVs
└── notebooks/                       # Scratch/analysis notebooks

pglib-uc/                            # power-grid-lib repo clone (separate directory)
├── uc_model.py                      # Pyomo UC model file
└── data/                            # JSON instance files
```

> Note: `thesis-dev/` and `pglib-uc/` are in separate directories. Use a config variable or environment variable to point at the pglib data path rather than hardcoding it.

---

## Solver Preferences

**For ED subproblem (Stage 1 fitness evaluation):**
1. **Gurobi** (preferred — fastest)
2. **CPLEX** (fallback)
3. **CBC** (fallback if commercial solvers unavailable)

**For OPF (if network constraints active):**
1. **IPOPT** (available in environment, suitable for nonlinear AC OPF)

**Pyomo** is the modeling layer for all optimization subproblems.

---

## Environment

- **Conda environment:** `power-systems` (Python 3.11.10)
- **Key packages:** Pyomo 6.9.4, NetworkX 3.5, NumPy 1.26.4, SciPy 1.12.0, Pandas 2.3.1, Matplotlib 3.10.8, Plotly 6.6.0
- **Solvers available:** CBC 2.10.12, IPOPT 3.14.19, Gurobi (external), CPLEX (external)
- **IDE:** VSCode with Claude Code extension
- **Notable packages:** EGRET 0.5.6 (Sandia power systems toolkit with UC/ED formulations built on Pyomo — worth exploring for subproblem formulations)

---

## Development Approach

- **Format:** Research scripts and Jupyter notebooks (not a production package)
- **Start small:** Validate algorithm on a small power-grid-lib UC instance before scaling to FERC RTO case
- **EDA first:** Replicate FERC EDA notebook for whichever small instance is selected
- **Test incrementally:** Each stage should be testable independently before integration
- **Commit frequently:** GitHub is used to sync state between VSCode development sessions and the Claude.ai chat project used for design discussions

---

## Related Resources

- **power-grid-lib UC instances:** https://github.com/power-grid-lib/pglib-uc
- **Kazarlis et al. (1996):** Foundational GA for UC paper — IEEE Trans. Power Systems, Vol. 11, No. 1, pp. 83–92
- **UC_GA repo (pwdemars):** Python implementation of Kazarlis (1996) — useful reference for GA operators
- **EGRET docs:** https://github.com/grid-parity-exchange/Egret