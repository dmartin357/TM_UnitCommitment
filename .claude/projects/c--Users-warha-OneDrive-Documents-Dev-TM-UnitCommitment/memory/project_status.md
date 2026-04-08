---
name: Project Implementation Status
description: Current implementation state of the 4-stage UC heuristic — which stages exist and what they do
type: project
---

Stage 1 (GA) is fully implemented in src/stage1_ga/. Stage 2 (Graph Builder) stub is complete in src/stage2_graph/. Stage 3 (Shortest Path) and Stage 4 (Min Up/Down) are not yet started.

**Stage 1 files:**
- chromosome.py — binary commitment vector + SHA-256 hash duplicate detection
- config.py — GAConfig dataclass with all tunable parameters
- population.py — BoundedPopulation (sorted, bounded, hash-deduplicated)
- ga.py — run_stage1_ga() single-period entry point + GAStats
- parallel.py — run_all_periods() ProcessPoolExecutor runner
- initial_population/generator.py — CDF/PDF-based iterative cut sampling
- initial_population/cdf_utils.py — helper for computing location probabilities
- ed/piecewise_linear.py — Pyomo piecewise-linear ED subproblem (fitness fn)
- operators/crossover.py — single_point crossover + CROSSOVER_REGISTRY
- operators/mutation.py — bit_flip mutation

**Why:** Stage 1 smoke-tested successfully on the FERC 2020-01-27 test case (48 periods, ~420s wall time per git history).

**Next up:** Stage 2 — Graph Builder (directed graph connecting chromosome populations across adjacent time periods, with ramp constraint enforcement and startup/shutdown cost edges).

**How to apply:** When user asks "what's next" or "where were we", Stage 2 is the immediate next task.
