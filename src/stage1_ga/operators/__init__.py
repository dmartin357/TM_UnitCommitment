"""
Crossover operator registry.

To add a new operator:
  1. Create a module in this package with a function matching the signature:
       fn(parent_a, parent_b, must_run_mask, rng) -> tuple[Chromosome, Chromosome]
  2. Import it here and add it to CROSSOVER_REGISTRY with a string key.
  3. Pass that key as GAConfig.crossover_operator.
"""

from .crossover import single_point as _single_point

CROSSOVER_REGISTRY: dict = {
    "single_point": _single_point,
}
