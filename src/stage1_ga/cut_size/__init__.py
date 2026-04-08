"""
Cut-size distribution registry.

To add a new distribution:
  1. Create a new module in this package (e.g., beta.py) with a `sample`
     function matching the signature:
       sample(n_cuttable: int, renewable_uncertainty: float,
              rng: np.random.Generator) -> int
  2. Import it here and add it to REGISTRY with a string key.
  3. Pass that key as GAConfig.cut_size_distribution.
"""

from .uniform import sample as _uniform_sample

REGISTRY: dict = {
    "uniform": _uniform_sample,
}
