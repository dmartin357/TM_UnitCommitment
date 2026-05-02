"""
Pre-solve target smoothing utilities.

Applies optional smoothing to the raw per-period n_committed targets produced
by the pre-solve sampler.  Raw targets can bounce between adjacent periods
(e.g. 28 → 34 → 27) due to sampling noise; smoothing reduces this variance
and produces a cleaner transition sequence for Stage 2 to follow.

Available methods
-----------------
"none"
    Pass-through — targets are returned unchanged.

"moving_average"
    Centered moving average over `window` periods.  Edge periods use a
    narrower window (all available neighbors) so no padding or truncation
    occurs.  Result is rounded to the nearest integer so Stage 2 always
    receives an integer commitment count.

Adding a new method
-------------------
1. Write a function with signature:
       def my_method(targets: list[int], **kwargs) -> list[int]
2. Add it to SMOOTHING_REGISTRY below.
3. Expose any parameters you need via TargetGuidanceConfig in control_panel.py.
"""

from __future__ import annotations


# ── Registry ──────────────────────────────────────────────────────────────────

SMOOTHING_REGISTRY: dict[str, object] = {}   # populated at bottom of file


# ── Implementations ───────────────────────────────────────────────────────────

def _no_smoothing(targets: list[int], **_kwargs) -> list[int]:
    return list(targets)


def _moving_average(targets: list[int], window: int = 3, **_kwargs) -> list[int]:
    """Centered moving average; edge periods use all available neighbors."""
    if window < 1:
        raise ValueError(f"smoothing_window must be >= 1, got {window}")
    n    = len(targets)
    half = window // 2
    smoothed = []
    for i in range(n):
        lo  = max(0, i - half)
        hi  = min(n, i + half + 1)
        avg = sum(targets[lo:hi]) / (hi - lo)
        smoothed.append(round(avg))
    return smoothed


SMOOTHING_REGISTRY["none"]           = _no_smoothing
SMOOTHING_REGISTRY["moving_average"] = _moving_average


# ── Public API ────────────────────────────────────────────────────────────────

def smooth_targets(
    targets: list[int],
    method: str,
    window: int = 3,
) -> list[int]:
    """
    Apply smoothing to a list of integer per-period n_committed targets.

    Parameters
    ----------
    targets : raw per-period targets (one int per time period).
    method  : smoothing method name — must be a key in SMOOTHING_REGISTRY.
    window  : window size in periods; only used by methods that require it
              (e.g. 'moving_average').

    Returns
    -------
    Smoothed targets (new list; input is never modified).
    """
    if method not in SMOOTHING_REGISTRY:
        raise ValueError(
            f"Unknown smoothing method {method!r}. "
            f"Valid options: {sorted(SMOOTHING_REGISTRY)}"
        )
    fn = SMOOTHING_REGISTRY[method]
    return fn(targets, window=window)
