"""Multiple-testing helpers for scale-by-scale lead-lag inference."""
from __future__ import annotations

import numpy as np


def step_down_pvalues(
    boot_dist: np.ndarray,
    observed: np.ndarray,
    *,
    center: np.ndarray | None = None,
    studentize: bool = True,
) -> np.ndarray:
    """Romano-Wolf step-down adjusted p-values across scales.

    boot_dist has shape (n_bootstrap, n_tests), observed has shape (n_tests,).
    center approximates the null (bootstrap mean if omitted); for
    percentile-style bootstrap output it recenters the draws before the max
    statistic. studentize divides observed and null statistics by each test's
    bootstrap standard deviation.

    This only adjusts across scales, it does not turn a descriptive bootstrap
    into a calibrated null: the adjusted p-values are only as good as the
    bootstrap design supplied by the caller.
    """
    boot = np.asarray(boot_dist, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if boot.ndim != 2:
        raise ValueError("boot_dist must have shape (n_bootstrap, n_tests)")
    if obs.ndim != 1:
        raise ValueError("observed must be a 1D array")
    if boot.shape[1] != obs.shape[0]:
        raise ValueError("boot_dist and observed disagree on n_tests")

    finite_rows = np.all(np.isfinite(boot), axis=1)
    boot = boot[finite_rows]
    if boot.shape[0] < 2:
        raise ValueError("at least two finite bootstrap rows are required")

    if center is None:
        center_arr = np.mean(boot, axis=0)
    else:
        center_arr = np.asarray(center, dtype=float)
        if center_arr.shape != obs.shape:
            raise ValueError("center must have the same shape as observed")

    null_draws = boot - center_arr
    scale = np.ones(obs.shape, dtype=float)
    if studentize:
        scale = np.std(null_draws, axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale <= 0)] = 1.0

    observed_stats = np.abs(obs) / scale
    null_stats = np.abs(null_draws) / scale
    order = np.argsort(-observed_stats)

    adjusted = np.empty_like(observed_stats, dtype=float)
    running_max = 0.0
    n_boot = null_stats.shape[0]
    for step, idx in enumerate(order):
        remaining = order[step:]
        max_stats = np.max(null_stats[:, remaining], axis=1)
        p_value = (1.0 + np.sum(max_stats >= observed_stats[idx])) / (n_boot + 1.0)
        running_max = max(running_max, float(p_value))
        adjusted[idx] = running_max

    return np.clip(adjusted, 0.0, 1.0)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg FDR-adjusted p-values.

    NaN inputs remain NaN and are excluded from the number of tested
    hypotheses.
    """
    pvals = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(pvals, np.nan, dtype=float)
    finite = np.isfinite(pvals)
    if not np.any(finite):
        return adjusted

    flat = pvals[finite].ravel()
    order = np.argsort(flat)
    ranked = flat[order] * len(flat) / np.arange(1, len(flat) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(flat)
    out[order] = np.clip(ranked, 0.0, 1.0)
    adjusted[finite] = out
    return adjusted


__all__ = ["step_down_pvalues", "benjamini_hochberg"]
