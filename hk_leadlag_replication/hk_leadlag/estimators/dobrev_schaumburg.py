"""Dobrev & Schaumburg (2017) lead-lag estimator - baseline.

Simplified DS estimator based on tick-by-tick correlations of mid-price
changes. The full DS framework uses a more elaborate weighting; this follows
HK20's empirical section: aggregate signed-tick coincidences over a search
grid and pick the argmax.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hk_leadlag.base import BaseLeadLagEstimator, NonSyncSeries, LeadLagResult


@dataclass(slots=True)
class DobrevSchaumburgEstimator(BaseLeadLagEstimator):
    """Tick-coincidence DS-like estimator over a discrete lag grid."""

    delta_N: float = 1.0
    grid_half_width: int = 100
    name: str = "dobrev_schaumburg"

    @property
    def search_grid(self) -> np.ndarray:
        return self.delta_N * np.arange(-self.grid_half_width, self.grid_half_width + 1)

    def fit(self, series: NonSyncSeries) -> LeadLagResult:
        # Sign-of-return indicators on each native tick.
        s1 = np.sign(np.diff(series.prices1))
        s2 = np.sign(np.diff(series.prices2))
        # Bin ticks onto the equidistant grid of step delta_N.
        t1 = series.times1[1:]
        t2 = series.times2[1:]
        t_min = min(t1.min(), t2.min())
        t_max = max(t1.max(), t2.max())
        n_bins = int(np.ceil((t_max - t_min) / self.delta_N)) + 1
        b1 = np.zeros(n_bins)
        b2 = np.zeros(n_bins)
        np.add.at(b1, ((t1 - t_min) / self.delta_N).astype(int), s1)
        np.add.at(b2, ((t2 - t_min) / self.delta_N).astype(int), s2)
        grid = self.search_grid
        contrast = np.zeros(len(grid))
        for k, tau in enumerate(grid):
            shift = int(round(tau / self.delta_N))
            if shift >= 0:
                contrast[k] = np.dot(b1[: n_bins - shift], b2[shift:])
            else:
                contrast[k] = np.dot(b1[-shift:], b2[: n_bins + shift])
        idx = int(np.argmax(contrast))
        return LeadLagResult(
            theta_hat=np.array([float(grid[idx])]),
            levels=np.array([0]),
            contrast=contrast[None, :],
            grid=grid,
            estimator_name=self.name,
            metadata={"contrast_max": float(contrast[idx])},
        )
