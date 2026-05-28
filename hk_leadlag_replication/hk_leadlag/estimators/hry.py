"""Hoffmann-Rosenbaum-Yoshida (2013) lead-lag estimator - baseline."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hk_leadlag.base import BaseLeadLagEstimator, NonSyncSeries, LeadLagResult
from hk_leadlag.estimators.hayashi_yoshida import HayashiYoshidaEstimator


@dataclass(slots=True)
class HRYEstimator(BaseLeadLagEstimator):
    """Single-scale HRY lead-lag estimator: theta_hat = argmax_tau |U_hat_N(tau)|.

    This is a special case of the wavelet estimator at the lowest level - no
    multi-scale structure. Reported alongside the multi-scale results for
    comparison.
    """

    delta_N: float = 1.0
    grid_half_width: int = 100
    name: str = "hry"

    @property
    def search_grid(self) -> np.ndarray:
        return self.delta_N * np.arange(-self.grid_half_width, self.grid_half_width + 1)

    def fit(self, series: NonSyncSeries) -> LeadLagResult:
        hy = HayashiYoshidaEstimator(grid=self.search_grid)
        u = hy.cross_cov(series)
        idx = int(np.argmax(np.abs(u)))
        return LeadLagResult(
            theta_hat=np.array([float(self.search_grid[idx])]),
            levels=np.array([0]),
            contrast=np.abs(u)[None, :],
            grid=self.search_grid,
            estimator_name=self.name,
            metadata={"u_hat_max": float(u[idx])},
        )
