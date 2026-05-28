"""Wavelet-based scale-by-scale lead-lag estimator (Hayashi & Koike, 2020).

Given the lagged Hayashi-Yoshida cross-covariance ``U_hat_N(tau)`` and the
autocorrelation wavelet ``Psi_j(l)`` at level j, the scale-j cross-covariance
estimator is

    Gamma_hat_{N-j+1}(tau) = sum_{l=-(L_j-1)}^{L_j-1} U_hat_N(tau - l Delta_N) Psi_j(l).

The lead-lag at scale j is then

    theta_hat_j = argmax_{tau in G_N} |Gamma_hat_{N-j+1}(tau)|.

Combines the ``HayashiYoshidaEstimator`` with the wavelet filters from
:mod:`hk_leadlag.wavelet`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hk_leadlag.base import BaseLeadLagEstimator, NonSyncSeries, LeadLagResult
from hk_leadlag.estimators.hayashi_yoshida import HayashiYoshidaEstimator
from hk_leadlag.wavelet import DaubechiesFilter, autocorrelation_wavelet


@dataclass(slots=True)
class WaveletLeadLagEstimator(BaseLeadLagEstimator):
    """Multi-scale lead-lag estimator of Hayashi & Koike (2020).

    ``delta_N`` is the finest time resolution Delta_N (same unit as the times).
    ``j_max`` is the highest scale estimated (j = 1..j_max). ``grid_half_width``
    sets the search grid ``[-G, G]`` in steps of Delta_N; G must be large enough
    to bracket the true theta_j at every scale. ``daub`` defaults to ``db10``
    (length L = 20 as in HK20).
    """

    delta_N: float = 1.0
    j_max: int = 8
    grid_half_width: int = 100
    daub: DaubechiesFilter = field(default_factory=lambda: DaubechiesFilter("db10"))
    name: str = "wavelet_leadlag"

    def __post_init__(self) -> None:
        if self.j_max < 1:
            raise ValueError("j_max must be >= 1")

    @property
    def search_grid(self) -> np.ndarray:
        """G_N = {-G, -G+1, ..., G} * Delta_N (in continuous-time units)."""
        return self.delta_N * np.arange(-self.grid_half_width, self.grid_half_width + 1)

    def fit(self, series: NonSyncSeries) -> LeadLagResult:
        """Estimate theta_j for j = 1..j_max."""
        # Gamma needs U_hat_N(tau - l * Delta_N) for l in [-L_j+1, L_j-1] and
        # tau in G_N, so pre-compute U_hat on G_N extended by the largest L_j.
        max_Lj = (2**self.j_max - 1) * (self.daub.L - 1) + 1
        ext_lags = np.arange(
            -self.grid_half_width - max_Lj + 1,
            self.grid_half_width + max_Lj,
        )
        ext_grid = self.delta_N * ext_lags
        hy = HayashiYoshidaEstimator(grid=ext_grid)
        u_hat = hy.cross_cov(series)

        # For each j, convolve U_hat with Psi_j and locate the max.
        grid = self.search_grid
        n_grid = len(grid)
        theta = np.zeros(self.j_max)
        contrast = np.zeros((self.j_max, n_grid))
        levels = np.arange(1, self.j_max + 1)

        for j_idx, j in enumerate(levels):
            psi_j = autocorrelation_wavelet(self.daub, int(j))
            L_j = (len(psi_j) + 1) // 2  # psi_j has length 2 L_j - 1
            gamma = np.convolve(u_hat, psi_j[::-1], mode="valid")
            # Slice the part of gamma that lines up with tau in G_N.
            offset = max_Lj - L_j  # extra lags on the left
            gamma_on_grid = gamma[offset : offset + n_grid]
            contrast[j_idx, :] = np.abs(gamma_on_grid)
            theta[j_idx] = grid[int(np.argmax(np.abs(gamma_on_grid)))]

        return LeadLagResult(
            theta_hat=theta,
            levels=levels,
            contrast=contrast,
            grid=grid,
            estimator_name=self.name,
            metadata={
                "delta_N": self.delta_N,
                "j_max": self.j_max,
                "wavelet": self.daub.name,
                "L": self.daub.L,
                "n1": series.n1,
                "n2": series.n2,
                "T": series.T,
            },
        )
