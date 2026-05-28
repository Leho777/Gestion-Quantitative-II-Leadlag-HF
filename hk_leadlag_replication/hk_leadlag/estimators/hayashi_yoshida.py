"""Hayashi-Yoshida cross-covariance estimator for non-synchronous data.

Implements the lagged HY estimator used as the building block in HK20:

    U_hat_N(tau) =
        sum_{I in O1, J in O2: bar(I) <= bar(J) + tau} dX1(I) dX2(J) K(I, J + tau)
                                                      if tau >= 0,
        sum_{I in O1, J in O2: bar(J) <= bar(I) - tau} dX1(I) dX2(J) K(I + tau, J)
                                                      if tau <  0,

where K(I, J) = 1{I cap J != empty}, and ``+ tau`` shifts an interval in time.

The Numba-accelerated path scans all lags in a single nested loop, with a
pure-Python fallback when Numba isn't available.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit  # type: ignore
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def deco(f):
            return f
        if args and callable(args[0]):
            return args[0]
        return deco

from hk_leadlag.base import BaseLeadLagEstimator, NonSyncSeries, LeadLagResult


@dataclass(slots=True)
class HayashiYoshidaEstimator(BaseLeadLagEstimator):
    """Lagged Hayashi-Yoshida cross-covariance and lead-lag estimator.

    ``grid`` holds the lags tau (continuous-time units) at which U_hat_N(tau)
    is evaluated.
    """

    grid: np.ndarray
    name: str = "hayashi_yoshida"

    def cross_cov(self, series: NonSyncSeries) -> np.ndarray:
        """Return U_hat_N(tau) for tau in self.grid."""
        return _ccov_lagged(
            series.times1, series.prices1,
            series.times2, series.prices2,
            np.asarray(self.grid, dtype=float),
        )

    def fit(self, series: NonSyncSeries) -> LeadLagResult:
        """Fit a single lead-lag = argmax_tau |U_hat(tau)|."""
        u = self.cross_cov(series)
        idx = int(np.argmax(np.abs(u)))
        theta_hat = float(self.grid[idx])
        return LeadLagResult(
            theta_hat=np.array([theta_hat]),
            levels=np.array([0]),
            contrast=u[None, :],
            grid=np.asarray(self.grid),
            estimator_name=self.name,
            metadata={"u_hat_max": float(u[idx])},
        )


def _ccov_lagged(
    t1: np.ndarray,
    x1: np.ndarray,
    t2: np.ndarray,
    x2: np.ndarray,
    taus: np.ndarray,
) -> np.ndarray:
    """Lagged HY cross-covariance over an array of lags.

    For each tau, shift the second observation grid by tau (positive tau =>
    series 2 is delayed) and apply the standard HY estimator on overlapping
    increment intervals. Numba-accelerated when available.
    """
    t1 = np.ascontiguousarray(t1, dtype=np.float64)
    t2 = np.ascontiguousarray(t2, dtype=np.float64)
    x1 = np.ascontiguousarray(x1, dtype=np.float64)
    x2 = np.ascontiguousarray(x2, dtype=np.float64)
    taus = np.ascontiguousarray(taus, dtype=np.float64)
    return _ccov_lagged_jit(t1, x1, t2, x2, taus)


@njit(cache=True, fastmath=True)
def _ccov_lagged_jit(
    t1: np.ndarray,
    x1: np.ndarray,
    t2: np.ndarray,
    x2: np.ndarray,
    taus: np.ndarray,
) -> np.ndarray:
    """Numba-compiled inner loop for all lags."""
    n1 = len(t1) - 1
    n2 = len(t2) - 1
    n_tau = len(taus)
    out = np.empty(n_tau, dtype=np.float64)
    for k in range(n_tau):
        tau = taus[k]
        i = 0
        j = 0
        s = 0.0
        while i < n1 and j < n2:
            a1i = t1[i]
            b1i = t1[i + 1]
            a2j = t2[j] + tau
            b2j = t2[j + 1] + tau
            if b1i <= a2j:
                i += 1
                continue
            if b2j <= a1i:
                j += 1
                continue
            s += (x1[i + 1] - x1[i]) * (x2[j + 1] - x2[j])
            if b1i < b2j:
                i += 1
            else:
                j += 1
        out[k] = s
    return out


def _ccov_single_lag(
    a1: np.ndarray, b1: np.ndarray, dx1: np.ndarray,
    a2: np.ndarray, b2: np.ndarray, dx2: np.ndarray,
) -> float:
    """Pure-Python single-lag HY estimator (kept for tests and fallback)."""
    n1 = len(a1)
    n2 = len(a2)
    i = 0
    j = 0
    s = 0.0
    while i < n1 and j < n2:
        if b1[i] <= a2[j]:
            i += 1
            continue
        if b2[j] <= a1[i]:
            j += 1
            continue
        s += dx1[i] * dx2[j]
        if b1[i] < b2[j]:
            i += 1
        else:
            j += 1
    return s
