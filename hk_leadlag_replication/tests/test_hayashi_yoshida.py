"""Tests for the Hayashi-Yoshida cross-covariance estimator."""
from __future__ import annotations

import numpy as np
import pytest

from hk_leadlag.base import NonSyncSeries
from hk_leadlag.estimators.hayashi_yoshida import HayashiYoshidaEstimator, _ccov_single_lag


def _build_series(rng: np.random.Generator, n: int, rho: float, lag: float = 0.0):
    """Synchronous bivariate Brownian with correlation rho, observed every dt=1."""
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(max(1 - rho ** 2, 0.0)) * rng.standard_normal(n)
    times = np.arange(n + 1, dtype=float)
    x1 = np.concatenate([[0.0], np.cumsum(z1)])
    x2 = np.concatenate([[0.0], np.cumsum(z2)])
    series = NonSyncSeries(times1=times, prices1=x1, times2=times + lag, prices2=x2)
    return series


def test_hy_zero_lag_identifies_correlation_sign():
    rng = np.random.default_rng(0)
    pos = _build_series(rng, 5_000, rho=0.7)
    neg = _build_series(rng, 5_000, rho=-0.7)
    hy = HayashiYoshidaEstimator(grid=np.array([0.0]))
    assert hy.cross_cov(pos)[0] > 0
    assert hy.cross_cov(neg)[0] < 0


def test_hy_argmax_recovers_codebase_lag_convention():
    """When series 2 is delayed, this implementation returns a negative lag.

    The paper's empirical convention reports positive values when asset 1 leads
    asset 2. The current code shifts series-2 intervals by +tau, so its raw
    output has the opposite sign: theta < 0 means series 1 leads series 2.
    """
    rng = np.random.default_rng(1)
    series = _build_series(rng, 5_000, rho=0.7, lag=2.0)
    grid = np.arange(-10, 11, dtype=float)
    hy = HayashiYoshidaEstimator(grid=grid)
    res = hy.fit(series)
    assert abs(res.theta_hat[0] + 2.0) <= 1.0


def test_single_lag_helper_independent_zero():
    rng = np.random.default_rng(2)
    n = 1000
    a1 = np.arange(n, dtype=float)
    b1 = a1 + 1
    a2 = a1.copy()
    b2 = b1.copy()
    dx1 = rng.standard_normal(n)
    dx2 = rng.standard_normal(n)
    s = _ccov_single_lag(a1, b1, dx1, a2, b2, dx2)
    # Independent → expectation 0; sample value finite.
    assert np.isfinite(s)
