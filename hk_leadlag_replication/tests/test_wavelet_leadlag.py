"""End-to-end smoke test: simulate -> sample -> estimate -> recover theta."""
from __future__ import annotations

import numpy as np
import pytest

from hk_leadlag.estimators import WaveletLeadLagEstimator
from hk_leadlag.simulation import BivariateBrownianSimulator, RegularSampler
from hk_leadlag.wavelet import DaubechiesFilter


@pytest.mark.slow
def test_recover_known_theta_constant_vol():
    """In the constant-vol synchronous case, the wavelet estimator should
    recover the input lead-lags theta_j on the small scales (j <= 4).

    We use a short Daubechies filter (db4) and a relatively small grid for
    speed. A few units of MAD are tolerated.
    """
    R = [0.5, 0.5, 0.7, 0.5]
    theta = [1, 1, 2, 2]
    daub = DaubechiesFilter("db4")
    sim = BivariateBrownianSimulator(R=R, theta=theta, daub=daub)
    sampler = RegularSampler()
    est = WaveletLeadLagEstimator(delta_N=1.0, j_max=len(theta), grid_half_width=20, daub=daub)

    rng = np.random.default_rng(0)
    times, X1, X2 = sim.simulate(8192, dt=1.0, rng=rng)
    series = sampler.sample(times, X1, X2, rng)
    res = est.fit(series)

    # Loose bound: each scale should be within +/- 2 lags of the truth.
    for j_idx, true in enumerate(theta):
        assert abs(res.theta_hat[j_idx] - true) <= 2
