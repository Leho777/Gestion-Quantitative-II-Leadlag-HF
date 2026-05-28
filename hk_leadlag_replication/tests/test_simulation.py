"""Sanity tests for the simulator + sampler stack."""
from __future__ import annotations

import numpy as np

from hk_leadlag.simulation import (
    BivariateBrownianSimulator,
    LoMacKinlaySampler,
    RegularSampler,
)


def test_brownian_increments_shape():
    sim = BivariateBrownianSimulator(R=[0.5, 0.5], theta=[1, 2])
    rng = np.random.default_rng(0)
    incr = sim.simulate_increments(2048, rng)
    assert incr.shape == (2, 2048)
    assert np.all(np.isfinite(incr))


def test_lo_mackinlay_drops_observations():
    sim = BivariateBrownianSimulator(R=[0.3], theta=[1])
    rng = np.random.default_rng(1)
    times, X1, X2 = sim.simulate(1000, dt=1.0, rng=rng)
    sampler = LoMacKinlaySampler(p1=0.5, p2=0.5)
    series = sampler.sample(times, X1, X2, rng)
    # With 50% drop on each side we expect roughly half the observations.
    assert series.n1 < len(times)
    assert series.n2 < len(times)
    assert series.n1 > 0.3 * len(times)
    assert series.n2 > 0.3 * len(times)


def test_regular_sampler_keeps_everything():
    sim = BivariateBrownianSimulator(R=[0.3], theta=[1])
    rng = np.random.default_rng(2)
    times, X1, X2 = sim.simulate(500, dt=1.0, rng=rng)
    series = RegularSampler().sample(times, X1, X2, rng)
    assert series.n1 == len(times)
    assert series.n2 == len(times)
