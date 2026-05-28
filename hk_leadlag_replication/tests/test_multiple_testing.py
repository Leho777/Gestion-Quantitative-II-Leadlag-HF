"""Tests for multiple-testing corrections."""
from __future__ import annotations

import numpy as np
import pytest

from hk_leadlag.analysis import benjamini_hochberg, step_down_pvalues


def test_benjamini_hochberg_matches_known_example():
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.20]))

    np.testing.assert_allclose(adjusted, [0.04, 0.0533333333, 0.0533333333, 0.20])


def test_benjamini_hochberg_preserves_nan():
    adjusted = benjamini_hochberg(np.array([0.01, np.nan, 0.20]))

    assert np.isnan(adjusted[1])
    np.testing.assert_allclose(adjusted[[0, 2]], [0.02, 0.20])


def test_step_down_pvalues_are_monotone_in_test_strength():
    rng = np.random.default_rng(123)
    boot = rng.normal(0.0, 1.0, size=(500, 3))
    observed = np.array([3.0, 2.0, 0.2])

    adjusted = step_down_pvalues(boot, observed, center=np.zeros(3))

    assert adjusted[0] <= adjusted[1] <= adjusted[2]
    assert np.all((adjusted >= 0.0) & (adjusted <= 1.0))


def test_step_down_pvalues_dominate_raw_bootstrap_pvalues():
    rng = np.random.default_rng(456)
    boot = rng.normal(0.0, 1.0, size=(400, 4))
    observed = np.array([2.8, 1.5, 0.8, 0.1])

    adjusted = step_down_pvalues(boot, observed, center=np.zeros(4), studentize=False)
    raw = np.array([
        (1.0 + np.sum(np.abs(boot[:, k]) >= abs(observed[k]))) / (len(boot) + 1.0)
        for k in range(boot.shape[1])
    ])

    assert np.all(adjusted >= raw)


def test_step_down_pvalues_validates_shapes():
    with pytest.raises(ValueError, match="shape"):
        step_down_pvalues(np.ones(3), np.ones(3))

    with pytest.raises(ValueError, match="disagree"):
        step_down_pvalues(np.ones((5, 2)), np.ones(3))
