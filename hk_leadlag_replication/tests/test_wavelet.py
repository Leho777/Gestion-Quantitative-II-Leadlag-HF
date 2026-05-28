"""Tests for the wavelet module."""
from __future__ import annotations

import numpy as np
import pytest

from hk_leadlag.wavelet import (
    DaubechiesFilter,
    autocorrelation_wavelet,
    level_j_wavelet_filter,
)


def test_filter_length_db10():
    daub = DaubechiesFilter("db10")
    assert daub.L == 20  # the paper uses L = 20


def test_qmr_relationship():
    """Scaling and wavelet filters should satisfy g_p = (-1)^(p+1) h_{L-p-1}."""
    daub = DaubechiesFilter("db4")
    L = daub.L
    expected = np.array([((-1) ** (p + 1)) * daub.h[L - p - 1] for p in range(L)])
    np.testing.assert_allclose(daub.g, expected)


def test_level_j_filter_length():
    daub = DaubechiesFilter("db10")
    for j in range(1, 5):
        h_j = level_j_wavelet_filter(daub, j)
        expected_len = (2**j - 1) * (daub.L - 1) + 1
        assert len(h_j) == expected_len


def test_autocorrelation_wavelet_symmetry():
    daub = DaubechiesFilter("db4")
    psi = autocorrelation_wavelet(daub, j=2)
    L_j = (len(psi) + 1) // 2
    # Psi_j(-l) = Psi_j(l).
    for l in range(L_j):
        np.testing.assert_allclose(psi[L_j - 1 + l], psi[L_j - 1 - l])


def test_autocorrelation_wavelet_zero_at_origin_nonneg():
    """Psi_j(0) is the energy of h_j → strictly positive."""
    daub = DaubechiesFilter("db10")
    psi = autocorrelation_wavelet(daub, j=3)
    L_j = (len(psi) + 1) // 2
    assert psi[L_j - 1] > 0
