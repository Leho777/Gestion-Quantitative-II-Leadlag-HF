"""Daubechies wavelet filters and autocorrelation wavelets Psi_j(l).

Construction from Section 3 of Hayashi & Koike (2020):

* Daubechies filter (h_p) of length L (via PyWavelets).
* Quadrature-mirror scaling filter g_p = (-1)^(p+1) h_{L-p-1}.
* Recursive level-j wavelet filter h_{j,p} = sum_q g_{p-2q} h_{j-1,q}.
* Autocorrelation wavelet Psi_j(l) = sum_p h_{j,p} h_{j,p+|l|} (Nason et al. 2000).

Psi_j is the discrete approximation of the continuous Littlewood-Paley wavelet
that localises the cross-covariance at scale j.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pywt


@dataclass(frozen=True)
class DaubechiesFilter:
    """Daubechies wavelet of even length L.

    PyWavelets' decomposition high-pass filter ``dec_hi`` is the wavelet filter
    (h_p) up to a sign convention. The scaling filter is re-derived via the QMR
    relationship rather than ``dec_lo`` so the convention matches Hayashi & Koike.
    """

    name: str = "db10"

    def __post_init__(self) -> None:
        wav = pywt.Wavelet(self.name)
        if len(wav.dec_hi) % 2:
            raise ValueError(f"Wavelet {self.name!r} has odd length, expected even L.")
        h = np.asarray(wav.dec_hi, dtype=float)
        L = len(h)
        g = np.array([((-1) ** (p + 1)) * h[L - p - 1] for p in range(L)], dtype=float)
        # frozen=True forbids direct assignment, hence object.__setattr__.
        object.__setattr__(self, "_wav", wav)
        object.__setattr__(self, "_h", h)
        object.__setattr__(self, "_g", g)
        object.__setattr__(self, "_L", L)

    @property
    def L(self) -> int:
        return self._L  # type: ignore[attr-defined]

    @property
    def h(self) -> np.ndarray:
        """Wavelet (high-pass) filter, length L."""
        return self._h  # type: ignore[attr-defined]

    @property
    def g(self) -> np.ndarray:
        """Scaling (low-pass) filter via QMR: g_p = (-1)^(p+1) h_{L-p-1}."""
        return self._g  # type: ignore[attr-defined]


def level_j_wavelet_filter(daub: DaubechiesFilter, j: int) -> np.ndarray:
    """Return the level-j wavelet filter (h_{j,p}) of length L_j = (2^j - 1)(L-1) + 1.

    Constructed recursively as in eq. (just before Theorem 1) of HK20:

        h_{1,p} = h_p,     p = 0,...,L-1
        h_{j,p} = sum_q g_{p-2q} h_{j-1,q},  p = 0,...,L_j - 1

    with g_p = 0 outside {0,...,L-1}.
    """
    if j < 1:
        raise ValueError("j must be >= 1")
    h = daub.h
    g = daub.g
    L = daub.L
    if j == 1:
        return h.copy()
    h_prev = level_j_wavelet_filter(daub, j - 1)
    L_prev = len(h_prev)
    L_j = (2**j - 1) * (L - 1) + 1
    h_j = np.zeros(L_j)
    for p in range(L_j):
        s = 0.0
        for q in range(L_prev):
            idx = p - 2 * q
            if 0 <= idx < L:
                s += g[idx] * h_prev[q]
        h_j[p] = s
    return h_j


def autocorrelation_wavelet(daub: DaubechiesFilter, j: int) -> np.ndarray:
    """Return Psi_j(l) = sum_p h_{j,p} h_{j,p+|l|} for l = -(L_j-1), ..., (L_j-1).

    Discrete approximation of phi^LP(2^{N-j+1} l Delta_N) in Hayashi & Koike,
    localising the cross-covariance around scale j. The output is symmetric.
    """
    h_j = level_j_wavelet_filter(daub, j)
    L_j = len(h_j)
    # Full autocorrelation of h_j: length 2 L_j - 1, lags -(L_j-1)..(L_j-1).
    psi = np.correlate(h_j, h_j, mode="full")
    expected = 2 * L_j - 1
    if len(psi) != expected:
        raise RuntimeError(f"unexpected length {len(psi)} (expected {expected})")
    return psi


def transfer_function(daub: DaubechiesFilter, j: int, n_freq: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Return (omega, H_{j,L}(omega)) on a uniform [-pi, pi] grid.

    For diagnostic plots: H_{j,L} should approximate 2^j 1_{Lambda_j}.
    """
    psi = autocorrelation_wavelet(daub, j)
    L_j = (len(psi) + 1) // 2
    lags = np.arange(-(L_j - 1), L_j)
    omega = np.linspace(-np.pi, np.pi, n_freq)
    # H_{j,L}(omega) = sum_l Psi_j(l) e^{-i l omega}
    H = np.array([np.sum(psi * np.exp(-1j * lags * w)) for w in omega])
    return omega, H.real  # imaginary part is zero by symmetry
