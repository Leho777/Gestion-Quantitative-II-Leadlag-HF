"""Bivariate Brownian motion with prescribed scale-by-scale lead-lag structure.

Simulates ``B = (B^1, B^2)`` whose increments on the finest grid have the
covariance structure (eq. before Section 5):

    E[Delta_k B^1 Delta_{k+l} B^2]
        approx (Delta_N / 2^{N-J+1}) sum_{j=1}^{J+1} 2^{J-j+1} R_j Psi^LP(2^{J-j+1}(l Delta_N - theta_j))

Psi^LP is approximated by its compactly-supported counterpart (Daubechies
autocorrelation wavelet; see :mod:`hk_leadlag.wavelet`). The bivariate stationary
Gaussian sequence is then drawn by circulant embedding (Davies & Harte 1987 /
Dietrich & Newsam 1997, multivariate extension by Chan & Wood 1999).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from hk_leadlag.base import BaseSimulator
from hk_leadlag.wavelet import DaubechiesFilter, autocorrelation_wavelet


@dataclass(slots=True)
class BivariateBrownianSimulator(BaseSimulator):
    """Generate the increments of (B^1, B^2) on a regular grid of size n.

    R holds the cross-spectral correlations (R_1, ..., R_J), zero outside that
    range; theta holds the per-scale lead-lags (in units of Delta_N); daub is
    the Daubechies filter backing the LP-wavelet approximation.
    """

    R: Sequence[float]
    theta: Sequence[int]
    daub: DaubechiesFilter = field(default_factory=lambda: DaubechiesFilter("db10"))

    def __post_init__(self) -> None:
        if len(self.R) != len(self.theta):
            raise ValueError("R and theta must have the same length")

    def cross_covariance(self, n: int) -> np.ndarray:
        """Cross-covariance c(l) = E[Delta_0 B^1 Delta_l B^2] for |l| < n.

        It sums shifted autocorrelation wavelets over scales (the diagonal
        terms c11 = c22 are unit, standard Brownian on Delta_N = 1).
        """
        J = len(self.R)
        lags = np.arange(-(n - 1), n)
        c12 = np.zeros_like(lags, dtype=float)
        for j in range(1, J + 1):
            R_j = self.R[j - 1]
            if R_j == 0:
                continue
            theta_j = self.theta[j - 1]
            psi = autocorrelation_wavelet(self.daub, j)
            L_j = (len(psi) + 1) // 2
            psi_lags = np.arange(-(L_j - 1), L_j)
            # psi_j shifted by theta_j and scaled by 2^{-(j-1)} R_j (cf. eq.).
            scale = R_j * 2.0 ** (-(j - 1))
            target_lags = psi_lags + theta_j
            mask = (target_lags >= -(n - 1)) & (target_lags <= n - 1)
            idx = target_lags[mask] + (n - 1)
            c12[idx] += scale * psi[mask]
        return c12

    def covariance_blocks(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (c11, c12, c22) auto/cross covariance functions of length 2n-1."""
        # Standard Brownian increments: c11(l) = 1 at lag 0, else 0.
        c = np.zeros(2 * n - 1)
        c[n - 1] = 1.0
        c12 = self.cross_covariance(n)
        return c.copy(), c12, c.copy()

    def simulate(
        self,
        n_steps: int,
        dt: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (times, X1, X2) on the equidistant grid t_k = k * dt."""
        increments = self.simulate_increments(n_steps, rng)
        times = np.arange(n_steps + 1) * dt
        X1 = np.concatenate([[0.0], np.cumsum(increments[0]) * np.sqrt(dt)])
        X2 = np.concatenate([[0.0], np.cumsum(increments[1]) * np.sqrt(dt)])
        return times, X1, X2

    def simulate_increments(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Bivariate stationary Gaussian increments, shape (2, n), via circulant embedding."""
        c11, c12, c22 = self.covariance_blocks(n)

        # Embed each scalar covariance into a length-2n circulant spectrum
        # (Wood-Chan), then couple the components in the spectral domain below.
        N = 2 * n
        r11 = _circulant_row(c11, n)
        r22 = _circulant_row(c22, n)
        r12 = _circulant_row_cross(c12, n)

        lambda11 = np.fft.fft(r11).real
        lambda22 = np.fft.fft(r22).real
        lambda12 = np.fft.fft(r12)  # cross spectrum, complex in general

        # Diagonals must stay non-negative despite FFT round-off.
        lambda11 = np.clip(lambda11, 0.0, None)
        lambda22 = np.clip(lambda22, 0.0, None)

        eps = 1e-12
        Z = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))
        out = np.empty((N, 2), dtype=complex)
        for k in range(N):
            S = np.array(
                [[lambda11[k] + eps, lambda12[k]],
                 [np.conj(lambda12[k]), lambda22[k] + eps]],
                dtype=complex,
            )
            # Hermitian square root with eigenvalues clipped at 0 to enforce PSD.
            w, V = np.linalg.eigh(S)
            w = np.clip(w.real, 0.0, None)
            sqrtS = V @ np.diag(np.sqrt(w)) @ V.conj().T
            out[k] = sqrtS @ Z[k]
        time_domain = np.fft.ifft(out, axis=0) * np.sqrt(N)
        incr = time_domain[:n].real.T  # first n samples, shape (2, n)
        return incr


def _circulant_row(cov: np.ndarray, n: int) -> np.ndarray:
    """Wrap a length-(2n-1) symmetric covariance into a length-2n circulant row."""
    c0 = cov[n - 1]  # lag-0 value
    pos = cov[n - 1 :]  # lags 0..n-1
    r = np.zeros(2 * n)
    r[:n] = pos
    r[n + 1 :] = pos[1:][::-1]
    r[n] = c0
    return r


def _circulant_row_cross(cov: np.ndarray, n: int) -> np.ndarray:
    """Like :func:`_circulant_row` but for a possibly asymmetric cross-covariance."""
    pos = cov[n - 1 :]  # lags 0..n-1
    neg = cov[: n - 1]  # lags -(n-1)..-1
    r = np.zeros(2 * n)
    r[:n] = pos
    r[n + 1 :] = neg[::-1]
    r[n] = pos[0]
    return r
