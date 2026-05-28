"""Heston stochastic-volatility modulation of a Brownian driver (HK20 Scenario 2).

Integrate the Brownian ``B^ell`` against a Heston ``sigma^ell_t`` to get
log-prices ``X^ell_t = int_0^t sigma_s dB_s``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hk_leadlag.base import BaseSimulator
from hk_leadlag.simulation.brownian import BivariateBrownianSimulator


@dataclass(slots=True)
class HestonVolatilitySimulator:
    """Simulate a CIR variance process v_t and return sigma_t = sqrt(v_t).

    SDE:  dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) (rho dB_t + sqrt(1-rho^2) dW_t).
    """

    kappa: float = 5.0
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.5

    def simulate(
        self,
        n_steps: int,
        dt: float,
        rng: np.random.Generator,
        dB: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simulate sigma on the grid t_k = k dt (length n_steps + 1)."""
        v = np.empty(n_steps + 1)
        # Start v from the stationary (Gamma) distribution of the CIR process.
        shape = 2 * self.kappa * self.theta / (self.xi ** 2)
        scale = (self.xi ** 2) / (2 * self.kappa)
        v[0] = rng.gamma(shape=shape, scale=scale)
        if dB is None:
            dB = rng.standard_normal(n_steps) * np.sqrt(dt)
        dW = rng.standard_normal(n_steps) * np.sqrt(dt)
        for k in range(n_steps):
            v_prev = max(v[k], 0.0)
            dv = self.kappa * (self.theta - v_prev) * dt + self.xi * np.sqrt(v_prev) * (
                self.rho * dB[k] + np.sqrt(max(1 - self.rho ** 2, 0.0)) * dW[k]
            )
            v[k + 1] = max(v_prev + dv, 0.0)
        return np.sqrt(v)


@dataclass(slots=True)
class HestonModulatedSimulator(BaseSimulator):
    """Wrap a bivariate Brownian simulator and modulate by Heston volatilities."""

    brownian: BivariateBrownianSimulator
    heston1: HestonVolatilitySimulator
    heston2: HestonVolatilitySimulator

    def simulate(
        self,
        n_steps: int,
        dt: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        incr = self.brownian.simulate_increments(n_steps, rng)
        # Heston vols share the Brownian leverage term but have their own W.
        sigma1 = self.heston1.simulate(n_steps, dt, rng, dB=incr[0] * np.sqrt(dt))
        sigma2 = self.heston2.simulate(n_steps, dt, rng, dB=incr[1] * np.sqrt(dt))
        # Left-rectangle approximation of the stochastic integral on the fine grid.
        dx1 = sigma1[:-1] * incr[0] * np.sqrt(dt)
        dx2 = sigma2[:-1] * incr[1] * np.sqrt(dt)
        X1 = np.concatenate([[0.0], np.cumsum(dx1)])
        X2 = np.concatenate([[0.0], np.cumsum(dx2)])
        times = np.arange(n_steps + 1) * dt
        return times, X1, X2
