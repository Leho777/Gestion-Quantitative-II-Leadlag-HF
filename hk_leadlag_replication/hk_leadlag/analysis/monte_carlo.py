"""Monte Carlo runner: replicate Tables 2-3 of Hayashi & Koike (2020)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from hk_leadlag.base import BaseLeadLagEstimator, BaseSimulator, BaseSampler


@dataclass(slots=True)
class MonteCarloResults:
    """Container of MC outputs: shape (n_paths, j_max)."""

    estimates: np.ndarray
    levels: np.ndarray
    true_theta: np.ndarray
    estimator_name: str
    metadata: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        med = np.median(self.estimates, axis=0)
        mad = np.median(np.abs(self.estimates - med), axis=0)
        bias = med - self.true_theta
        return pd.DataFrame(
            {
                "level_j": self.levels,
                "true": self.true_theta,
                "median": med,
                "MAD": mad,
                "bias": bias,
            }
        )


@dataclass(slots=True)
class MonteCarloRunner:
    """Run an estimator over many simulated paths.

    The simulator produces (times, X1, X2) on a fine grid, the sampler turns
    that into a non-synchronous NonSyncSeries, and the estimator is refit on
    each of the n_paths replicates.
    """

    simulator: BaseSimulator
    sampler: BaseSampler
    estimator: BaseLeadLagEstimator
    n_paths: int = 1000
    n_steps: int = 30_000
    dt: float = 1.0
    seed: int | None = 42

    def run(self, true_theta: Sequence[float], show_progress: bool = True) -> MonteCarloResults:
        rng_master = np.random.default_rng(self.seed)
        seeds = rng_master.integers(0, 2**31 - 1, size=self.n_paths)
        # Probe one path to learn the levels.
        first_rng = np.random.default_rng(int(seeds[0]))
        times, X1, X2 = self.simulator.simulate(self.n_steps, self.dt, first_rng)
        series = self.sampler.sample(times, X1, X2, first_rng)
        result0 = self.estimator.fit(series)
        j_max = len(result0.levels)
        estimates = np.empty((self.n_paths, j_max))
        estimates[0] = result0.theta_hat

        iterator = range(1, self.n_paths)
        if show_progress:
            iterator = tqdm(iterator, desc=self.estimator.name, leave=False)

        for i in iterator:
            rng = np.random.default_rng(int(seeds[i]))
            times, X1, X2 = self.simulator.simulate(self.n_steps, self.dt, rng)
            series = self.sampler.sample(times, X1, X2, rng)
            res = self.estimator.fit(series)
            estimates[i] = res.theta_hat
        return MonteCarloResults(
            estimates=estimates,
            levels=result0.levels,
            true_theta=np.asarray(true_theta, dtype=float),
            estimator_name=self.estimator.name,
            metadata={"n_paths": self.n_paths, "n_steps": self.n_steps},
        )
