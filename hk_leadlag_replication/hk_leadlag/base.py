"""Abstract base classes and value objects shared across the framework."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NonSyncSeries:
    """A pair of non-synchronously sampled log-price series.

    times1/times2 are sorted observation times; prices1/prices2 the log-prices
    observed at those times. label is optional, e.g. "BTCUSDT@Binance vs Bybit".
    """

    times1: np.ndarray
    prices1: np.ndarray
    times2: np.ndarray
    prices2: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        if len(self.times1) != len(self.prices1):
            raise ValueError("times1 and prices1 must have the same length")
        if len(self.times2) != len(self.prices2):
            raise ValueError("times2 and prices2 must have the same length")
        if len(self.times1) < 2 or len(self.times2) < 2:
            raise ValueError("each series must contain at least two observations")
        if not np.all(np.diff(self.times1) > 0):
            raise ValueError("times1 must be strictly increasing")
        if not np.all(np.diff(self.times2) > 0):
            raise ValueError("times2 must be strictly increasing")

    @property
    def n1(self) -> int:
        return len(self.times1)

    @property
    def n2(self) -> int:
        return len(self.times2)

    @property
    def T(self) -> float:
        """Common observation horizon (max of the two ends)."""
        return float(max(self.times1[-1], self.times2[-1]))

    def returns(self) -> tuple[np.ndarray, np.ndarray]:
        """Tick-by-tick returns (price increments) for both series."""
        return np.diff(self.prices1), np.diff(self.prices2)


@dataclass(frozen=True, slots=True)
class LeadLagResult:
    """Container for the output of a multi-scale lead-lag estimation."""

    theta_hat: np.ndarray  # shape (j_max,) - estimated lead-lags per scale
    levels: np.ndarray  # shape (j_max,) - scale indices j = 1..j_max
    contrast: np.ndarray | None = None  # shape (j_max, |grid|) - |Gamma_hat(tau)|
    grid: np.ndarray | None = None  # search grid in same units as Delta_N
    estimator_name: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "estimator": self.estimator_name,
            "levels": self.levels.tolist(),
            "theta_hat": self.theta_hat.tolist(),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Abstract bases
# ---------------------------------------------------------------------------


class BaseLeadLagEstimator(ABC):
    """Abstract estimator of lead-lag relationships between two HF series."""

    name: str = "base"

    @abstractmethod
    def fit(self, series: NonSyncSeries) -> LeadLagResult:
        """Estimate lead-lag parameters from a non-synchronous price pair."""

    def __call__(self, series: NonSyncSeries) -> LeadLagResult:
        return self.fit(series)


class BaseSimulator(ABC):
    """Abstract simulator producing a continuous-time bivariate process."""

    @abstractmethod
    def simulate(self, n_steps: int, dt: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (times, X1, X2) on a fine equidistant grid."""


class BaseSampler(ABC):
    """Abstract sampling scheme that produces non-synchronous observations."""

    @abstractmethod
    def sample(
        self,
        times: np.ndarray,
        X1: np.ndarray,
        X2: np.ndarray,
        rng: np.random.Generator,
    ) -> NonSyncSeries:
        """Return non-synchronously observed series from a fine grid."""


class BaseDataLoader(ABC):
    """Abstract real-world data loader producing a NonSyncSeries."""

    @abstractmethod
    def load(self) -> NonSyncSeries:
        """Load and pre-process a pair of HF price series."""

    def load_many(self, pairs: Iterable[tuple[str, str]]) -> dict[tuple[str, str], NonSyncSeries]:
        """Convenience: load a batch of pairs."""
        return {pair: self.load() for pair in pairs}
