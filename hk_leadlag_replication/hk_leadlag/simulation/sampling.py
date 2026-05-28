"""Sampling schemes that produce non-synchronous observation times."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hk_leadlag.base import BaseSampler, NonSyncSeries


@dataclass(slots=True)
class RegularSampler(BaseSampler):
    """Equidistant, synchronous sampling - used as a sanity check."""

    label: str = "regular"

    def sample(
        self,
        times: np.ndarray,
        X1: np.ndarray,
        X2: np.ndarray,
        rng: np.random.Generator,
    ) -> NonSyncSeries:
        return NonSyncSeries(
            times1=times.copy(),
            prices1=X1.copy(),
            times2=times.copy(),
            prices2=X2.copy(),
            label=self.label,
        )


@dataclass(slots=True)
class LoMacKinlaySampler(BaseSampler):
    """Lo & MacKinlay (1990) Bernoulli thinning of a regular grid.

    Each observation on the fine grid ``times[i]`` is *kept* for series
    ``ell`` independently with probability ``1 - p_ell`` and *dropped* with
    probability ``p_ell``.
    """

    p1: float
    p2: float
    label: str = "lo_mackinlay"

    def __post_init__(self) -> None:
        if not (0 <= self.p1 < 1) or not (0 <= self.p2 < 1):
            raise ValueError("p1, p2 must be in [0, 1)")

    def sample(
        self,
        times: np.ndarray,
        X1: np.ndarray,
        X2: np.ndarray,
        rng: np.random.Generator,
    ) -> NonSyncSeries:
        keep1 = rng.random(len(times)) >= self.p1
        keep2 = rng.random(len(times)) >= self.p2
        # Pin the first and last point so both series span the full window.
        keep1[0] = keep1[-1] = True
        keep2[0] = keep2[-1] = True
        return NonSyncSeries(
            times1=times[keep1],
            prices1=X1[keep1],
            times2=times[keep2],
            prices2=X2[keep2],
            label=f"{self.label}(p1={self.p1}, p2={self.p2})",
        )
