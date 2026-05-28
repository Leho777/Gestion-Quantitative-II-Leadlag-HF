"""Block bootstrap CI for theta_hat_j via tick-level resampling.

A naive iid bootstrap on ticks destroys their temporal dependence, so this
uses a moving-block bootstrap (Kunsch 1989, Politis-Romano 1994) that
preserves local dynamics. Each replicate draws random blocks, concatenates
them into a synthetic series, and refits the estimator. The resulting
distribution over theta_hat_j gives percentile CIs, a standard error, and
(legacy) p-values.

Each replicate is a full fit, so this is slow: on 24h of BTC/ETH with
delta_N=0.05, jmax=6, expect roughly 30s per replicate. Drop B or parallelise
if it matters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from tqdm import tqdm

from hk_leadlag.base import (
    BaseLeadLagEstimator,
    LeadLagResult,
    NonSyncSeries,
)


@dataclass(slots=True)
class BlockBootstrapCI:
    """Moving-block bootstrap for theta_hat_j.

    block_length_s should be much larger than the largest expected theta_j,
    otherwise resampling breaks the lead-lag. alpha sets the (1 - alpha)
    two-sided CI.
    """

    estimator: BaseLeadLagEstimator
    block_length_s: float = 300.0  # 5 min
    n_resamples: int = 100
    alpha: float = 0.05
    seed: int | None = 0

    def confidence_interval(
        self,
        series: NonSyncSeries,
        observed: LeadLagResult | None = None,
        show_progress: bool = True,
    ) -> dict:
        """Return CI bounds and the bootstrap distribution for each scale."""
        rng = np.random.default_rng(self.seed)
        if observed is None:
            observed = self.estimator.fit(series)
        j_max = len(observed.levels)

        t_min = float(min(series.times1[0], series.times2[0]))
        t_max = float(max(series.times1[-1], series.times2[-1]))
        total_T = t_max - t_min
        n_blocks = int(np.floor(total_T / self.block_length_s))
        if n_blocks < 2:
            raise ValueError(
                f"Series too short ({total_T:.1f}s) for block length {self.block_length_s}s."
            )

        boot = np.empty((self.n_resamples, j_max))
        iterator = range(self.n_resamples)
        if show_progress:
            iterator = tqdm(iterator, desc="bootstrap", leave=False)

        for b in iterator:
            synth = _resample_blocks(
                series=series,
                block_length_s=self.block_length_s,
                n_blocks=n_blocks,
                t_min=t_min,
                rng=rng,
            )
            try:
                res_b = self.estimator.fit(synth)
                boot[b] = res_b.theta_hat
            except Exception:  # noqa: BLE001
                boot[b] = np.nan

        ok = ~np.any(np.isnan(boot), axis=1)
        if ok.sum() < 2:
            raise RuntimeError("All bootstrap replicates failed.")
        b_clean = boot[ok]

        lower = np.quantile(b_clean, self.alpha / 2, axis=0)
        upper = np.quantile(b_clean, 1 - self.alpha / 2, axis=0)
        std_err = b_clean.std(axis=0, ddof=1)

        # WARNING (legacy, biased): this p_value is NOT a calibrated test of
        # H0: theta_j = 0. Since the bootstrap is centered on theta_obs (not 0),
        # the proportion |theta_boot| >= |theta_obs| tends to 1 by construction
        # for a consistent estimator. Kept only for backward compatibility with
        # old runs. For calibrated inference use the two-sided percentile p
        # (`two_sided_p_raw` in scripts/run_bootstrap.py) or the Romano-Wolf
        # step-down (`multiple_testing.step_down_pvalues`, controls FWER across
        # the j-levels).
        p_value = np.empty(j_max)
        for k in range(j_max):
            obs = abs(observed.theta_hat[k])
            p_value[k] = float(np.mean(np.abs(b_clean[:, k]) >= obs))

        return {
            "theta_obs": observed.theta_hat,
            "levels": observed.levels,
            "theta_boot": boot,
            "lower": lower,
            "upper": upper,
            "std_err": std_err,
            "p_value": p_value,
            "n_ok": int(ok.sum()),
            "n_total": int(self.n_resamples),
            "block_length_s": self.block_length_s,
        }


def _resample_blocks(
    series: NonSyncSeries,
    block_length_s: float,
    n_blocks: int,
    t_min: float,
    rng: np.random.Generator,
) -> NonSyncSeries:
    """Build a synthetic NonSyncSeries by concatenating random blocks.

    Each block is the slice of (times, prices) in [t_start, t_start + L]. The
    synthetic time axis is rebuilt as 0, L, 2L, ... so blocks stay contiguous.
    """
    t1, p1 = series.times1, series.prices1
    t2, p2 = series.times2, series.prices2
    t_max = float(max(t1[-1], t2[-1]))
    L = block_length_s

    starts = rng.uniform(t_min, t_max - L, size=n_blocks)

    out_t1, out_p1 = [], []
    out_t2, out_p2 = [], []
    new_offset = 0.0
    for s in starts:
        m1 = (t1 >= s) & (t1 < s + L)
        m2 = (t2 >= s) & (t2 < s + L)
        if m1.sum() < 2 or m2.sum() < 2:
            # Block too sparse - skip but advance to maintain timeline.
            new_offset += L
            continue
        tt1 = t1[m1] - s + new_offset
        tt2 = t2[m2] - s + new_offset
        pp1 = p1[m1]
        pp2 = p2[m2]
        # De-trend so successive blocks don't introduce jumps.
        pp1 = pp1 - pp1[0]
        pp2 = pp2 - pp2[0]
        if out_p1:
            pp1 = pp1 + out_p1[-1][-1]
            pp2 = pp2 + out_p2[-1][-1]
        out_t1.append(tt1)
        out_p1.append(pp1)
        out_t2.append(tt2)
        out_p2.append(pp2)
        new_offset += L

    if not out_t1:
        raise RuntimeError("No valid blocks could be assembled.")

    return NonSyncSeries(
        times1=np.concatenate(out_t1),
        prices1=np.concatenate(out_p1),
        times2=np.concatenate(out_t2),
        prices2=np.concatenate(out_p2),
        label="bootstrap synthetic",
    )
