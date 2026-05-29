"""Sign-convention audit for HY and HK estimators.

Critical question: when theta_hat > 0, does it mean
  (A) series 1 leads series 2  (X^1 moves first, X^2 reacts)
or
  (B) series 2 leads series 1  ?

We construct a synthetic case where series 2 is a delayed copy of series 1
with KNOWN lag d > 0 (i.e. X^2 follows X^1 with delay d). We then check the
sign of theta_hat that the estimator returns.

If theta_hat ≈ +d → convention (A): theta>0 means series 1 leads.
If theta_hat ≈ -d → convention (B): theta>0 means series 2 leads.

Run::

    python scripts/audit_sign_convention.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hk_leadlag.base import NonSyncSeries
from hk_leadlag.estimators import HayashiYoshidaEstimator, WaveletLeadLagEstimator
from hk_leadlag.wavelet import DaubechiesFilter


def build_lagged_pair(n: int, lag: float, rho: float, seed: int = 0):
    """Build two log-price series where series 2 is a delayed copy of series 1.

    Construction:
      X^1_t = sum_{k<=t} z_k   (random walk)
      X^2_t = X^1_{t - d}      (exactly delayed, but with iid noise added)
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    # log-price as cumulative sum
    x1 = np.cumsum(z)
    # times of series 1: regular every 1 second
    t1 = np.arange(n, dtype=float)
    # series 2 = series 1 shifted by +lag (i.e. series 2 is observed at later times)
    # so x2(t + lag) = x1(t)
    # equivalently, observed at times t1 + lag, with values x1
    t2 = t1 + lag
    x2 = x1.copy()
    # Add a tiny independent noise to series 2 so it's not exactly identical
    x2 = x2 + rng.standard_normal(n) * 1e-3
    return NonSyncSeries(times1=t1, prices1=x1, times2=t2, prices2=x2,
                         label=f"S2 = S1(t-{lag})")


def main() -> int:
    print("=" * 72)
    print(" Sign convention audit for HY / HK estimators ")
    print("=" * 72)

    LAG = 3.0  # series 2 = series 1 shifted by +3 seconds
    series = build_lagged_pair(n=5000, lag=LAG, rho=1.0)
    print(f"\nConstruction: series 2 observed at t1 + {LAG}s,")
    print("              series 2 values = series 1 values + tiny noise")
    print("=> series 1 is the LEADER, series 2 is the FOLLOWER (lag = +3s)")
    print()

    # --- HY estimator ---
    grid = np.arange(-10, 11, dtype=float)
    hy = HayashiYoshidaEstimator(grid=grid)
    r_hy = hy.fit(series)
    print(f"[HY] theta_hat = {r_hy.theta_hat[0]:+.2f}  (grid {grid[0]:+.0f} to {grid[-1]:+.0f})")

    # --- HK estimator (low j_max so we get a single dominant scale) ---
    est = WaveletLeadLagEstimator(
        delta_N=1.0, j_max=3, grid_half_width=10,
        daub=DaubechiesFilter("db4"),
    )
    r_hk = est.fit(series)
    print(f"[HK] theta_hat per scale: {dict(zip(r_hk.levels.tolist(), r_hk.theta_hat.tolist()))}")

    # --- Verdict ---
    median_hk = float(np.median(r_hk.theta_hat))
    print()
    print("Expected lag value: +%.2f (series 1 leads series 2)" % LAG)
    print(f"HY  estimate     : {r_hy.theta_hat[0]:+.2f}")
    print(f"HK  median       : {median_hk:+.2f}")
    print()
    if r_hy.theta_hat[0] > 0:
        print("VERDICT (HY): theta > 0  =>  series 1 LEADS series 2 [convention A]")
    elif r_hy.theta_hat[0] < 0:
        print("VERDICT (HY): theta < 0 when series 1 leads => convention B (sign INVERTED)")
    else:
        print("VERDICT (HY): theta = 0, inconclusive")

    if median_hk > 0:
        print("VERDICT (HK): theta > 0  =>  series 1 LEADS series 2 [convention A]")
    elif median_hk < 0:
        print("VERDICT (HK): theta < 0 when series 1 leads => convention B (sign INVERTED)")
    else:
        print("VERDICT (HK): theta = 0, inconclusive")

    print()
    print("INTERPRETATION RULE FOR THIS CODEBASE:")
    if r_hy.theta_hat[0] > 0 and median_hk > 0:
        print("  theta_j > 0  <=>  sym1 leads sym2")
        print("  In our BTC/ETH run: theta>0 at j=4..8 means BTC (sym1) leads ETH (sym2).")
    elif r_hy.theta_hat[0] < 0 and median_hk < 0:
        print("  theta_j < 0  <=>  sym1 leads sym2  (SIGN INVERTED FROM INTUITION)")
        print("  In our BTC/ETH run: theta>0 at j=4..8 means ETH (sym2) leads BTC (sym1).")
    else:
        print("  Mixed signs between HY and HK - implementation bug, needs investigation.")
    return 0


def audit_both_directions() -> None:
    """Triple-check: also test the REVERSE construction (series 2 LEADS series 1)."""
    print()
    print("=" * 72)
    print(" Reverse-direction check: series 2 leads series 1 by +3s ")
    print("=" * 72)
    rng = np.random.default_rng(7)
    n = 5000
    LAG = 3.0
    z = rng.standard_normal(n)
    base = np.cumsum(z)
    # Series 1 lags by +3: observed at t1+3, values = base
    t2 = np.arange(n, dtype=float)
    x2 = base.copy()
    t1 = t2 + LAG
    x1 = base.copy() + rng.standard_normal(n) * 1e-3
    series = NonSyncSeries(times1=t1, prices1=x1, times2=t2, prices2=x2,
                           label=f"S1 = S2(t-{LAG})")
    grid = np.arange(-10, 11, dtype=float)
    hy = HayashiYoshidaEstimator(grid=grid)
    r_hy = hy.fit(series)
    print(f"[HY] theta_hat = {r_hy.theta_hat[0]:+.2f}  (expected: opposite sign of previous case)")
    if r_hy.theta_hat[0] > 0:
        print("CONFIRMED CONVENTION: theta > 0  <=>  sym2 (the leader) is ahead of sym1")
        print("  Equivalently: theta_j > 0  <=>  sym2 LEADS sym1")
        print("  Equivalently: theta_j < 0  <=>  sym1 LEADS sym2")
    else:
        print("UNEXPECTED - inconsistent with first test.")


if __name__ == "__main__":
    main()
    audit_both_directions()
    sys.exit(0)
