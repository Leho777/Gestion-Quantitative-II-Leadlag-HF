"""Controlled Monte Carlo diagnostics for the HK estimator.

This script is intentionally narrower than the full Hayashi-Koike Tables 2-3.
It checks whether the local implementation recovers known scale-by-scale
lead-lags under:

* synchronous sampling,
* Lo-MacKinlay non-synchronous sampling,
* Heston stochastic-volatility modulation,
* a previous-tick interpolated baseline ("WCCF-like", not the exact WCCF
  estimator from the earlier HK paper).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hk_leadlag.base import NonSyncSeries
from hk_leadlag.estimators import WaveletLeadLagEstimator
from hk_leadlag.simulation import (
    BivariateBrownianSimulator,
    HestonModulatedSimulator,
    HestonVolatilitySimulator,
    LoMacKinlaySampler,
    RegularSampler,
)
from hk_leadlag.wavelet import DaubechiesFilter


@dataclass(frozen=True)
class Scenario:
    name: str
    vol: str
    p1: float
    p2: float
    include_interpolated: bool = True


def previous_tick_grid(series: NonSyncSeries, dt: float) -> NonSyncSeries:
    """Previous-tick interpolation on a regular grid.

    This is a WCCF-like diagnostic baseline: it deliberately reintroduces a
    regular grid before applying the same wavelet contrast machinery.
    """
    horizon = min(float(series.times1[-1]), float(series.times2[-1]))
    grid = np.arange(0.0, horizon + 0.5 * dt, dt)
    i1 = np.searchsorted(series.times1, grid, side="right") - 1
    i2 = np.searchsorted(series.times2, grid, side="right") - 1
    i1 = np.clip(i1, 0, len(series.prices1) - 1)
    i2 = np.clip(i2, 0, len(series.prices2) - 1)
    return NonSyncSeries(
        times1=grid,
        prices1=series.prices1[i1],
        times2=grid,
        prices2=series.prices2[i2],
        label=f"previous_tick_grid(dt={dt})",
    )


def build_simulator(vol: str, true_r: list[float], true_theta: list[int], daub: DaubechiesFilter):
    brownian = BivariateBrownianSimulator(R=true_r, theta=true_theta, daub=daub)
    if vol == "constant":
        return brownian
    if vol == "heston":
        return HestonModulatedSimulator(
            brownian=brownian,
            heston1=HestonVolatilitySimulator(),
            heston2=HestonVolatilitySimulator(),
        )
    raise ValueError(f"unknown volatility scenario: {vol}")


def summarize(paths: pd.DataFrame, true_theta: list[int]) -> pd.DataFrame:
    rows = []
    for (scenario, estimator, j), sub in paths.groupby(["scenario", "estimator", "j"]):
        vals = sub["theta_hat"].to_numpy(dtype=float)
        truth = float(true_theta[int(j) - 1])
        med = float(np.median(vals))
        rows.append(
            {
                "scenario": scenario,
                "estimator": estimator,
                "j": int(j),
                "true_theta": truth,
                "median": med,
                "MAD": float(np.median(np.abs(vals - med))),
                "bias": med - truth,
                "mean_abs_error": float(np.mean(np.abs(vals - truth))),
                "within_1_delta": float(np.mean(np.abs(vals - truth) <= 1.0)),
                "within_2_delta": float(np.mean(np.abs(vals - truth) <= 2.0)),
                "correct_sign": float(np.mean(np.sign(vals) == np.sign(truth))),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario", "estimator", "j"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-paths", type=int, default=30)
    parser.add_argument("--n-steps", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--output-dir", default="outputs/monte_carlo_hk_diagnostic")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    true_r = [0.3, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5]
    true_theta = [1, 1, 2, 2, 3, 5, 7, 10]
    daub = DaubechiesFilter("db10")
    estimator = WaveletLeadLagEstimator(
        delta_N=1.0,
        j_max=len(true_theta),
        grid_half_width=100,
        daub=daub,
    )
    scenarios = [
        Scenario("constant_sync", "constant", 0.0, 0.0, include_interpolated=False),
        Scenario("constant_lm_moderate", "constant", 0.25, 0.50),
        Scenario("constant_lm_high", "constant", 0.25, 0.75),
        Scenario("heston_lm_high", "heston", 0.25, 0.75),
    ]

    rng_master = np.random.default_rng(args.seed)
    rows = []
    started = time.time()
    for scenario in scenarios:
        simulator = build_simulator(scenario.vol, true_r, true_theta, daub)
        sampler = RegularSampler() if scenario.p1 == scenario.p2 == 0 else LoMacKinlaySampler(scenario.p1, scenario.p2)
        seeds = rng_master.integers(0, 2**31 - 1, size=args.n_paths)
        print(f"[{scenario.name}] {args.n_paths} paths, n_steps={args.n_steps}", flush=True)
        for path_idx, seed in enumerate(seeds, start=1):
            rng = np.random.default_rng(int(seed))
            times, x1, x2 = simulator.simulate(args.n_steps, 1.0, rng)
            series = sampler.sample(times, x1, x2, rng)

            estimates = [("HK_HY_non_sync", series)]
            if scenario.include_interpolated:
                estimates.append(("previous_tick_interpolated", previous_tick_grid(series, dt=1.0)))

            for estimator_name, estimator_series in estimates:
                result = estimator.fit(estimator_series)
                for j, theta_hat in zip(result.levels, result.theta_hat):
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "vol": scenario.vol,
                            "p1": scenario.p1,
                            "p2": scenario.p2,
                            "path": path_idx,
                            "seed": int(seed),
                            "estimator": estimator_name,
                            "j": int(j),
                            "theta_hat": float(theta_hat),
                            "n1": int(estimator_series.n1),
                            "n2": int(estimator_series.n2),
                        }
                    )
            if path_idx % max(1, args.n_paths // 5) == 0:
                print(f"  path {path_idx}/{args.n_paths}", flush=True)

    paths = pd.DataFrame(rows)
    summary = summarize(paths, true_theta)
    paths.to_csv(out_dir / "paths.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "n_paths": args.n_paths,
                "n_steps": args.n_steps,
                "seed": args.seed,
                "true_r": true_r,
                "true_theta": true_theta,
                "wavelet": daub.name,
                "scenarios": [asdict(s) for s in scenarios],
                "elapsed_seconds": time.time() - started,
            },
            fh,
            indent=2,
        )
    print(f"Saved {out_dir / 'summary.csv'}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
