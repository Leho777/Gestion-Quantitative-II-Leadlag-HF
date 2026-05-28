"""Compute bootstrap CI for theta_hat_j on an existing run.

Reuses the artifact folder of a previous empirical run and adds a
``bootstrap.json`` + ``bootstrap_boxplot.png`` artifact.

Usage::

    python scripts/run_bootstrap.py \\
        --run 2026-04-13to19_btc_eth_spot_full168h \\
        --hours-cap 24 \\
        --B 100 \\
        --block-s 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hk_leadlag.analysis.artifacts import ArtifactStore, timed
from hk_leadlag.analysis.bootstrap import BlockBootstrapCI
from hk_leadlag.analysis.multiple_testing import benjamini_hochberg, step_down_pvalues
from hk_leadlag.base import NonSyncSeries
from hk_leadlag.estimators import WaveletLeadLagEstimator
from hk_leadlag.wavelet import DaubechiesFilter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Name of an existing outputs/<run>/ folder.")
    p.add_argument("--data", default="data/processed/btc_eth_2026_04_13_to_19.npz",
                   help="Path to the .npz data file used by the run.")
    p.add_argument("--hours-cap", type=float, default=24.0, help="Restrict to first H hours for speed.")
    p.add_argument("--B", type=int, default=100, help="Number of bootstrap replicates.")
    p.add_argument("--block-s", type=float, default=300.0, help="Block length in seconds (5min default).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = ArtifactStore(name=args.run)
    if not store.exists("config.json"):
        print(f"Run {args.run} not found under outputs/. Aborting.", file=sys.stderr)
        return 1

    config = store.read_json("config.json")
    print(f"Loaded config from {store.path('config.json')}:")
    print(f"  delta_N={config['delta_N']}, jmax={config['jmax']}, "
          f"grid_half_width={config['grid_half_width']}, wavelet={config['wavelet']}")

    # Load the saved data.
    d = np.load(args.data)
    mask1 = d['t1'] - d['t1'][0] < args.hours_cap * 3600
    mask2 = d['t2'] - d['t2'][0] < args.hours_cap * 3600
    series = NonSyncSeries(
        times1=d['t1'][mask1], prices1=d['p1'][mask1],
        times2=d['t2'][mask2], prices2=d['p2'][mask2],
        label=f"bootstrap subset {args.hours_cap}h",
    )
    print(f"Subset: {series.n1:,} + {series.n2:,} ticks over {args.hours_cap}h")

    # Build the estimator from config.
    est = WaveletLeadLagEstimator(
        delta_N=config["delta_N"],
        j_max=config["jmax"],
        grid_half_width=config["grid_half_width"],
        daub=DaubechiesFilter(config["wavelet"]),
    )

    # Observed estimate on the subset (not the full 168h - to be consistent).
    with timed("observed fit"):
        observed = est.fit(series)
    print("Observed theta_hat:", observed.theta_hat)

    # Bootstrap.
    bs = BlockBootstrapCI(
        estimator=est,
        block_length_s=args.block_s,
        n_resamples=args.B,
        alpha=0.05,
        seed=args.seed,
    )
    with timed(f"bootstrap B={args.B}, L={args.block_s}s"):
        ci = bs.confidence_interval(series, observed=observed)

    # Multi-test corrections (Romano-Wolf + Benjamini-Hochberg).
    # The raw 'p_value' returned by BlockBootstrapCI is the proportion of
    # bootstrap replicates with |theta_boot| >= |theta_obs|; this tends to 1
    # by construction and is NOT a formal test of H0: theta = 0. We
    # supplement it with:
    #   - two_sided_p_raw: 2 * min(P(boot<=0), P(boot>=0)) (a two-sided
    #     percentile-style test against zero), then
    #   - benjamini_hochberg(two_sided_p_raw) for FDR control, and
    #   - step_down_pvalues on the bootstrap distribution itself
    #     (Romano-Wolf with studentization) for FWER-style control.
    boot = ci["theta_boot"]
    obs = ci["theta_obs"]
    finite_rows = ~np.any(np.isnan(boot), axis=1)
    boot_clean = boot[finite_rows]
    two_sided = np.empty(boot.shape[1])
    for k in range(boot.shape[1]):
        col = boot_clean[:, k]
        if len(col) == 0:
            two_sided[k] = float("nan")
        else:
            two_sided[k] = 2.0 * min(np.mean(col <= 0), np.mean(col >= 0))
    two_sided = np.clip(two_sided, 0.0, 1.0)
    bh = benjamini_hochberg(two_sided)
    try:
        rw = step_down_pvalues(boot_clean, obs, studentize=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] step_down_pvalues failed: {exc}")
        rw = np.full_like(two_sided, np.nan)

    # Save
    payload = {
        "levels": ci["levels"].tolist(),
        "theta_obs": ci["theta_obs"].tolist(),
        "lower": ci["lower"].tolist(),
        "upper": ci["upper"].tolist(),
        "std_err": ci["std_err"].tolist(),
        "p_value": ci["p_value"].tolist(),  # legacy / biased - kept for traceability
        "two_sided_p_raw": two_sided.tolist(),
        "benjamini_hochberg_p": bh.tolist(),
        "romano_wolf_p": rw.tolist(),
        "n_ok": ci["n_ok"],
        "n_total": ci["n_total"],
        "block_length_s": ci["block_length_s"],
        "hours_cap": args.hours_cap,
        "seed": args.seed,
    }
    store.write_json("bootstrap.json", payload)
    store.write_pickle("bootstrap_dist.pkl", ci["theta_boot"])

    # Pretty-print table with multi-test columns.
    print("\n=== Bootstrap CI + multi-test summary ===")
    print(f"{'j':>3} | {'theta_obs':>10} | {'95% CI':>22} | {'SE':>7} | "
          f"{'two-sided p':>11} | {'BH adj':>8} | {'RW step':>8} | {'sig':>3}")
    for k, j in enumerate(ci["levels"]):
        sig_p = rw[k] if not np.isnan(rw[k]) else two_sided[k]
        sig = "***" if sig_p < 0.01 else "**" if sig_p < 0.05 else "*" if sig_p < 0.10 else ""
        print(f"{int(j):>3} | {ci['theta_obs'][k]:>+10.3f} | "
              f"[{ci['lower'][k]:>+6.3f}, {ci['upper'][k]:>+6.3f}] | "
              f"{ci['std_err'][k]:>7.3f} | "
              f"{two_sided[k]:>11.3f} | {bh[k]:>8.3f} | {rw[k]:>8.3f} | {sig:>3}")

    # Boxplot
    fig, ax = plt.subplots(figsize=(10, 5))
    valid = ~np.any(np.isnan(ci["theta_boot"]), axis=1)
    boot = ci["theta_boot"][valid]
    ax.boxplot(
        [boot[:, k] for k in range(boot.shape[1])],
        positions=ci["levels"].astype(float),
        widths=0.6, patch_artist=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black"),
        showmeans=True,
    )
    ax.scatter(ci["levels"].astype(float), ci["theta_obs"],
               marker="x", color="crimson", s=100, lw=2, label="observed", zorder=5)
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.set_xlabel("scale j")
    ax.set_ylabel(r"$\hat\theta_j$ (s)")
    ax.set_title(f"Block bootstrap distribution (B={ci['n_ok']}/{ci['n_total']}, L={args.block_s}s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(store.path("bootstrap_boxplot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {store.path('bootstrap.json')}, {store.path('bootstrap_boxplot.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
