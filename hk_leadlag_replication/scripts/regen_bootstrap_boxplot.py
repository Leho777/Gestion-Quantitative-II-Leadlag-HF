"""Regenerate the bootstrap boxplot for a run whose bootstrap script crashed
between writing ``bootstrap_dist.pkl`` / ``bootstrap.json`` and saving the
``bootstrap_boxplot.png`` artifact.

Usage::

    python scripts/regen_bootstrap_boxplot.py \\
        --run 2026-04-13to19_btc_binance_vs_kraken
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

from hk_leadlag.analysis.artifacts import ArtifactStore


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()

    store = ArtifactStore(name=args.run)
    bs = store.read_json("bootstrap.json")
    boot = store.read_pickle("bootstrap_dist.pkl")

    levels = np.asarray(bs["levels"])
    theta_obs = np.asarray(bs["theta_obs"])
    block_s = bs["block_length_s"]
    n_ok = bs["n_ok"]
    n_total = bs["n_total"]

    valid = ~np.any(np.isnan(boot), axis=1)
    boot = boot[valid]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(
        [boot[:, k] for k in range(boot.shape[1])],
        positions=levels.astype(float),
        widths=0.6, patch_artist=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black"),
        showmeans=True,
    )
    ax.scatter(levels.astype(float), theta_obs,
               marker="x", color="crimson", s=100, lw=2, label="observed", zorder=5)
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.set_xlabel("scale j")
    ax.set_ylabel(r"$\hat\theta_j$ (s)")
    ax.set_title(f"Block bootstrap distribution (B={n_ok}/{n_total}, L={block_s}s)")
    ax.legend()
    fig.tight_layout()
    out_png = store.path("bootstrap_boxplot.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
