"""Build the cross-venue panel comparing all 6 full-month runs (avril 2026).

Output: outputs/comparison_panel_full_month.png + prez/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREZ_FIGURES = ROOT.parent / "prez" / "figures"

RUNS = [
    ("2026-04_full_month_btc_binance_vs_kraken",
        "BTC: Binance vs Kraken",       "tab:red"),
    ("2026-04_full_month_eth_binance_vs_kraken",
        "ETH: Binance vs Kraken",       "tab:orange"),
    ("2026-04_full_month_btc_usdt_perp_vs_coin_perp",
        "BTC: USDT-M perp vs COIN-M perp", "tab:purple"),
    ("2026-04_full_month_btc_binance_vs_bybit",
        "BTC: Binance vs Bybit",        "tab:blue"),
    ("2026-04_full_month_btc_spot_vs_perp",
        "BTC: spot vs USDT-M perp",     "tab:green"),
    ("2026-04_full_month_eth_spot_vs_perp",
        "ETH: spot vs USDT-M perp",     "tab:olive"),
]


def main() -> int:
    fig, ax = plt.subplots(figsize=(11, 6))
    for run_name, label, color in RUNS:
        path = ROOT / "outputs" / run_name / "summary.csv"
        if not path.exists():
            print(f"  [warn] missing {path}")
            continue
        df = pd.read_csv(path)
        ax.plot(df["j"], df["theta_hat_s"],
                marker="o", lw=2, color=color, label=label)

    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.set_xlabel("scale j (period $\\propto 2^j \\cdot \\Delta_N$)")
    ax.set_ylabel(r"$\hat\theta_j$ (s)")
    ax.set_title("Cross-venue / cross-product lead-lag, 30 jours (avril 2026)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out1 = ROOT / "outputs" / "comparison_panel_full_month.png"
    out2 = PREZ_FIGURES / "comparison_panel_full_month.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")
    print(f"Saved: {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
