"""Distribution per-day du lead-lag HK (analogue de la Figure 1 du papier).

HK estime un theta_hat_j par stock x par JOUR, puis etudie la DISTRIBUTION
(histogrammes Fig.1, medianes Table 4). Nos runs mensuels poolent tout le mois
en un seul fit (un point par mois). Ce script comble l'ecart : il fit
theta_hat_j sur chaque jour UTC de la periode, puis agrege la distribution
inter-jours par echelle et trace les histogrammes.

Charge la donnee JOUR PAR JOUR depuis le cache parquet produit par les runs
mensuels (memoire legere ; telecharge si un jour manque). Conventions de signe
identiques aux scripts mensuels :
  - binance_vs_bybit : theta<0 => Binance mene Bybit ; theta>0 => Bybit mene Binance
  - spot_vs_perp     : theta<0 => spot mene perp   ; theta>0 => perp mene spot

Usage::

    python scripts/run_perday_distribution.py --mode binance_vs_bybit \\
        --symbol BTCUSDT --start 2026-01-01 --end 2026-01-31
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from hk_leadlag.base import NonSyncSeries
from hk_leadlag.data.binance import BinanceTradesLoader, _to_unix_seconds, _dedupe
from hk_leadlag.data.bybit import BybitTradesLoader
from hk_leadlag.estimators import WaveletLeadLagEstimator
from hk_leadlag.wavelet import DaubechiesFilter


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def load_binance_day(symbol: str, market: str, day: date) -> tuple[np.ndarray, np.ndarray]:
    loader = BinanceTradesLoader(
        symbol1=symbol, symbol2=symbol,
        start_date=day, end_date=day, market=market, log_prices=True,
    )
    df = loader._load_symbol(symbol)
    t = _to_unix_seconds(df["timestamp_ms"].to_numpy(dtype=float))
    p = np.log(df["price"].to_numpy(dtype=float))
    return _dedupe(t, p)


def load_bybit_day(symbol: str, market: str, day: date) -> tuple[np.ndarray, np.ndarray]:
    loader = BybitTradesLoader(
        symbol1=symbol, symbol2=symbol,
        start_date=day, end_date=day, market1=market, market2=market, log_prices=True,
    )
    return loader._load_side(market, symbol)


def series_for_day(mode: str, symbol: str, day: date):
    if mode == "binance_vs_bybit":
        t1, p1 = load_binance_day(symbol, "spot", day)
        t2, p2 = load_bybit_day(symbol, "spot", day)
        conv = "theta<0: Binance mene Bybit | theta>0: Bybit mene Binance"
    elif mode == "spot_vs_perp":
        t1, p1 = load_binance_day(symbol, "spot", day)
        t2, p2 = load_binance_day(symbol, "futures", day)
        conv = "theta<0: spot mene perp | theta>0: perp mene spot"
    else:
        raise ValueError(f"mode inconnu: {mode}")
    return t1, p1, t2, p2, conv


# --- Kraken : cache par plage entiere (pas par jour). On charge le mois cache UNE
# fois puis on tranche en memoire, pour ne JAMAIS re-paginer (consigne : pas de
# nouvelle pagination Kraken). Kraken est petit (~1M trades/mois), tient en RAM.
KRAKEN_PAIR = {"BTCUSDT": "XBTUSD", "ETHUSDT": "ETHUSD"}


def load_full_binance(symbol: str, market: str, start: date, end: date):
    loader = BinanceTradesLoader(symbol1=symbol, symbol2=symbol,
                                 start_date=start, end_date=end, market=market, log_prices=True)
    df = loader._load_symbol(symbol)
    t = _to_unix_seconds(df["timestamp_ms"].to_numpy(dtype=float))
    p = np.log(df["price"].to_numpy(dtype=float))
    return _dedupe(t, p)


def load_full_kraken(symbol: str, start: date, end: date):
    from hk_leadlag.data.kraken import KrakenTradesLoader
    pair = KRAKEN_PAIR[symbol]
    loader = KrakenTradesLoader(pair1=pair, pair2=pair, start_date=start, end_date=end, log_prices=True)
    return loader._load_side(pair)


def day_bounds(day: date) -> tuple[float, float]:
    from datetime import datetime, timezone
    ds = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()
    return ds, ds + 86400.0


def slice_day(t: np.ndarray, p: np.ndarray, ds: float, de: float):
    lo = int(np.searchsorted(t, ds, side="left"))
    hi = int(np.searchsorted(t, de, side="left"))
    return t[lo:hi], p[lo:hi]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["binance_vs_bybit", "spot_vs_perp", "binance_vs_kraken"])
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", required=True, type=parse_date)
    p.add_argument("--end", required=True, type=parse_date)
    p.add_argument("--delta-N", type=float, default=0.05)
    p.add_argument("--jmax", type=int, default=8)
    p.add_argument("--grid-half", type=int, default=400)
    p.add_argument("--wavelet", default="db10")
    p.add_argument("--min-trades", type=int, default=5000,
                   help="Saute un jour si une des deux series a moins de N trades.")
    p.add_argument("--name", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    _short = {"BTCUSDT": "btc", "ETHUSDT": "eth"}.get(args.symbol, args.symbol.lower())
    name = args.name or f"{args.start:%Y-%m}_{_short}_{args.mode}_perday"
    out = ROOT / "outputs" / name
    out.mkdir(parents=True, exist_ok=True)

    est = WaveletLeadLagEstimator(
        delta_N=args.delta_N, j_max=args.jmax,
        grid_half_width=args.grid_half, daub=DaubechiesFilter(args.wavelet),
    )

    print(f"=== per-day HK: {args.mode} {args.symbol} {args.start}..{args.end} ===", flush=True)

    # Mode Kraken : pre-charge le mois cache UNE fois, puis tranche par jour.
    full = None
    conv = ""
    if args.mode == "binance_vs_kraken":
        print("pre-chargement du mois (cache) ...", flush=True)
        t1f, p1f = load_full_binance(args.symbol, "spot", args.start, args.end)
        t2f, p2f = load_full_kraken(args.symbol, args.start, args.end)
        full = (t1f, p1f, t2f, p2f)
        conv = "theta<0: Binance mene Kraken | theta>0: Kraken mene Binance"
        print(f"  Binance {len(t1f):,} | Kraken {len(t2f):,}", flush=True)

    rows = []
    for day in daterange(args.start, args.end):
        try:
            if full is not None:
                ds, de = day_bounds(day)
                t1, p1 = slice_day(full[0], full[1], ds, de)
                t2, p2 = slice_day(full[2], full[3], ds, de)
            else:
                t1, p1, t2, p2, conv = series_for_day(args.mode, args.symbol, day)
        except Exception as e:  # jour manquant / download KO -> on saute
            print(f"{day} SKIP (load: {type(e).__name__})", flush=True)
            continue
        if len(t1) < args.min_trades or len(t2) < args.min_trades:
            print(f"{day} SKIP (trades {len(t1)}/{len(t2)} < {args.min_trades})", flush=True)
            continue
        series = NonSyncSeries(times1=t1, prices1=p1, times2=t2, prices2=p2,
                               label=f"{args.symbol} {args.mode} {day}")
        res = est.fit(series)
        row = {"day": day.isoformat(), "n1": int(len(t1)), "n2": int(len(t2))}
        for j, th in zip(res.levels.tolist(), res.theta_hat.tolist()):
            row[f"theta_j{int(j)}"] = float(th)
        rows.append(row)
        thetas = " ".join(f"{x:+.2f}" for x in res.theta_hat.tolist())
        print(f"{day} OK  n={len(t1)}/{len(t2)}  theta=[{thetas}]", flush=True)

    if not rows:
        print("Aucun jour exploitable.", flush=True)
        return 1

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_day_theta.csv", index=False)

    # Resume par echelle : distribution inter-jours.
    levels = list(range(1, args.jmax + 1))
    summ = []
    for j in levels:
        col = df[f"theta_j{j}"].to_numpy(dtype=float)
        period_min = args.delta_N * (2 ** j)
        period_max = args.delta_N * (2 ** (j + 1))
        summ.append({
            "j": j,
            "period_min_s": round(period_min, 4),
            "period_max_s": round(period_max, 4),
            "n_days": int(len(col)),
            "median_s": float(np.median(col)),
            "mean_s": float(np.mean(col)),
            "q25_s": float(np.percentile(col, 25)),
            "q75_s": float(np.percentile(col, 75)),
            "pct_neg": float(np.mean(col < 0) * 100),
            "pct_zero": float(np.mean(col == 0) * 100),
            "pct_pos": float(np.mean(col > 0) * 100),
        })
    sdf = pd.DataFrame(summ)
    sdf.to_csv(out / "summary_perday.csv", index=False)

    (out / "config.json").write_text(json.dumps({
        "mode": args.mode, "symbol": args.symbol,
        "start": str(args.start), "end": str(args.end),
        "delta_N": args.delta_N, "jmax": args.jmax,
        "grid_half_width": args.grid_half, "wavelet": args.wavelet,
        "min_trades": args.min_trades, "n_days_used": len(df),
        "convention": conv,
    }, indent=2), encoding="utf-8")

    # Histogrammes (analogue Figure 1 HK) : un sous-graphe par echelle.
    ncol = 4
    nrow = int(np.ceil(args.jmax / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    step = args.delta_N
    for idx, j in enumerate(levels):
        ax = axes[idx // ncol][idx % ncol]
        col = df[f"theta_j{j}"].to_numpy(dtype=float)
        lo, hi = col.min(), col.max()
        # bins alignes sur la grille (multiples de delta_N)
        edges = np.arange(lo - step / 2, hi + step, step)
        if len(edges) < 2:
            edges = np.array([lo - step / 2, lo + step / 2])
        ax.hist(col, bins=edges, color="#3b6ea5", edgecolor="white")
        ax.axvline(0.0, color="black", lw=1)
        ax.axvline(np.median(col), color="crimson", ls="--", lw=1.2,
                   label=f"med {np.median(col):+.2f}s")
        pmin = args.delta_N * (2 ** j)
        pmax = args.delta_N * (2 ** (j + 1))
        ax.set_title(f"j={j}  ({pmin:g}-{pmax:g}s)", fontsize=10)
        ax.set_xlabel("theta_hat (s)")
        ax.legend(fontsize=8)
    for k in range(len(levels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"Per-day lead-lag distribution  |  {args.symbol} {args.mode}  "
                 f"{args.start:%Y-%m-%d}..{args.end:%Y-%m-%d}  (n={len(df)} jours)\n{conv}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out / "hist_perday.png", dpi=130)
    plt.close(fig)

    print()
    print(sdf.to_string(index=False), flush=True)
    print(f"\nConvention: {conv}", flush=True)
    print(f"Artefacts -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
