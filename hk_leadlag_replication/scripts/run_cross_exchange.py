"""Cross-exchange HK lead-lag : same ticker, two CEXes.

Closest available analogue of HK 2020 (NASDAQ vs BATS): same asset on two
truly independent exchanges, both with publicly archived tick data.

Implemented: BTCUSDT @ Binance vs BTCUSDT @ Bybit (both spot).

Usage::

    python scripts/run_cross_exchange.py \\
        --start 2026-04-13 --end 2026-04-19 \\
        --symbol BTCUSDT \\
        --delta-N 0.05 --jmax 8 --grid-half 400
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hk_leadlag.analysis.artifacts import ArtifactStore, timed
from hk_leadlag.base import NonSyncSeries
from hk_leadlag.data.binance import BinanceTradesLoader, _to_unix_seconds, _dedupe
from hk_leadlag.data.bybit import BybitTradesLoader
from hk_leadlag.estimators import WaveletLeadLagEstimator
from hk_leadlag.viz.plots import LeadLagPlotter
from hk_leadlag.wavelet import DaubechiesFilter


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, type=parse_date)
    p.add_argument("--end", required=True, type=parse_date)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--delta-N", type=float, default=0.05)
    p.add_argument("--jmax", type=int, default=8)
    p.add_argument("--grid-half", type=int, default=400)
    p.add_argument("--wavelet", default="db10")
    p.add_argument("--name", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    name = args.name or (
        f"{args.start.isoformat()}_{args.symbol.lower()}_binance_vs_bybit"
        f"_d{args.delta_N:g}_j{args.jmax}"
    )
    store = ArtifactStore(name=name)
    store.mkdir()

    with store.log_tee():
        print(f"=== Cross-exchange HK: {args.symbol} Binance vs Bybit ===")
        store.write_json("config.json", {
            "setup": f"{args.symbol} @ Binance spot (sym1) vs Bybit spot (sym2)",
            "start": str(args.start), "end": str(args.end),
            "symbol": args.symbol,
            "delta_N": args.delta_N, "jmax": args.jmax,
            "grid_half_width": args.grid_half, "wavelet": args.wavelet,
        })

        with timed("load Binance spot"):
            binance_loader = BinanceTradesLoader(
                symbol1=args.symbol, symbol2=args.symbol,
                start_date=args.start, end_date=args.end,
                market="spot", log_prices=True,
            )
            df_bn = binance_loader._load_symbol(args.symbol)
            t1 = _to_unix_seconds(df_bn["timestamp_ms"].to_numpy(dtype=float))
            p1 = np.log(df_bn["price"].to_numpy(dtype=float))
            t1, p1 = _dedupe(t1, p1)

        with timed("load Bybit spot"):
            bybit_loader = BybitTradesLoader(
                symbol1=args.symbol, symbol2=args.symbol,
                start_date=args.start, end_date=args.end,
                market1="spot", market2="spot",
            )
            t2, p2 = bybit_loader._load_side("spot", args.symbol)

        series = NonSyncSeries(
            times1=t1, prices1=p1,
            times2=t2, prices2=p2,
            label=f"{args.symbol} @ Binance vs Bybit ({args.start}/{args.end})",
        )
        print(f"Binance: {series.n1:,} trades   Bybit: {series.n2:,} trades")
        store.save_series_stats(series)

        est = WaveletLeadLagEstimator(
            delta_N=args.delta_N, j_max=args.jmax,
            grid_half_width=args.grid_half,
            daub=DaubechiesFilter(args.wavelet),
        )
        with timed("HK fit"):
            result = est.fit(series)

        store.save_result(result)
        store.save_summary(result, delta_N=args.delta_N)

        plotter = LeadLagPlotter(figsize=(11, 5))
        plotter.plot_heatmap_2d(result, save=store.path("heatmap.png"))
        plotter.plot_contrast(result, save=store.path("contrast.png"))

        import pandas as pd
        print()
        print(pd.read_csv(store.path("summary.csv")).to_string(index=False))
        print()
        print("Convention: theta > 0 means Bybit (sym2) leads Binance (sym1).")
        print("            theta < 0 means Binance (sym1) leads Bybit (sym2).")

        store.save_metadata(
            n1=series.n1, n2=series.n2,
            duration_s=float(series.times1[-1] - series.times1[0]),
            theta_hat=result.theta_hat.tolist(),
            levels=result.levels.tolist(),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
