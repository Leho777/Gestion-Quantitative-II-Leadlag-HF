"""Same-asset cross-market lead-lag : BTCUSDT spot vs BTCUSDT USDT-M perp.

Closest analogue to HK 2020 (NASDAQ vs BATS) available in Binance Vision:
same underlying asset (BTC), two different markets (spot order book vs
perpetual futures order book) trading in parallel within Binance.

Differences from HK setup:
- Same parent venue (Binance) instead of two truly independent exchanges.
- Different products (spot vs perp) instead of same product cross-exchange.

But it preserves the spirit: same-asset, two markets, expecting price
discovery to flow in one direction or the other depending on scale.

Usage::

    python scripts/run_cross_market_btc.py \\
        --start 2026-04-13 --end 2026-04-13 \\
        --delta-N 0.05 --jmax 8 --grid-half 400
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hk_leadlag.analysis.artifacts import ArtifactStore, timed
from hk_leadlag.base import NonSyncSeries
from hk_leadlag.data.binance import BinanceTradesLoader
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


def load_one_market(symbol: str, market: str, start: date, end: date) -> tuple[np.ndarray, np.ndarray]:
    """Load aggTrades for one (symbol, market) pair, return (times_s, log_prices)."""
    # Reuse BinanceTradesLoader by giving it the same symbol twice; pick series 1.
    loader = BinanceTradesLoader(
        symbol1=symbol, symbol2=symbol,
        start_date=start, end_date=end,
        market=market, log_prices=True,
    )
    df = loader._load_symbol(symbol)
    from hk_leadlag.data.binance import _to_unix_seconds, _dedupe
    t = _to_unix_seconds(df["timestamp_ms"].to_numpy(dtype=float))
    p = np.log(df["price"].to_numpy(dtype=float))
    return _dedupe(t, p)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    name = args.name or (
        f"{args.start.isoformat()}_{args.symbol.lower()}_spot_vs_perp"
        f"_d{args.delta_N:g}_j{args.jmax}"
    )
    store = ArtifactStore(name=name)
    store.mkdir()

    with store.log_tee():
        print(f"=== HK cross-market: {args.symbol} spot vs perp ===")
        store.write_json("config.json", {
            "setup": "BTCUSDT spot (sym1) vs BTCUSDT USDT-M perp (sym2)",
            "start": str(args.start), "end": str(args.end),
            "symbol": args.symbol,
            "delta_N": args.delta_N, "jmax": args.jmax,
            "grid_half_width": args.grid_half, "wavelet": args.wavelet,
        })

        with timed("load spot"):
            t1, p1 = load_one_market(args.symbol, "spot", args.start, args.end)
        with timed("load perp"):
            t2, p2 = load_one_market(args.symbol, "futures", args.start, args.end)

        series = NonSyncSeries(
            times1=t1, prices1=p1,
            times2=t2, prices2=p2,
            label=f"{args.symbol} spot vs perp ({args.start} to {args.end})",
        )
        print(f"spot: {series.n1:,} trades   perp: {series.n2:,} trades")
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
        print("Convention: theta > 0 means PERP (sym2) leads SPOT (sym1).")
        print("            theta < 0 means SPOT (sym1) leads PERP (sym2).")
        print("Crypto literature expectation: perp typically leads spot.")

        store.save_metadata(
            n1=series.n1, n2=series.n2,
            duration_s=float(series.times1[-1] - series.times1[0]),
            theta_hat=result.theta_hat.tolist(),
            levels=result.levels.tolist(),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
