"""BTC USDT-margined perp vs Coin-margined perp on Binance Futures.

Same asset (BTC), two distinct futures product families:
  - USDT-M : BTCUSDT contract, settles in USDT (stablecoin), held by
    retail + macro players.
  - COIN-M : BTCUSD_PERP contract, settles in BTC, held more by
    crypto-native treasury / hedgers.

These have different funding curves, different participant mixes, and
different basis dynamics vs spot.

Usage::

    python scripts/run_usdt_vs_coin_perp.py \\
        --start 2026-04-13 --end 2026-04-19 \\
        --delta-N 0.05 --jmax 8 --grid-half 400
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hk_leadlag.analysis.artifacts import ArtifactStore, timed
from hk_leadlag.base import NonSyncSeries
from hk_leadlag.data.binance import _to_unix_seconds, _dedupe
from hk_leadlag.estimators import WaveletLeadLagEstimator
from hk_leadlag.viz.plots import LeadLagPlotter
from hk_leadlag.wavelet import DaubechiesFilter


COINM_BASE = "https://data.binance.vision/data/futures/cm/daily/aggTrades"
CACHE = ROOT / "data" / "raw" / "binance" / "coinm"


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, type=parse_date)
    p.add_argument("--end", required=True, type=parse_date)
    p.add_argument("--delta-N", type=float, default=0.05)
    p.add_argument("--jmax", type=int, default=8)
    p.add_argument("--grid-half", type=int, default=400)
    p.add_argument("--name", default=None)
    return p.parse_args(argv)


def download_coinm_day(day: date) -> pd.DataFrame:
    sym = "BTCUSD_PERP"
    path = CACHE / sym / f"{sym}-{day:%Y-%m-%d}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{COINM_BASE}/{sym}/{sym}-aggTrades-{day:%Y-%m-%d}.zip"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        inner = zf.namelist()[0]
        df = pd.read_csv(zf.open(inner))
    # COIN-M schema may differ; we keep the columns we need.
    df = df.rename(columns={
        "agg_trade_id": "agg_id",
        "quantity": "qty",
        "first_trade_id": "first_id",
        "last_trade_id": "last_id",
        "transact_time": "timestamp_ms",
    })
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    df = df[["price", "qty", "timestamp_ms", "is_buyer_maker"]]
    df.to_parquet(path, index=False)
    return df


def load_coinm(start: date, end: date) -> tuple[np.ndarray, np.ndarray]:
    d = start
    frames = []
    while d <= end:
        frames.append(download_coinm_day(d))
        d += timedelta(days=1)
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)
    t = _to_unix_seconds(df["timestamp_ms"].to_numpy(dtype=float))
    p = np.log(df["price"].to_numpy(dtype=float))
    return _dedupe(t, p)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    name = args.name or f"{args.start.isoformat()}_btc_usdt_perp_vs_coin_perp_d{args.delta_N:g}_j{args.jmax}"
    store = ArtifactStore(name=name)
    store.mkdir()
    with store.log_tee():
        print("=== HK USDT-M perp vs COIN-M perp (BTC) ===")
        store.write_json("config.json", {
            "setup": "BTCUSDT USDT-M perp (sym1) vs BTCUSD_PERP COIN-M perp (sym2)",
            "start": str(args.start), "end": str(args.end),
            "delta_N": args.delta_N, "jmax": args.jmax,
            "grid_half_width": args.grid_half, "wavelet": "db10",
        })

        from hk_leadlag.data.binance import BinanceTradesLoader
        with timed("load USDT-M perp"):
            usdt_loader = BinanceTradesLoader(
                symbol1="BTCUSDT", symbol2="BTCUSDT",
                start_date=args.start, end_date=args.end,
                market="futures", log_prices=True,
            )
            df_usdt = usdt_loader._load_symbol("BTCUSDT")
            t1 = _to_unix_seconds(df_usdt["timestamp_ms"].to_numpy(dtype=float))
            p1 = np.log(df_usdt["price"].to_numpy(dtype=float))
            t1, p1 = _dedupe(t1, p1)

        with timed("load COIN-M perp"):
            t2, p2 = load_coinm(args.start, args.end)

        series = NonSyncSeries(
            times1=t1, prices1=p1, times2=t2, prices2=p2,
            label=f"BTC USDT-M perp vs COIN-M perp ({args.start}/{args.end})",
        )
        print(f"USDT-M: {series.n1:,} trades   COIN-M: {series.n2:,} trades")
        store.save_series_stats(series)

        est = WaveletLeadLagEstimator(
            delta_N=args.delta_N, j_max=args.jmax,
            grid_half_width=args.grid_half,
            daub=DaubechiesFilter("db10"),
        )
        with timed("HK fit"):
            result = est.fit(series)
        store.save_result(result)
        store.save_summary(result, delta_N=args.delta_N)
        plotter = LeadLagPlotter(figsize=(11, 5))
        plotter.plot_heatmap_2d(result, save=store.path("heatmap.png"))
        plotter.plot_contrast(result, save=store.path("contrast.png"))

        print()
        print(pd.read_csv(store.path("summary.csv")).to_string(index=False))
        print()
        print("Convention: theta > 0 means COIN-M perp (sym2) leads USDT-M perp (sym1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
