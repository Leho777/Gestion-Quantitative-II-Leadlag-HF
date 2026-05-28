"""Live multi-venue tick + L2 book collector.

Motivation
----------
Historical L2 book data (= true HK-style micro-price) is paywalled or
requester-pays on every CEX/DEX archive we have access to. We can however
*collect it ourselves going forward*: poll public REST endpoints every few
seconds and save snapshots.

After N hours of polling we have a real tick-ish dataset including:

* Hyperliquid L2 book (20 levels bid + ask, real micro-price possible)
* Binance bookTicker (best bid/ask, NOT historical, but live works)
* Bybit recent trades
* dYdX v4 trades

This script is intentionally simple: one process, one poll loop, append to
parquet files. No async, no websockets - fine for a 24h overnight run.

Usage::

    python scripts/collect_live_data.py \\
        --duration-h 24 --poll-s 1.0 \\
        --out data/raw/live_collect

Outputs (one file per venue per hour):
    <out>/hyperliquid_l2/<UTC_hour>.parquet  -- one row per snapshot
    <out>/binance_bookticker/<UTC_hour>.parquet
    <out>/bybit_trades/<UTC_hour>.parquet
    <out>/dydx_trades/<UTC_hour>.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"
BINANCE_BOOK = "https://api.binance.com/api/v3/ticker/bookTicker"
BYBIT_TRADES = "https://api.bybit.com/v5/market/recent-trade"
DYDX_TRADES = "https://indexer.dydx.trade/v4/trades/perpetualMarket/BTC-USD"


def utc_hour() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")


def append_parquet(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if path.exists():
        df_old = pd.read_parquet(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(path, index=False)


def poll_hyperliquid(coin: str = "BTC") -> dict | None:
    try:
        r = requests.post(HYPERLIQUID_API, json={"type": "l2Book", "coin": coin}, timeout=5)
        r.raise_for_status()
        book = r.json()
        levels = book.get("levels", [[], []])
        bid_top = levels[0][0] if levels[0] else None
        ask_top = levels[1][0] if levels[1] else None
        if not bid_top or not ask_top:
            return None
        # Micro-price (Stoll inversion).
        pa = float(ask_top["px"]); va = float(ask_top["sz"])
        pb = float(bid_top["px"]); vb = float(bid_top["sz"])
        micro = (pa * vb + pb * va) / (vb + va) if (vb + va) > 0 else None
        return {
            "ts": time.time(),
            "bid_px": pb, "bid_sz": vb,
            "ask_px": pa, "ask_sz": va,
            "mid": 0.5 * (pa + pb),
            "micro": micro,
            "n_bid_levels": len(levels[0]),
            "n_ask_levels": len(levels[1]),
        }
    except Exception:
        return None


def poll_binance_book(symbol: str = "BTCUSDT") -> dict | None:
    try:
        r = requests.get(BINANCE_BOOK, params={"symbol": symbol}, timeout=5)
        r.raise_for_status()
        d = r.json()
        pb = float(d["bidPrice"]); vb = float(d["bidQty"])
        pa = float(d["askPrice"]); va = float(d["askQty"])
        micro = (pa * vb + pb * va) / (vb + va) if (vb + va) > 0 else None
        return {
            "ts": time.time(),
            "bid_px": pb, "bid_sz": vb,
            "ask_px": pa, "ask_sz": va,
            "mid": 0.5 * (pa + pb),
            "micro": micro,
        }
    except Exception:
        return None


def poll_bybit_trades(symbol: str = "BTCUSDT") -> list[dict]:
    try:
        r = requests.get(BYBIT_TRADES, params={"category": "spot", "symbol": symbol, "limit": 50}, timeout=5)
        r.raise_for_status()
        data = r.json()
        rows = data.get("result", {}).get("list", [])
        out = []
        for tr in rows:
            out.append({
                "ts": float(tr["time"]) / 1000.0,
                "price": float(tr["price"]),
                "size": float(tr["size"]),
                "side": tr["side"],
            })
        return out
    except Exception:
        return []


def poll_dydx_trades() -> list[dict]:
    try:
        r = requests.get(DYDX_TRADES, timeout=5)
        r.raise_for_status()
        data = r.json()
        rows = data.get("trades", [])
        out = []
        for tr in rows[:50]:
            out.append({
                "ts": datetime.fromisoformat(tr["createdAt"].replace("Z", "+00:00")).timestamp(),
                "price": float(tr["price"]),
                "size": float(tr["size"]),
                "side": tr["side"],
            })
        return out
    except Exception:
        return []


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--duration-h", type=float, default=24.0)
    p.add_argument("--poll-s", type=float, default=1.0,
                   help="Seconds between L2 book polls. Bybit/dYdX trade polls scale with this.")
    p.add_argument("--out", type=Path, default=Path("data/raw/live_collect"))
    args = p.parse_args()

    start = time.time()
    end = start + args.duration_h * 3600
    print(f"Live data collection: {args.duration_h}h starting at "
          f"{datetime.fromtimestamp(start, tz=timezone.utc).isoformat()}", flush=True)
    print(f"Output: {args.out}", flush=True)

    n_polls = 0
    while time.time() < end:
        n_polls += 1
        hour = utc_hour()
        # Hyperliquid L2 book (the only source of true micro-price).
        hl = poll_hyperliquid("BTC")
        if hl:
            append_parquet(args.out / "hyperliquid_l2" / f"{hour}.parquet", hl)
        # Binance book ticker (best bid/ask LIVE).
        bn = poll_binance_book("BTCUSDT")
        if bn:
            append_parquet(args.out / "binance_bookticker" / f"{hour}.parquet", bn)
        # Bybit recent trades (every Nth poll to avoid duplicates).
        if n_polls % 5 == 0:
            for row in poll_bybit_trades("BTCUSDT"):
                append_parquet(args.out / "bybit_trades" / f"{hour}.parquet", row)
        # dYdX trades.
        if n_polls % 5 == 0:
            for row in poll_dydx_trades():
                append_parquet(args.out / "dydx_trades" / f"{hour}.parquet", row)

        if n_polls % 60 == 0:
            elapsed = time.time() - start
            remaining = end - time.time()
            print(f"  [{n_polls} polls / {elapsed/3600:.2f}h elapsed / {remaining/3600:.2f}h remaining]",
                  flush=True)

        # Sleep
        time.sleep(args.poll_s)
    print(f"Done. Total polls: {n_polls}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
