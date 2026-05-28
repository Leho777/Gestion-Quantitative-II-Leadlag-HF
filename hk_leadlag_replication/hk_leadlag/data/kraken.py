"""Kraken public REST trades loader.

Public ``Trades`` endpoint, up to 1000 trades per call, paginated forward by
a ``last`` cursor (nanosecond unix timestamp):

  GET https://api.kraken.com/0/public/Trades?pair=XBTUSDT&since=<unix_s>

Each trade is
``[price, volume, time_s_float, side("b"/"s"), order_type("m"/"l"), misc, trade_id]``.
The unauthenticated rate limit is conservative (~1 call/3s), so a single day
of BTC takes 5-30 minutes depending on activity. Results are cached on disk.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from hk_leadlag.base import BaseDataLoader, NonSyncSeries
from hk_leadlag.data.binance import _dedupe


KRAKEN_BASE = "https://api.kraken.com/0/public"


@dataclass(slots=True)
class KrakenTradesLoader(BaseDataLoader):
    """Download Kraken trades for two pairs and pair them as a NonSyncSeries.

    ``pair1``/``pair2`` are Kraken pair codes over an inclusive UTC date range;
    note Kraken uses XBT for Bitcoin (e.g. ``"XBTUSDT"``). ``sleep_s`` (default
    1.0) leaves margin under the rate limit.
    """

    pair1: str
    pair2: str
    start_date: date
    end_date: date
    cache_dir: Path = field(default_factory=lambda: Path("data/raw/kraken"))
    log_prices: bool = True
    sleep_s: float = 1.0

    def _path(self, pair: str) -> Path:
        return self.cache_dir / pair / f"{pair}-{self.start_date:%Y-%m-%d}-{self.end_date:%Y-%m-%d}.parquet"

    def _download_pair(self, pair: str) -> pd.DataFrame:
        path = self._path(pair)
        if path.exists():
            return pd.read_parquet(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        start_ts = int(datetime(
            self.start_date.year, self.start_date.month, self.start_date.day,
            tzinfo=timezone.utc,
        ).timestamp())
        # end_date is inclusive: collect up to end_date + 1 day at 00:00 UTC.
        end_ts = int(datetime(
            self.end_date.year, self.end_date.month, self.end_date.day,
            tzinfo=timezone.utc,
        ).timestamp()) + 86400

        all_trades: list[list] = []
        since_param = start_ts  # seconds initially, then the ns 'last' cursor
        n_calls = 0
        while True:
            url = f"{KRAKEN_BASE}/Trades"
            params = {"pair": pair, "since": since_param}
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            if data.get("error"):
                raise RuntimeError(f"Kraken API error: {data['error']}")
            result = data["result"]
            keys = [k for k in result.keys() if k != "last"]
            if not keys:
                break
            pair_key = keys[0]
            trades = result[pair_key]
            last_cursor = result["last"]
            if not trades:
                break
            all_trades.extend(trades)
            n_calls += 1

            # Last trade timestamp (in seconds float).
            last_ts = float(trades[-1][2])
            if n_calls % 10 == 0:
                print(f"  [Kraken {pair}] call {n_calls}: {len(all_trades):,} trades "
                      f"so far, at {datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat()}")
            if last_ts >= end_ts:
                break

            # 'last' is nanoseconds; reuse it as the next 'since'.
            since_param = last_cursor
            time.sleep(self.sleep_s)

        df = pd.DataFrame(all_trades, columns=[
            "price", "volume", "time", "side", "order_type", "misc", "trade_id",
        ])
        df["price"] = df["price"].astype(float)
        df["time"] = df["time"].astype(float)
        df = df[(df["time"] >= start_ts) & (df["time"] < end_ts)].reset_index(drop=True)
        df = df[["time", "price", "side"]]
        df["is_buyer_maker"] = df["side"].eq("s")  # 's' = sell market order hit the bid
        df = df.drop(columns=["side"])
        df.to_parquet(path, index=False)
        return df

    def _load_side(self, pair: str) -> tuple[np.ndarray, np.ndarray]:
        df = self._download_pair(pair).sort_values("time").reset_index(drop=True)
        t = df["time"].to_numpy(dtype=float)
        p = df["price"].to_numpy(dtype=float)
        if self.log_prices:
            p = np.log(p)
        return _dedupe(t, p)

    def load(self) -> NonSyncSeries:
        t1, p1 = self._load_side(self.pair1)
        t2, p2 = self._load_side(self.pair2)
        return NonSyncSeries(
            times1=t1, prices1=p1,
            times2=t2, prices2=p2,
            label=f"{self.pair1}@Kraken vs {self.pair2}@Kraken",
        )
