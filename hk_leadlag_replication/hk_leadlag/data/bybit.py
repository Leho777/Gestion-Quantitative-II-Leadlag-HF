"""Bybit historical trades loader (free public archive).

Bybit publishes per-day trade dumps at https://public.bybit.com:
* Spot: ``public.bybit.com/spot/<symbol>/<symbol>_<YYYY-MM-DD>.csv.gz``
   columns: id, timestamp(ms), price, volume, side, rpi
* Derivatives (linear perpetual): ``public.bybit.com/trading/<symbol>/<symbol><YYYY-MM-DD>.csv.gz``
   columns: timestamp(s, float), symbol, side, size, price, tickDirection, trdMatchID, grossValue, homeNotional, foreignNotional, RPI

Both markets are mapped to the same ``(times_s, log_prices)`` representation
as the Binance loader, then paired into a ``NonSyncSeries``.
"""
from __future__ import annotations

import gzip
import io
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from hk_leadlag.base import BaseDataLoader, NonSyncSeries
from hk_leadlag.data.binance import _to_unix_seconds, _dedupe


BYBIT_SPOT_BASE = "https://public.bybit.com/spot"
BYBIT_PERP_BASE = "https://public.bybit.com/trading"


@dataclass(slots=True)
class BybitTradesLoader(BaseDataLoader):
    """Download Bybit historical trades for two symbols/markets and pair them.

    ``symbol1``/``symbol2`` (e.g. ``"BTCUSDT"``) over an inclusive date range.
    ``market1``/``market2`` are ``"spot"`` or ``"perp"`` per side, default spot.
    """

    symbol1: str
    symbol2: str
    start_date: date
    end_date: date
    market1: str = "spot"
    market2: str = "spot"
    cache_dir: Path = field(default_factory=lambda: Path("data/raw/bybit"))
    log_prices: bool = True

    def _date_range(self) -> Iterable[date]:
        d = self.start_date
        while d <= self.end_date:
            yield d
            d += timedelta(days=1)

    def _url(self, market: str, symbol: str, day: date) -> str:
        if market == "spot":
            return f"{BYBIT_SPOT_BASE}/{symbol}/{symbol}_{day:%Y-%m-%d}.csv.gz"
        return f"{BYBIT_PERP_BASE}/{symbol}/{symbol}{day:%Y-%m-%d}.csv.gz"

    def _path(self, market: str, symbol: str, day: date) -> Path:
        return self.cache_dir / market / symbol / f"{symbol}-{day:%Y-%m-%d}.parquet"

    def _download_day(self, market: str, symbol: str, day: date) -> pd.DataFrame:
        path = self._path(market, symbol, day)
        if path.exists():
            return pd.read_parquet(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        url = self._url(market, symbol, day)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with gzip.open(io.BytesIO(resp.content)) as fh:
            if market == "spot":
                df = pd.read_csv(fh)
                df = df.rename(columns={
                    "timestamp": "timestamp_ms",
                    "volume": "qty",
                })
                df["is_buyer_maker"] = df["side"].str.lower().eq("sell")
            else:
                df = pd.read_csv(fh)
                # Perp timestamps are float seconds: scale to ms to match the
                # spot schema and downstream _to_unix_seconds logic.
                df["timestamp_ms"] = (df["timestamp"].astype(float) * 1000.0).astype("int64")
                df = df.rename(columns={"size": "qty"})
                df["is_buyer_maker"] = df["side"].str.lower().eq("sell")
        keep = ["timestamp_ms", "price", "qty", "is_buyer_maker"]
        df = df[keep]
        df.to_parquet(path, index=False)
        return df

    def _load_side(self, market: str, symbol: str) -> tuple[np.ndarray, np.ndarray]:
        frames = [self._download_day(market, symbol, d) for d in self._date_range()]
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("timestamp_ms").reset_index(drop=True)
        t = _to_unix_seconds(df["timestamp_ms"].to_numpy(dtype=float))
        p = df["price"].to_numpy(dtype=float)
        if self.log_prices:
            p = np.log(p)
        return _dedupe(t, p)

    def load(self) -> NonSyncSeries:
        t1, p1 = self._load_side(self.market1, self.symbol1)
        t2, p2 = self._load_side(self.market2, self.symbol2)
        return NonSyncSeries(
            times1=t1, prices1=p1,
            times2=t2, prices2=p2,
            label=f"{self.symbol1}@Bybit-{self.market1} vs {self.symbol2}@Bybit-{self.market2}",
        )
