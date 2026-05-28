"""Binance historical trades loader.

Pulls daily aggregate-trade zips from the free public S3 bucket
``data.binance.vision`` and caches them under ``data/raw/binance/``.

Columns of the *aggTrades* CSVs (no header):
    aggregate_tradeId, price, quantity, first_tradeId, last_tradeId,
    timestamp_ms, was_buyer_maker, was_best_match
"""
from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from hk_leadlag.base import BaseDataLoader, NonSyncSeries

BINANCE_BASE = "https://data.binance.vision/data/spot/daily/aggTrades"
BINANCE_FUT_BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"


@dataclass(slots=True)
class BinanceTradesLoader(BaseDataLoader):
    """Download/cache Binance aggregate trades for two symbols and pair them.

    ``symbol1``/``symbol2`` (e.g. ``"BTCUSDT"``, ``"ETHUSDT"``) must trade on
    the same venue, over an inclusive date range (one CSV per day). ``market``
    is ``"spot"`` (default) or ``"futures"`` (USDT-M perps).
    """

    symbol1: str
    symbol2: str
    start_date: date
    end_date: date
    market: str = "spot"
    cache_dir: Path = field(default_factory=lambda: Path("data/raw/binance"))
    log_prices: bool = True

    # ------------------------------------------------------------------
    def _base_url(self) -> str:
        return BINANCE_BASE if self.market == "spot" else BINANCE_FUT_BASE

    def _date_range(self) -> Iterable[date]:
        d = self.start_date
        while d <= self.end_date:
            yield d
            d += timedelta(days=1)

    def _path(self, symbol: str, day: date) -> Path:
        # Key by (market, symbol) so spot and futures caches never collide.
        return self.cache_dir / self.market / symbol / f"{symbol}-aggTrades-{day:%Y-%m-%d}.parquet"

    def _download_day(self, symbol: str, day: date) -> pd.DataFrame:
        path = self._path(symbol, day)
        if path.exists():
            return pd.read_parquet(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self._base_url()}/{symbol}/{symbol}-aggTrades-{day:%Y-%m-%d}.zip"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            inner = zf.namelist()[0]
            if self.market == "spot":
                # Spot CSVs: no header, 8 columns.
                df = pd.read_csv(
                    zf.open(inner),
                    header=None,
                    names=[
                        "agg_id", "price", "qty",
                        "first_id", "last_id",
                        "timestamp_ms", "is_buyer_maker", "is_best_match",
                    ],
                )
            else:
                # Futures (USDT-M / COIN-M) CSVs: header row, 7 columns (no
                # is_best_match). Standardise to the spot schema.
                df = pd.read_csv(zf.open(inner))
                df = df.rename(columns={
                    "agg_trade_id": "agg_id",
                    "quantity": "qty",
                    "first_trade_id": "first_id",
                    "last_trade_id": "last_id",
                    "transact_time": "timestamp_ms",
                })
                df["is_best_match"] = False  # not provided by futures archive
                df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().map(
                    {"true": True, "false": False}
                )
                df = df[[
                    "agg_id", "price", "qty",
                    "first_id", "last_id",
                    "timestamp_ms", "is_buyer_maker", "is_best_match",
                ]]
        df.to_parquet(path, index=False)
        return df

    def _load_symbol(self, symbol: str) -> pd.DataFrame:
        frames = [self._download_day(symbol, d) for d in self._date_range()]
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("timestamp_ms").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    def load(self) -> NonSyncSeries:
        df1 = self._load_symbol(self.symbol1)
        df2 = self._load_symbol(self.symbol2)
        # Binance changed aggTrades resolution around 2024-2025: older files are
        # ms (13 digits), newer ones µs (16 digits). _to_unix_seconds auto-detects.
        t1 = _to_unix_seconds(df1["timestamp_ms"].to_numpy(dtype=float))
        t2 = _to_unix_seconds(df2["timestamp_ms"].to_numpy(dtype=float))
        p1 = df1["price"].to_numpy(dtype=float)
        p2 = df2["price"].to_numpy(dtype=float)
        if self.log_prices:
            p1 = np.log(p1)
            p2 = np.log(p2)
        t1, p1 = _dedupe(t1, p1)
        t2, p2 = _dedupe(t2, p2)
        return NonSyncSeries(
            times1=t1,
            prices1=p1,
            times2=t2,
            prices2=p2,
            label=f"{self.symbol1} vs {self.symbol2} ({self.market})",
        )


def _dedupe(t: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse repeated timestamps to their mean price, keeping times strictly increasing."""
    if len(t) == 0:
        return t, p
    df = pd.DataFrame({"t": t, "p": p}).groupby("t", as_index=False)["p"].mean()
    return df["t"].to_numpy(), df["p"].to_numpy()


def _to_unix_seconds(raw: np.ndarray) -> np.ndarray:
    """Convert aggTrades timestamps to seconds, detecting ms vs µs resolution.

    Binance aggTrades CSVs are ms (13 digits, ~1e12) before ~2025 and µs
    (16 digits, ~1e15) after the resolution upgrade; the median magnitude
    picks the divisor.
    """
    if len(raw) == 0:
        return raw.astype(float)
    median_val = float(np.median(raw))
    # 2020 ≈ 1.6e9 s, 1.6e12 ms, 1.6e15 µs; 2030 ≈ 1.9e9 / 1.9e12 / 1.9e15.
    if median_val > 1e14:
        divisor = 1_000_000.0  # microseconds
    elif median_val > 1e11:
        divisor = 1_000.0  # milliseconds
    else:
        divisor = 1.0  # already seconds
    return raw.astype(float) / divisor
