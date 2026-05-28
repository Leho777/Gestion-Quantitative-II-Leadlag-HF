"""LOBSTER sample loader.

LOBSTER files come in pairs:

* ``*_message_<level>.csv``: event timestamps and order metadata.
* ``*_orderbook_<level>.csv``: ask/bid prices and sizes after each event.

Extracts a level-1 quote-based price proxy from the order book and returns a
``NonSyncSeries`` for the Hayashi-Koike estimators.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
import pandas as pd

from hk_leadlag.base import BaseDataLoader, NonSyncSeries
from hk_leadlag.data.preprocess import build_micro_price


@dataclass(slots=True)
class LobsterLoader(BaseDataLoader):
    """Load two LOBSTER message/orderbook pairs as quote-based log-price series.

    ``data_dir`` holds either LOBSTER sample ZIPs or extracted CSVs; ``level``
    is the book depth in the filename (only level-1 columns are read) and
    ``date`` matches the standard LOBSTER filename date. ``price_scale``
    reflects that LOBSTER stores equity prices as integer dollars x 10,000.
    ``price_mode`` is ``"midpoint"`` (default) or ``"microprice"`` from level-1
    bid/ask sizes; ``start_time_s``/``end_time_s`` give an optional session
    filter in seconds since midnight.
    """

    ticker1: str
    ticker2: str
    data_dir: Path = field(default_factory=lambda: Path("data/equity"))
    level: int = 10
    date: str = "2012-06-21"
    price_scale: float = 10_000.0
    log_prices: bool = True
    price_mode: Literal["midpoint", "microprice"] = "midpoint"
    start_time_s: float | None = None
    end_time_s: float | None = None

    def load(self) -> NonSyncSeries:
        t1, p1 = self._load_ticker(self.ticker1)
        t2, p2 = self._load_ticker(self.ticker2)
        return NonSyncSeries(
            times1=t1,
            prices1=p1,
            times2=t2,
            prices2=p2,
            label=f"{self.ticker1} vs {self.ticker2} (LOBSTER {self.price_mode})",
        )

    def _load_ticker(self, ticker: str) -> tuple[np.ndarray, np.ndarray]:
        source = self._find_source(ticker)
        if source.suffix.lower() == ".zip":
            times, midpoint = self._read_zip(source)
        else:
            times, midpoint = self._read_directory(source, ticker)
        if self.start_time_s is not None or self.end_time_s is not None:
            lo = -np.inf if self.start_time_s is None else self.start_time_s
            hi = np.inf if self.end_time_s is None else self.end_time_s
            mask = (times >= lo) & (times <= hi)
            times = times[mask]
            midpoint = midpoint[mask]
        if self.log_prices:
            midpoint = np.log(midpoint)
        return _dedupe(times, midpoint)

    def _find_source(self, ticker: str) -> Path:
        data_dir = Path(self.data_dir)
        zip_name = f"LOBSTER_SampleFile_{ticker}_{self.date}_{self.level}.zip"
        direct_zip = data_dir / zip_name
        if direct_zip.exists():
            return direct_zip

        extracted = data_dir / f"{ticker}_lvl{self.level}"
        if extracted.exists():
            return extracted

        matches = sorted(data_dir.rglob(zip_name))
        if matches:
            return matches[0]

        orderbook_glob = f"{ticker}_{self.date}_*_orderbook_{self.level}.csv"
        csv_matches = sorted(data_dir.rglob(orderbook_glob))
        if csv_matches:
            return csv_matches[0].parent

        raise FileNotFoundError(
            f"Could not find LOBSTER files for {ticker} level {self.level} under {data_dir}"
        )

    def _read_zip(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        with zipfile.ZipFile(path) as zf:
            message_name, orderbook_name = _select_lobster_pair(zf.namelist(), self.level)
            with zf.open(message_name) as msg_fh, zf.open(orderbook_name) as book_fh:
                return self._read_frames(msg_fh, book_fh, source=str(path))

    def _read_directory(self, path: Path, ticker: str) -> tuple[np.ndarray, np.ndarray]:
        files = [p.name for p in Path(path).iterdir() if p.is_file()]
        message_name, orderbook_name = _select_lobster_pair(files, self.level, ticker=ticker)
        with open(Path(path) / message_name, "rb") as msg_fh:
            with open(Path(path) / orderbook_name, "rb") as book_fh:
                return self._read_frames(msg_fh, book_fh, source=str(path))

    def _read_frames(
        self,
        message_fh: BinaryIO,
        orderbook_fh: BinaryIO,
        *,
        source: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        msg = pd.read_csv(
            message_fh,
            header=None,
            usecols=[0],
            names=["time_s"],
        )
        book = pd.read_csv(
            orderbook_fh,
            header=None,
            usecols=[0, 1, 2, 3],
            names=["ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1"],
        )
        if len(msg) != len(book):
            raise ValueError(
                f"LOBSTER message/orderbook length mismatch in {source}: "
                f"{len(msg)} vs {len(book)}"
            )
        times = msg["time_s"].to_numpy(dtype=float)
        ask = book["ask_price_1"].to_numpy(dtype=float) / self.price_scale
        bid = book["bid_price_1"].to_numpy(dtype=float) / self.price_scale
        ask_sz = book["ask_size_1"].to_numpy(dtype=float)
        bid_sz = book["bid_size_1"].to_numpy(dtype=float)
        if self.price_mode == "microprice":
            price = build_micro_price(
                bid_px=bid,
                ask_px=ask,
                bid_sz=bid_sz,
                ask_sz=ask_sz,
            )
        else:
            price = 0.5 * (ask + bid)
        valid = (
            np.isfinite(times)
            & np.isfinite(price)
            & np.isfinite(ask_sz)
            & np.isfinite(bid_sz)
            & (price > 0)
        )
        return times[valid], price[valid]


def _select_lobster_pair(
    names: list[str],
    level: int,
    *,
    ticker: str | None = None,
) -> tuple[str, str]:
    """Return the message and orderbook filenames from a LOBSTER file list."""
    candidates = [(Path(n).name, n) for n in names]
    if ticker is not None:
        candidates = [
            (base, original)
            for base, original in candidates
            if base.startswith(f"{ticker}_")
        ]
    message = [original for base, original in candidates if f"_message_{level}.csv" in base]
    orderbook = [original for base, original in candidates if f"_orderbook_{level}.csv" in base]
    if len(message) != 1 or len(orderbook) != 1:
        raise FileNotFoundError(
            f"Expected one message and one orderbook CSV for level {level}; "
            f"found {len(message)} message, {len(orderbook)} orderbook"
        )
    return message[0], orderbook[0]


def _dedupe(times: np.ndarray, prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse repeated timestamps by averaging prices."""
    if len(times) == 0:
        return times, prices
    df = pd.DataFrame({"time": times, "price": prices})
    df = df.groupby("time", as_index=False)["price"].mean()
    return df["time"].to_numpy(dtype=float), df["price"].to_numpy(dtype=float)
