"""Synthetic midpoint loader for Binance aggTrades.

Binance Data Vision (data.binance.vision) does not publish historical
bookTicker archives (HTTP 404 on all probed paths as of 2026-05), so the
closest public midpoint proxy is the aggTrades stream and its
``is_buyer_maker`` flag. Per Binance's matching-engine convention:

  * ``is_buyer_maker == True``  -> trade at the best bid (a sell market order
    hit a resting buy limit order).
  * ``is_buyer_maker == False`` -> trade at the best ask (a buy market order
    hit a resting sell limit order).

An approximate midpoint follows from two rolling-last filters:
  * ``last_bid_price(t)``: most recent ``is_buyer_maker=True`` trade at time <= t.
  * ``last_ask_price(t)``: most recent ``is_buyer_maker=False`` trade at time <= t.
  * ``synthetic_mid(t) = 0.5 * (last_bid_price(t) + last_ask_price(t))``.

One midpoint is emitted per aggTrade event. Caveats: it is stale between
updates (a side keeps its last traded price until the next trade on that
side), reflects only traded bid/ask rather than displayed top-of-book, and
under-estimates the spread when it is wider than the recent trade range. For
HK-style HF lead-lag it sits closer to the micro-price than the literal
midpoint, but is far less noisy than raw trades bouncing across the spread.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from hk_leadlag.base import BaseDataLoader, NonSyncSeries
from hk_leadlag.data.binance import BinanceTradesLoader, _to_unix_seconds, _dedupe


@dataclass(slots=True)
class BinanceSyntheticMidpointLoader(BaseDataLoader):
    """Approximate midpoint from Binance aggTrades using the is_buyer_maker flag.

    ``mode="midpoint"`` (default) averages the last bid/ask trades:
    ``mid = 0.5 * (last_bid_price + last_ask_price)``.

    ``mode="microprice"`` is a trade-volume-weighted stand-in for HK's Stoll
    micro-price. The true formula uses resting book sizes,
    ``micro = (P_ask * V_bid_book + P_bid * V_ask_book) / (V_bid_book + V_ask_book)``,
    but Binance Vision exposes no historical L2/bookTicker, so the executed
    volumes of the last bid and ask trades stand in for the resting depths.
    Trade volume is not resting depth, so this is a "trade-volume-weighted
    synthetic micro-price", not a true HK micro-price.
    """

    symbol1: str
    symbol2: str
    start_date: date
    end_date: date
    market: str = "spot"
    cache_dir: Path = field(default_factory=lambda: Path("data/raw/binance"))
    log_prices: bool = True
    mode: str = "midpoint"

    def _load_symbol(self, symbol: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (times_s, reconstructed_price) for one symbol."""
        if self.mode not in {"midpoint", "microprice"}:
            raise ValueError(f"Unknown mode: {self.mode!r}")

        loader = BinanceTradesLoader(
            symbol1=symbol, symbol2=symbol,
            start_date=self.start_date, end_date=self.end_date,
            market=self.market, cache_dir=self.cache_dir,
            log_prices=False,
        )
        df = loader._load_symbol(symbol)
        df = df.sort_values("timestamp_ms").reset_index(drop=True)
        ts = _to_unix_seconds(df["timestamp_ms"].to_numpy(dtype=float))
        price = df["price"].to_numpy(dtype=float)
        qty = df["qty"].to_numpy(dtype=float)
        is_maker = df["is_buyer_maker"].to_numpy(dtype=bool)

        # Rolling last price and qty for each side.
        last_bid_p = np.empty_like(price)
        last_ask_p = np.empty_like(price)
        last_bid_q = np.empty_like(price)
        last_ask_q = np.empty_like(price)
        cur_bid_p = cur_ask_p = np.nan
        cur_bid_q = cur_ask_q = np.nan
        for i in range(len(price)):
            if is_maker[i]:
                cur_bid_p = price[i]
                cur_bid_q = qty[i]
            else:
                cur_ask_p = price[i]
                cur_ask_q = qty[i]
            last_bid_p[i] = cur_bid_p
            last_ask_p[i] = cur_ask_p
            last_bid_q[i] = cur_bid_q
            last_ask_q[i] = cur_ask_q

        valid = (
            ~np.isnan(last_bid_p) & ~np.isnan(last_ask_p) & (last_ask_p >= last_bid_p)
            & ~np.isnan(last_bid_q) & ~np.isnan(last_ask_q)
            & (last_bid_q + last_ask_q > 0)
        )
        ts = ts[valid]
        bp = last_bid_p[valid]
        ap = last_ask_p[valid]
        bq = last_bid_q[valid]
        aq = last_ask_q[valid]

        if self.mode == "midpoint":
            recon = 0.5 * (bp + ap)
        else:  # microprice approximation (Stoll-style inversion)
            recon = (ap * bq + bp * aq) / (bq + aq)

        ts, recon = _dedupe(ts, recon)
        if self.log_prices:
            recon = np.log(recon)
        return ts, recon

    def load(self) -> NonSyncSeries:
        t1, p1 = self._load_symbol(self.symbol1)
        t2, p2 = self._load_symbol(self.symbol2)
        return NonSyncSeries(
            times1=t1, prices1=p1,
            times2=t2, prices2=p2,
            label=f"{self.symbol1} vs {self.symbol2} synth-{self.mode} ({self.market})",
        )
