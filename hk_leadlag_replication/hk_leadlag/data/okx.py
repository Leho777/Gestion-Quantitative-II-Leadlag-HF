"""OKX v5 history-trades REST loader.

Endpoint: ``GET https://www.okx.com/api/v5/market/history-trades``, 100 trades
per call, paginated by ``after=<tradeId>`` (older than tradeId) with
``type=2`` for history. Each trade is
``{instId, side, sz, px, tradeId, ts(unix_ms), ...}``.

Caveat: pagination only runs backwards from the latest trade, so reaching
data more than a few days old means scrolling 100k+ pages (~100 trades each
on BTC-USDT). Kept as a reusable component for near-realtime tests rather
than deep historical pulls.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from hk_leadlag.base import BaseDataLoader, NonSyncSeries
from hk_leadlag.data.binance import _dedupe


OKX_BASE = "https://www.okx.com/api/v5/market/history-trades"


@dataclass(slots=True)
class OKXTradesLoader(BaseDataLoader):
    """Download OKX trades for one symbol over a date range.

    Designed for cross-venue setups where one stream comes from OKX and the
    other from a different exchange. ``load()`` only covers OKX vs OKX, which
    is rarely what you want.
    """

    inst1: str
    inst2: str
    start_date: date
    end_date: date
    cache_dir: Path = field(default_factory=lambda: Path("data/raw/okx"))
    log_prices: bool = True
    sleep_s: float = 0.05
    page_size: int = 100  # OKX cap for history-trades

    def _path(self, inst: str) -> Path:
        return self.cache_dir / inst / f"{inst}-{self.start_date:%Y-%m-%d}-{self.end_date:%Y-%m-%d}.parquet"

    def _download_one(self, inst: str) -> pd.DataFrame:
        path = self._path(inst)
        if path.exists():
            return pd.read_parquet(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        start_ms = int(datetime(
            self.start_date.year, self.start_date.month, self.start_date.day,
            tzinfo=timezone.utc,
        ).timestamp() * 1000)
        end_ms = (int(datetime(
            self.end_date.year, self.end_date.month, self.end_date.day,
            tzinfo=timezone.utc,
        ).timestamp()) + 86400) * 1000

        # No cursor on the first call (latest trades), then walk back with 'after'.
        all_rows: list[dict] = []
        after: str | None = None
        n_calls = 0
        while True:
            params = {"instId": inst, "limit": str(self.page_size), "type": "2"}
            if after is not None:
                params["after"] = after
            r = requests.get(OKX_BASE, params=params, timeout=20)
            r.raise_for_status()
            payload = r.json()
            if payload.get("code") not in ("0", 0):
                raise RuntimeError(f"OKX API error: {payload.get('msg')} (code {payload.get('code')})")
            rows = payload.get("data", [])
            if not rows:
                break
            n_calls += 1

            # Drop trades outside the target window.
            keep = []
            stop = False
            for tr in rows:
                ts = int(tr["ts"])
                if ts > end_ms:
                    continue
                if ts < start_ms:
                    stop = True
                    continue
                keep.append({
                    "time_ms": ts,
                    "price": float(tr["px"]),
                    "side": tr["side"],
                    "tradeId": tr["tradeId"],
                })
            all_rows.extend(keep)

            oldest_ts = int(rows[-1]["ts"])
            if n_calls % 50 == 0:
                print(f"  [OKX {inst}] call {n_calls}: {len(all_rows):,} kept; oldest "
                      f"= {datetime.fromtimestamp(oldest_ts / 1000, tz=timezone.utc).isoformat()}")
            if stop or oldest_ts < start_ms:
                break
            after = rows[-1]["tradeId"]
            time.sleep(self.sleep_s)

        df = pd.DataFrame(all_rows)
        if df.empty:
            raise RuntimeError(f"No OKX trades collected for {inst} in range.")
        df = df.sort_values("time_ms").reset_index(drop=True)
        df["is_buyer_maker"] = df["side"].str.lower().eq("sell")
        df = df[["time_ms", "price", "is_buyer_maker"]]
        df.to_parquet(path, index=False)
        return df

    def _load_side(self, inst: str) -> tuple[np.ndarray, np.ndarray]:
        df = self._download_one(inst)
        t = df["time_ms"].to_numpy(dtype=float) / 1000.0  # to seconds
        p = df["price"].to_numpy(dtype=float)
        if self.log_prices:
            p = np.log(p)
        return _dedupe(t, p)

    def load(self) -> NonSyncSeries:
        t1, p1 = self._load_side(self.inst1)
        t2, p2 = self._load_side(self.inst2)
        return NonSyncSeries(
            times1=t1, prices1=p1,
            times2=t2, prices2=p2,
            label=f"{self.inst1}@OKX vs {self.inst2}@OKX",
        )
