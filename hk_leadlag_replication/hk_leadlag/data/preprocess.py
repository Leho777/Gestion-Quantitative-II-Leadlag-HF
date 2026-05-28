"""Pre-processing helpers: micro-prices, session filters, log returns."""
from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pandas as pd

from hk_leadlag.base import NonSyncSeries


def build_micro_price(
    bid_px: np.ndarray,
    ask_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_sz: np.ndarray,
) -> np.ndarray:
    """Volume-weighted micro-price, eq. (2.2) of HK20."""
    total = bid_sz + ask_sz
    out = np.where(total > 0, (ask_px * bid_sz + bid_px * ask_sz) / total, 0.5 * (bid_px + ask_px))
    return out


def restrict_to_session(
    series: NonSyncSeries,
    session_start: time,
    session_end: time,
    timestamp_origin: datetime | None = None,
) -> NonSyncSeries:
    """Keep only observations within ``[session_start, session_end]`` of each day."""
    def _mask(t: np.ndarray) -> np.ndarray:
        ts = pd.to_datetime(t, unit="s")
        local = ts.time
        return np.array([session_start <= h <= session_end for h in local])

    m1 = _mask(series.times1)
    m2 = _mask(series.times2)
    return NonSyncSeries(
        times1=series.times1[m1],
        prices1=series.prices1[m1],
        times2=series.times2[m2],
        prices2=series.prices2[m2],
        label=series.label,
    )
