"""Event-study extension: contrast lead-lag estimates around event windows.

For each event date, fit theta_hat_j on daily panels in a [-K, +K] window and
aggregate across events. Used to check whether the multi-scale lead-lag
structure breaks during regime shifts (FOMC, hacks, ETF approvals, halving).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from hk_leadlag.base import BaseLeadLagEstimator, NonSyncSeries


@dataclass(slots=True)
class EventStudy:
    """Aggregate per-day lead-lag estimates around event dates.

    window_days is the half-width K of the event window.
    """

    estimator: BaseLeadLagEstimator
    window_days: int = 5

    def run(
        self,
        daily_series: dict[date, NonSyncSeries],
        events: Iterable[date],
    ) -> pd.DataFrame:
        """Return a long DataFrame with columns ``event_date, offset, level, theta_hat``."""
        rows = []
        all_dates = sorted(daily_series.keys())
        for ev in events:
            try:
                idx = all_dates.index(ev)
            except ValueError:
                continue
            lo = max(0, idx - self.window_days)
            hi = min(len(all_dates), idx + self.window_days + 1)
            for k in range(lo, hi):
                day = all_dates[k]
                offset = (day - ev).days
                res = self.estimator.fit(daily_series[day])
                for j, theta in zip(res.levels, res.theta_hat):
                    rows.append(
                        {
                            "event_date": ev,
                            "day": day,
                            "offset": offset,
                            "level": int(j),
                            "theta_hat": float(theta),
                        }
                    )
        return pd.DataFrame(rows)
