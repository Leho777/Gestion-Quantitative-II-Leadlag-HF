"""Generic CSV / parquet loader.

Expected columns:

    timestamp_us  | price

(integer microsecond timestamp + float price). One file per series.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hk_leadlag.base import BaseDataLoader, NonSyncSeries


@dataclass(slots=True)
class CsvNonSyncLoader(BaseDataLoader):
    """Load two CSV/parquet files containing tick-level prices.

    ``path1``/``path2`` point at the two files (csv, csv.gz or parquet);
    ``log_prices`` takes ``log(price)`` after loading.
    """

    path1: Path
    path2: Path
    label: str = ""
    log_prices: bool = True
    time_col: str = "timestamp_us"
    price_col: str = "price"
    time_unit: str = "us"  # 'us', 'ms', 's' - interpreted relative to seconds

    def _read(self, path: Path) -> pd.DataFrame:
        path = Path(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def load(self) -> NonSyncSeries:
        df1 = self._read(self.path1)
        df2 = self._read(self.path2)
        scale = {"us": 1e-6, "ms": 1e-3, "s": 1.0}[self.time_unit]
        t1 = df1[self.time_col].to_numpy(dtype=float) * scale
        t2 = df2[self.time_col].to_numpy(dtype=float) * scale
        p1 = df1[self.price_col].to_numpy(dtype=float)
        p2 = df2[self.price_col].to_numpy(dtype=float)
        if self.log_prices:
            p1 = np.log(p1)
            p2 = np.log(p2)
        return NonSyncSeries(
            times1=t1,
            prices1=p1,
            times2=t2,
            prices2=p2,
            label=self.label or f"{self.path1.stem} vs {self.path2.stem}",
        )
