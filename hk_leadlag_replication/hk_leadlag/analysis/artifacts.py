"""Artifact management for reproducible experiment outputs.

Every experiment lives in its own subdirectory under ``outputs/<name>/`` with
the config, raw result, summary tables, plots, and metadata. Resumable via the
``ArtifactStore.exists`` check.

Layout::

    outputs/
    └── 2026-04-13_btc_eth_spot_db10_jmax6/
        ├── config.json          # serialised ExperimentConfig
        ├── metadata.json        # timestamps, n_ticks, durations, code-version
        ├── result.pkl           # full LeadLagResult (theta_hat, contrast, ...)
        ├── summary.csv          # one row per scale: j, theta_hat, contrast_peak
        ├── heatmap.png
        ├── contrast.png
        ├── data_overview.png
        └── log.txt              # captured stdout
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from hk_leadlag.base import LeadLagResult, NonSyncSeries


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ArtifactStore:
    """Single source of truth for an experiment's outputs.

    ``name`` becomes the subdirectory under ``root``. With ``overwrite=False``,
    steps whose output already exists are skipped.
    """

    name: str
    root: Path = field(default_factory=lambda: Path("outputs"))
    overwrite: bool = False

    @property
    def dir(self) -> Path:
        return self.root / self.name

    def mkdir(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir

    # ------------------------------------------------------------------
    # File primitives
    # ------------------------------------------------------------------

    def path(self, filename: str) -> Path:
        return self.dir / filename

    def exists(self, filename: str) -> bool:
        return self.path(filename).exists()

    def write_json(self, filename: str, payload: dict[str, Any]) -> Path:
        p = self.path(filename)
        self.mkdir()
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)
        return p

    def read_json(self, filename: str) -> dict[str, Any]:
        with open(self.path(filename), "r", encoding="utf-8") as fh:
            return json.load(fh)

    def write_pickle(self, filename: str, payload: Any) -> Path:
        p = self.path(filename)
        self.mkdir()
        with open(p, "wb") as fh:
            pickle.dump(payload, fh)
        return p

    def read_pickle(self, filename: str) -> Any:
        with open(self.path(filename), "rb") as fh:
            return pickle.load(fh)

    def write_csv(self, filename: str, df: pd.DataFrame) -> Path:
        p = self.path(filename)
        self.mkdir()
        df.to_csv(p, index=False)
        return p

    # ------------------------------------------------------------------
    # Domain helpers
    # ------------------------------------------------------------------

    def save_metadata(self, **extra: Any) -> Path:
        meta = {
            "name": self.name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
        }
        meta.update(extra)
        return self.write_json("metadata.json", meta)

    def save_result(self, result: LeadLagResult) -> Path:
        return self.write_pickle("result.pkl", _result_to_serialisable(result))

    def load_result(self) -> LeadLagResult:
        d = self.read_pickle("result.pkl")
        return LeadLagResult(
            theta_hat=np.asarray(d["theta_hat"]),
            levels=np.asarray(d["levels"]),
            contrast=np.asarray(d["contrast"]) if d.get("contrast") is not None else None,
            grid=np.asarray(d["grid"]) if d.get("grid") is not None else None,
            estimator_name=d.get("estimator_name", ""),
            metadata=d.get("metadata", {}),
        )

    def save_summary(self, result: LeadLagResult, delta_N: float) -> Path:
        rows = []
        for j_idx, j in enumerate(result.levels):
            period_min = (2 ** int(j)) * delta_N
            period_max = (2 ** (int(j) + 1)) * delta_N
            peak = float(result.contrast[j_idx].max()) if result.contrast is not None else float("nan")
            rows.append({
                "j": int(j),
                "period_min_s": period_min,
                "period_max_s": period_max,
                "theta_hat_s": float(result.theta_hat[j_idx]),
                "contrast_peak": peak,
            })
        df = pd.DataFrame(rows)
        return self.write_csv("summary.csv", df)

    def save_series_stats(self, series: NonSyncSeries) -> Path:
        dt1 = np.diff(series.times1)
        dt2 = np.diff(series.times2)
        stats = {
            "n1": int(series.n1),
            "n2": int(series.n2),
            "T_start_unix": float(min(series.times1[0], series.times2[0])),
            "T_end_unix": float(max(series.times1[-1], series.times2[-1])),
            "duration_seconds": float(max(series.times1[-1], series.times2[-1])
                                       - min(series.times1[0], series.times2[0])),
            "median_dt1_ms": float(np.median(dt1) * 1000) if len(dt1) else 0.0,
            "median_dt2_ms": float(np.median(dt2) * 1000) if len(dt2) else 0.0,
            "mean_dt1_ms": float(np.mean(dt1) * 1000) if len(dt1) else 0.0,
            "mean_dt2_ms": float(np.mean(dt2) * 1000) if len(dt2) else 0.0,
            "label": series.label,
        }
        return self.write_json("series_stats.json", stats)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @contextmanager
    def log_tee(self, filename: str = "log.txt") -> Iterator[None]:
        """Tee stdout to both console and ``filename`` for the duration."""
        self.mkdir()
        path = self.path(filename)

        class _Tee:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, data):
                for s in self.streams:
                    s.write(data)
                    s.flush()

            def flush(self):
                for s in self.streams:
                    s.flush()

        orig = sys.stdout
        fh = open(path, "w", encoding="utf-8", buffering=1)
        sys.stdout = _Tee(orig, fh)
        try:
            yield
        finally:
            sys.stdout = orig
            fh.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def _result_to_serialisable(r: LeadLagResult) -> dict[str, Any]:
    return {
        "theta_hat": np.asarray(r.theta_hat),
        "levels": np.asarray(r.levels),
        "contrast": None if r.contrast is None else np.asarray(r.contrast),
        "grid": None if r.grid is None else np.asarray(r.grid),
        "estimator_name": r.estimator_name,
        "metadata": dict(r.metadata),
    }


@contextmanager
def timed(label: str) -> Iterator[dict]:
    """Context manager that records elapsed seconds in a dict."""
    out = {"label": label, "elapsed_s": 0.0}
    t0 = time.time()
    print(f"[timed] start: {label}")
    try:
        yield out
    finally:
        out["elapsed_s"] = time.time() - t0
        print(f"[timed] done : {label} in {out['elapsed_s']:.1f}s")
