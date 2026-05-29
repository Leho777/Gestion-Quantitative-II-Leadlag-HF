"""Download LOBSTER sample data (free, 1 day, NASDAQ 5 stocks).

Sample = 21 June 2012, level 10 order book for AAPL, AMZN, GOOG, INTC, MSFT.
Source: https://lobsterdata.com/info/DataSamples.php

Usage::

    python scripts/download_lobster_sample.py
    python scripts/download_lobster_sample.py --tickers AAPL MSFT --level 5
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
LOBSTER_BASE = "https://lobsterdata.com/info/sample"


def download_one(ticker: str, level: int, out_dir: Path) -> Path:
    """Download and unzip one ticker / level combo. Returns the unzipped folder."""
    url = f"{LOBSTER_BASE}/LOBSTER_SampleFile_{ticker}_2012-06-21_{level}.zip"
    target = out_dir / f"{ticker}_lvl{level}"
    if target.exists() and any(target.iterdir()):
        print(f"  [{ticker}] already downloaded, skipping")
        return target
    target.mkdir(parents=True, exist_ok=True)
    print(f"  [{ticker}] fetching {url} ...")
    resp = requests.get(url, timeout=60, headers={"User-Agent": "hk-leadlag-academic/1.0"})
    if resp.status_code != 200:
        print(f"  [{ticker}] HTTP {resp.status_code} - sample may be moved.")
        print(f"  Try manually: https://lobsterdata.com/info/DataSamples.php")
        return target
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(target)
    print(f"  [{ticker}] saved to {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=["AAPL", "AMZN", "GOOG", "INTC", "MSFT"])
    p.add_argument("--level", type=int, default=10, choices=[1, 5, 10, 30, 50])
    p.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "lobster")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for ticker in args.tickers:
        download_one(ticker, args.level, args.out)
    print(f"\nDone. Files under: {args.out}")
    print("Next step: build a NonSyncSeries from the message/orderbook CSVs with")
    print("hk_leadlag/data/lobster.py (LobsterLoader).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
