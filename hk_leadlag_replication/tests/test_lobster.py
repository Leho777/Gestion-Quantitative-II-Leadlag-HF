"""Tests for the LOBSTER quote-based loader."""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from hk_leadlag.data import LobsterLoader


def _write_lobster_zip(
    root: Path,
    ticker: str,
    *,
    date: str = "2012-06-21",
    level: int = 10,
    message_rows: list[str] | None = None,
    orderbook_rows: list[str] | None = None,
) -> None:
    path = root / f"LOBSTER_SampleFile_{ticker}_{date}_{level}.zip"
    prefix = f"{ticker}_{date}_34200000_34201000"
    message = "\n".join(
        message_rows
        or [
            "34200.000000,1,1,10,1000000,1",
            "34200.000100,1,2,10,1000100,1",
            "34200.000100,1,3,10,1000200,1",
            "34200.000300,1,4,10,1000300,1",
        ]
    )
    orderbook = "\n".join(
        orderbook_rows
        or [
            "1010000,10,990000,20",
            "1012000,10,990000,20",
            "1014000,10,992000,20",
            "1016000,10,994000,20",
        ]
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{prefix}_message_{level}.csv", message)
        zf.writestr(f"{prefix}_orderbook_{level}.csv", orderbook)


def test_lobster_loader_reads_zip_midpoints_and_dedupes(tmp_path: Path):
    _write_lobster_zip(tmp_path, "AAA")
    _write_lobster_zip(tmp_path, "BBB")

    loader = LobsterLoader(
        ticker1="AAA",
        ticker2="BBB",
        data_dir=tmp_path,
        level=10,
        log_prices=False,
    )
    series = loader.load()

    assert series.label == "AAA vs BBB (LOBSTER midpoint)"
    np.testing.assert_allclose(series.times1, [34200.0, 34200.0001, 34200.0003])
    np.testing.assert_allclose(series.times2, [34200.0, 34200.0001, 34200.0003])
    # Midpoints after price_scale=10_000:
    # row 1: (101 + 99) / 2 = 100
    # rows 2/3 duplicate timestamp: mean((101.2+99)/2, (101.4+99.2)/2) = 100.2
    # row 4: (101.6 + 99.4) / 2 = 100.5
    np.testing.assert_allclose(series.prices1, [100.0, 100.2, 100.5])
    np.testing.assert_allclose(series.prices2, [100.0, 100.2, 100.5])


def test_lobster_loader_microprice_mode_uses_sizes(tmp_path: Path):
    _write_lobster_zip(tmp_path, "AAA")
    _write_lobster_zip(tmp_path, "BBB")

    loader = LobsterLoader(
        ticker1="AAA",
        ticker2="BBB",
        data_dir=tmp_path,
        level=10,
        log_prices=False,
        price_mode="microprice",
    )
    series = loader.load()

    assert series.label == "AAA vs BBB (LOBSTER microprice)"
    # row 1: (101 * 20 + 99 * 10) / 30 = 100.3333...
    # row 2: (101.2 * 20 + 99 * 10) / 30 = 100.4666...
    # row 3: (101.4 * 20 + 99.2 * 10) / 30 = 100.6666...
    # duplicate timestamp average = 100.5666...
    # row 4: (101.6 * 20 + 99.4 * 10) / 30 = 100.8666...
    np.testing.assert_allclose(
        series.prices1,
        [100.3333333333, 100.5666666667, 100.8666666667],
        rtol=1e-10,
        atol=1e-10,
    )


def test_lobster_loader_session_filter_and_log_prices(tmp_path: Path):
    _write_lobster_zip(tmp_path, "AAA")
    _write_lobster_zip(tmp_path, "BBB")

    loader = LobsterLoader(
        ticker1="AAA",
        ticker2="BBB",
        data_dir=tmp_path,
        level=10,
        start_time_s=34200.00005,
        end_time_s=34200.00035,
    )
    series = loader.load()

    np.testing.assert_allclose(series.times1, [34200.0001, 34200.0003])
    expected_duplicate_log = 0.5 * (np.log(100.1) + np.log(100.3))
    np.testing.assert_allclose(series.prices1, [expected_duplicate_log, np.log(100.5)])


def test_lobster_loader_missing_source_raises(tmp_path: Path):
    loader = LobsterLoader(
        ticker1="MISSING",
        ticker2="BBB",
        data_dir=tmp_path,
        level=10,
    )

    with pytest.raises(FileNotFoundError, match="Could not find LOBSTER files"):
        loader.load()


def test_lobster_loader_level_mismatch_raises(tmp_path: Path):
    _write_lobster_zip(tmp_path, "AAA", level=10)
    _write_lobster_zip(tmp_path, "BBB", level=10)
    loader = LobsterLoader(
        ticker1="AAA",
        ticker2="BBB",
        data_dir=tmp_path,
        level=30,
    )

    with pytest.raises(FileNotFoundError, match="level 30"):
        loader.load()


def test_lobster_loader_message_orderbook_length_mismatch_raises(tmp_path: Path):
    _write_lobster_zip(
        tmp_path,
        "AAA",
        message_rows=[
            "34200.000000,1,1,10,1000000,1",
            "34200.000100,1,2,10,1000100,1",
        ],
        orderbook_rows=[
            "1010000,10,990000,20",
        ],
    )
    _write_lobster_zip(tmp_path, "BBB")
    loader = LobsterLoader(
        ticker1="AAA",
        ticker2="BBB",
        data_dir=tmp_path,
        level=10,
    )

    with pytest.raises(ValueError, match="length mismatch"):
        loader.load()
