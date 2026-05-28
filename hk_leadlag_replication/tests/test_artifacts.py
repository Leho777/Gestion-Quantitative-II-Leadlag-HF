"""Tests for the ArtifactStore reproducibility helper."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hk_leadlag.analysis.artifacts import ArtifactStore, timed
from hk_leadlag.base import LeadLagResult, NonSyncSeries


def test_store_mkdir_and_path(tmp_path: Path):
    store = ArtifactStore(name="dummy", root=tmp_path)
    assert store.dir == tmp_path / "dummy"
    assert not store.dir.exists()
    store.mkdir()
    assert store.dir.is_dir()


def test_write_and_read_json(tmp_path: Path):
    store = ArtifactStore(name="run1", root=tmp_path)
    payload = {"k1": 1, "k2": [1.0, 2.0], "k3": "x"}
    p = store.write_json("config.json", payload)
    assert p.exists()
    loaded = store.read_json("config.json")
    assert loaded == payload


def test_write_json_handles_numpy(tmp_path: Path):
    store = ArtifactStore(name="run2", root=tmp_path)
    payload = {
        "arr": np.array([1.0, 2.0, 3.0]),
        "int": np.int64(42),
        "float": np.float64(3.14),
    }
    store.write_json("nu.json", payload)
    raw = json.loads((store.path("nu.json")).read_text())
    assert raw["arr"] == [1.0, 2.0, 3.0]
    assert raw["int"] == 42
    assert abs(raw["float"] - 3.14) < 1e-9


def test_save_result_round_trip(tmp_path: Path):
    store = ArtifactStore(name="rt", root=tmp_path)
    grid = np.linspace(-1, 1, 11)
    contrast = np.abs(np.sin(grid))[None, :].repeat(3, axis=0)
    result = LeadLagResult(
        theta_hat=np.array([0.0, 0.1, -0.2]),
        levels=np.array([1, 2, 3]),
        contrast=contrast,
        grid=grid,
        estimator_name="dummy",
        metadata={"n": 100},
    )
    store.save_result(result)
    loaded = store.load_result()
    np.testing.assert_allclose(loaded.theta_hat, result.theta_hat)
    np.testing.assert_array_equal(loaded.levels, result.levels)
    np.testing.assert_allclose(loaded.grid, result.grid)
    assert loaded.estimator_name == "dummy"
    assert loaded.metadata == {"n": 100}


def test_save_summary_row_per_scale(tmp_path: Path):
    store = ArtifactStore(name="sum", root=tmp_path)
    grid = np.linspace(-1, 1, 11)
    contrast = np.tile(np.abs(grid), (3, 1))
    result = LeadLagResult(
        theta_hat=np.array([0.05, 0.10, 0.20]),
        levels=np.array([1, 2, 3]),
        contrast=contrast,
        grid=grid,
        estimator_name="dummy",
    )
    p = store.save_summary(result, delta_N=0.05)
    df = pd.read_csv(p)
    assert list(df.columns) == [
        "j", "period_min_s", "period_max_s", "theta_hat_s", "contrast_peak",
    ]
    assert len(df) == 3
    assert df["j"].tolist() == [1, 2, 3]
    np.testing.assert_allclose(df["theta_hat_s"].tolist(), [0.05, 0.10, 0.20])


def test_save_series_stats(tmp_path: Path):
    rng = np.random.default_rng(0)
    t1 = np.cumsum(rng.exponential(scale=0.05, size=1000))
    t2 = np.cumsum(rng.exponential(scale=0.07, size=1000))
    series = NonSyncSeries(
        times1=t1, prices1=rng.standard_normal(1000),
        times2=t2, prices2=rng.standard_normal(1000),
        label="random",
    )
    store = ArtifactStore(name="series", root=tmp_path)
    p = store.save_series_stats(series)
    s = json.loads(p.read_text())
    assert s["n1"] == 1000
    assert s["n2"] == 1000
    assert s["median_dt1_ms"] > 0
    assert s["mean_dt1_ms"] > 0


def test_save_metadata_has_timestamp(tmp_path: Path):
    store = ArtifactStore(name="meta", root=tmp_path)
    p = store.save_metadata(custom_key=42)
    s = json.loads(p.read_text())
    assert s["name"] == "meta"
    assert "created_utc" in s
    assert s["python"]
    assert s["custom_key"] == 42


def test_timed_context_records_elapsed(capsys):
    with timed("dummy task") as info:
        # No-op
        pass
    assert info["elapsed_s"] >= 0
    captured = capsys.readouterr()
    assert "dummy task" in captured.out


def test_exists_check(tmp_path: Path):
    store = ArtifactStore(name="ex", root=tmp_path)
    assert not store.exists("anything.json")
    store.write_json("anything.json", {"a": 1})
    assert store.exists("anything.json")
