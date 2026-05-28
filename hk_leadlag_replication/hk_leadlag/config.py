"""Pydantic config models for experiments.

A single ``ExperimentConfig`` object, serialisable to/from YAML, drives every
experiment and keeps the magic constants out of the code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ConfigDict


class EstimatorConfig(BaseModel):
    """Configuration for the wavelet lead-lag estimator."""

    model_config = ConfigDict(extra="forbid")

    wavelet: str = Field(default="db10", description="PyWavelets name, e.g. 'db10'.")
    filter_length: int = Field(default=20, ge=2, description="Daubechies filter length L.")
    j_max: int = Field(default=8, ge=1, description="Highest scale level to estimate.")
    grid_half_width: int = Field(default=100, ge=1, description="Search grid: lags in [-G, G] * Delta_N.")
    delta_N: float = Field(default=1.0, gt=0, description="Finest time resolution Delta_N.")


class SimulationConfig(BaseModel):
    """Configuration for Monte Carlo simulation experiments."""

    model_config = ConfigDict(extra="forbid")

    n_paths: int = Field(default=1000, ge=1)
    N: int = Field(default=14, description="2^N - 1 finest grid points (paper uses N=14).")
    n_steps: int = Field(default=30_000, description="Number of simulation steps n.")
    R: list[float] = Field(
        default_factory=lambda: [0.3, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5],
        description="Cross-spectral correlations R_j (paper Table 1).",
    )
    theta: list[int] = Field(
        default_factory=lambda: [1, 1, 2, 2, 3, 5, 7, 10],
        description="True lead-lags theta_j in units of Delta_N.",
    )
    vol_scenario: Literal["constant", "heston"] = "constant"
    p_miss: tuple[float, float] = Field(default=(0.25, 0.50), description="Lo-MacKinlay missing probas (p1, p2).")
    seed: int | None = 42

    # Heston parameters (used when vol_scenario == 'heston')
    heston_kappa: float = 5.0
    heston_theta: float = 0.04
    heston_xi: float = 0.5
    heston_rho: float = -0.5


class DataConfig(BaseModel):
    """Configuration for empirical data application."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["binance", "coinbase", "bybit", "csv", "parquet"] = "binance"
    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    start_date: str = "2026-04-01"
    end_date: str = "2026-04-30"
    session_start: str = "00:00"
    session_end: str = "23:59"
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")


class ExperimentConfig(BaseModel):
    """Top-level configuration object."""

    model_config = ConfigDict(extra="forbid")

    name: str
    estimator: EstimatorConfig = Field(default_factory=EstimatorConfig)
    simulation: SimulationConfig | None = None
    data: DataConfig | None = None
    output_dir: Path = Path("outputs")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh)
        return cls(**payload)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False)
