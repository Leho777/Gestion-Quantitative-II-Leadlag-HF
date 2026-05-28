"""Top-level experiment orchestrator wiring config -> simulator -> estimator -> outputs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hk_leadlag.config import ExperimentConfig
from hk_leadlag.estimators import WaveletLeadLagEstimator, HRYEstimator
from hk_leadlag.simulation import (
    BivariateBrownianSimulator,
    HestonModulatedSimulator,
    HestonVolatilitySimulator,
    LoMacKinlaySampler,
)
from hk_leadlag.wavelet import DaubechiesFilter
from hk_leadlag.analysis.monte_carlo import MonteCarloRunner


@dataclass(slots=True)
class Experiment:
    """Build all components from an :class:`ExperimentConfig` and run them."""

    config: ExperimentConfig

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def build_estimator(self) -> WaveletLeadLagEstimator:
        ec = self.config.estimator
        return WaveletLeadLagEstimator(
            delta_N=ec.delta_N,
            j_max=ec.j_max,
            grid_half_width=ec.grid_half_width,
            daub=DaubechiesFilter(ec.wavelet),
        )

    def build_simulator(self):
        if self.config.simulation is None:
            raise ValueError("config.simulation is None")
        sc = self.config.simulation
        brownian = BivariateBrownianSimulator(
            R=list(sc.R),
            theta=list(sc.theta),
            daub=DaubechiesFilter(self.config.estimator.wavelet),
        )
        if sc.vol_scenario == "constant":
            return brownian
        return HestonModulatedSimulator(
            brownian=brownian,
            heston1=HestonVolatilitySimulator(
                kappa=sc.heston_kappa,
                theta=sc.heston_theta,
                xi=sc.heston_xi,
                rho=sc.heston_rho,
            ),
            heston2=HestonVolatilitySimulator(
                kappa=sc.heston_kappa,
                theta=sc.heston_theta,
                xi=sc.heston_xi,
                rho=sc.heston_rho,
            ),
        )

    def build_sampler(self) -> LoMacKinlaySampler:
        if self.config.simulation is None:
            raise ValueError("config.simulation is None")
        p1, p2 = self.config.simulation.p_miss
        return LoMacKinlaySampler(p1=p1, p2=p2)

    # ------------------------------------------------------------------
    # Runners
    # ------------------------------------------------------------------

    def run_monte_carlo(self) -> pd.DataFrame:
        sim = self.build_simulator()
        sampler = self.build_sampler()
        sc = self.config.simulation
        if sc is None:
            raise ValueError("simulation config required")

        # Run wavelet estimator + HRY baseline.
        wl = self.build_estimator()
        hry = HRYEstimator(
            delta_N=self.config.estimator.delta_N,
            grid_half_width=self.config.estimator.grid_half_width,
        )
        runner_w = MonteCarloRunner(
            simulator=sim, sampler=sampler, estimator=wl,
            n_paths=sc.n_paths, n_steps=sc.n_steps, dt=1.0, seed=sc.seed,
        )
        runner_h = MonteCarloRunner(
            simulator=sim, sampler=sampler, estimator=hry,
            n_paths=sc.n_paths, n_steps=sc.n_steps, dt=1.0, seed=sc.seed,
        )
        res_w = runner_w.run(true_theta=list(sc.theta))
        res_h = runner_h.run(true_theta=[float(np.mean(sc.theta))])
        out = pd.concat(
            [res_w.summary().assign(estimator=res_w.estimator_name),
             res_h.summary().assign(estimator=res_h.estimator_name)],
            ignore_index=True,
        )
        self._save(out, "monte_carlo_summary.csv")
        return out

    # ------------------------------------------------------------------
    def _save(self, df: pd.DataFrame, name: str) -> None:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / name, index=False)

    def dump_config(self) -> None:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.json", "w", encoding="utf-8") as fh:
            json.dump(self.config.model_dump(mode="json"), fh, indent=2, default=str)
