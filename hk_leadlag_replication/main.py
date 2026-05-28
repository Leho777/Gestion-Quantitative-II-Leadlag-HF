"""Autonomous CLI entry point for HK-LeadLag experiments.

Usage::

    python main.py --config configs/simulation_default.yaml
    python main.py --config configs/empirical_btc_eth.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hk_leadlag.config import ExperimentConfig
from hk_leadlag.analysis.experiment import Experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HK-LeadLag experiment runner")
    p.add_argument("--config", required=True, type=Path, help="YAML experiment config")
    p.add_argument("--mode", choices=["simulation", "empirical", "all"], default="simulation")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)
    exp = Experiment(config)
    exp.dump_config()

    if args.mode in {"simulation", "all"} and config.simulation is not None:
        print(f"[main] running Monte Carlo for '{config.name}'...")
        summary = exp.run_monte_carlo()
        print(summary.to_string(index=False))

    if args.mode in {"empirical", "all"} and config.data is not None:
        print("[main] Empirical runs are driven by dedicated scripts (richer CLI):")
        print("         python scripts/run_empirical.py --help          # cross-asset")
        print("         python scripts/run_cross_exchange.py --help      # cross-venue")
        print("         python scripts/run_cross_exchange_kraken.py --help")
        print("       or open notebooks/02_main_narrative.ipynb for the full story.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
