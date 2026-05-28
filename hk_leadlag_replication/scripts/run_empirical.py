"""CLI runner for empirical Hayashi-Koike experiments on crypto data.

Saves all artefacts under ``outputs/<expname>/``:
    config.json, metadata.json, series_stats.json, result.pkl, summary.csv,
    heatmap.png, contrast.png, data_overview.png, log.txt

Usage::

    python scripts/run_empirical.py \\
        --start 2026-04-13 --end 2026-04-19 \\
        --sym1 BTCUSDT --sym2 ETHUSDT \\
        --market spot \\
        --delta-N 0.05 --jmax 8 --grid-half 200

Subsets:
    --hours-cap 6.0   # only analyse the first 6h (sanity check)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hk_leadlag.analysis.artifacts import ArtifactStore, timed
from hk_leadlag.base import NonSyncSeries
from hk_leadlag.data import BinanceTradesLoader
from hk_leadlag.estimators import WaveletLeadLagEstimator
from hk_leadlag.viz.plots import LeadLagPlotter
from hk_leadlag.wavelet import DaubechiesFilter


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HK empirical experiment runner")
    p.add_argument("--start", required=True, type=parse_date)
    p.add_argument("--end", required=True, type=parse_date)
    p.add_argument("--sym1", default="BTCUSDT")
    p.add_argument("--sym2", default="ETHUSDT")
    p.add_argument("--market", choices=["spot", "futures"], default="spot")
    p.add_argument("--delta-N", type=float, default=0.05)
    p.add_argument("--jmax", type=int, default=8)
    p.add_argument("--grid-half", type=int, default=200)
    p.add_argument("--wavelet", default="db10")
    p.add_argument("--hours-cap", type=float, default=0.0, help="If >0, restrict to first H hours.")
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--name", default=None, help="Experiment name (auto if omitted).")
    return p.parse_args(argv)


def build_experiment_name(args: argparse.Namespace) -> str:
    cap = "" if args.hours_cap <= 0 else f"_cap{int(args.hours_cap)}h"
    return (
        f"{args.start.isoformat()}_{args.sym1}_{args.sym2}_{args.market}"
        f"_d{args.delta_N:g}_j{args.jmax}{cap}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    name = args.name or build_experiment_name(args)
    store = ArtifactStore(name=name, root=Path(args.outputs), overwrite=args.overwrite)
    store.mkdir()

    with store.log_tee():
        print(f"=== HK empirical experiment: {name} ===")

        # 1. Persist the config (always).
        store.write_json("config.json", {
            "start": str(args.start), "end": str(args.end),
            "sym1": args.sym1, "sym2": args.sym2, "market": args.market,
            "delta_N": args.delta_N, "jmax": args.jmax,
            "grid_half_width": args.grid_half, "wavelet": args.wavelet,
            "hours_cap": args.hours_cap,
        })

        # 2. Load data.
        with timed("data load") as t_load:
            loader = BinanceTradesLoader(
                symbol1=args.sym1, symbol2=args.sym2,
                start_date=args.start, end_date=args.end,
                market=args.market,
            )
            series = loader.load()
        print(f"Loaded {series.n1:,} + {series.n2:,} ticks")

        # 3. Optional time-cap subset.
        if args.hours_cap > 0:
            mask1 = series.times1 - series.times1[0] < args.hours_cap * 3600
            mask2 = series.times2 - series.times2[0] < args.hours_cap * 3600
            series = NonSyncSeries(
                times1=series.times1[mask1], prices1=series.prices1[mask1],
                times2=series.times2[mask2], prices2=series.prices2[mask2],
                label=f"{series.label} (first {args.hours_cap}h)",
            )
            print(f"Capped to {series.n1:,} + {series.n2:,} ticks")

        store.save_series_stats(series)

        # 4. Fit the estimator.
        est = WaveletLeadLagEstimator(
            delta_N=args.delta_N, j_max=args.jmax,
            grid_half_width=args.grid_half,
            daub=DaubechiesFilter(args.wavelet),
        )
        with timed("HK estimator fit") as t_fit:
            result = est.fit(series)

        # 5. Save the raw result + summary.
        store.save_result(result)
        store.save_summary(result, delta_N=args.delta_N)
        print("\nSummary:")
        import pandas as pd
        print(pd.read_csv(store.path("summary.csv")).to_string(index=False))

        # 6. Plots.
        with timed("plots"):
            plotter = LeadLagPlotter(figsize=(11, 5))
            plotter.plot_heatmap_2d(result, save=store.path("heatmap.png"))
            plotter.plot_contrast(result, save=store.path("contrast.png"))
            plotter.plot_data_overview(
                series.times1 - series.times1[0], series.prices1,
                series.times2 - series.times2[0], series.prices2,
                labels=(args.sym1, args.sym2),
                save=store.path("data_overview.png"),
            )

        # 7. Final metadata.
        store.save_metadata(
            n1=series.n1, n2=series.n2,
            duration_s=float(series.times1[-1] - series.times1[0]),
            elapsed_load_s=t_load["elapsed_s"],
            elapsed_fit_s=t_fit["elapsed_s"],
            theta_hat=result.theta_hat.tolist(),
            levels=result.levels.tolist(),
        )
        print(f"\nAll artefacts saved to: {store.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
