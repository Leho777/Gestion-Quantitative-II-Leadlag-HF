"""CLI runner for LOBSTER midpoint lead-lag experiments.

Example:

    python scripts/run_lobster_pair.py --ticker1 SPY --ticker2 AAPL --level 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hk_leadlag.analysis.artifacts import ArtifactStore, timed
from hk_leadlag.data import LobsterLoader
from hk_leadlag.estimators import WaveletLeadLagEstimator
from hk_leadlag.viz.plots import LeadLagPlotter
from hk_leadlag.wavelet import DaubechiesFilter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HK LOBSTER midpoint experiment runner")
    p.add_argument("--ticker1", required=True)
    p.add_argument("--ticker2", required=True)
    p.add_argument("--date", default="2012-06-21")
    p.add_argument("--level", type=int, default=10)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "equity")
    p.add_argument("--delta-N", type=float, default=0.001)
    p.add_argument("--jmax", type=int, default=8)
    p.add_argument("--grid-half", type=int, default=200)
    p.add_argument("--wavelet", default="db10")
    p.add_argument("--start-time-s", type=float, default=None)
    p.add_argument("--end-time-s", type=float, default=None)
    p.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--name", default=None, help="Experiment name (auto if omitted).")
    return p.parse_args(argv)


def build_experiment_name(args: argparse.Namespace) -> str:
    cap = ""
    if args.start_time_s is not None or args.end_time_s is not None:
        lo = "start" if args.start_time_s is None else f"{args.start_time_s:g}"
        hi = "end" if args.end_time_s is None else f"{args.end_time_s:g}"
        cap = f"_{lo}to{hi}s"
    return (
        f"{args.date}_{args.ticker1.lower()}_{args.ticker2.lower()}"
        f"_lobster_lvl{args.level}{cap}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    name = args.name or build_experiment_name(args)
    store = ArtifactStore(name=name, root=args.outputs, overwrite=args.overwrite)
    store.mkdir()

    with store.log_tee():
        print(f"=== LOBSTER equity experiment: {args.ticker1} vs {args.ticker2}, {args.date} ===")
        store.write_json("config.json", {
            "source": "LOBSTER sample (TU Berlin)",
            "date": args.date,
            "ticker1": args.ticker1,
            "ticker2": args.ticker2,
            "level": args.level,
            "data_dir": str(args.data_dir),
            "delta_N": args.delta_N,
            "jmax": args.jmax,
            "grid_half_width": args.grid_half,
            "wavelet": args.wavelet,
            "start_time_s": args.start_time_s,
            "end_time_s": args.end_time_s,
        })

        with timed("LOBSTER load") as t_load:
            loader = LobsterLoader(
                ticker1=args.ticker1,
                ticker2=args.ticker2,
                data_dir=args.data_dir,
                level=args.level,
                date=args.date,
                start_time_s=args.start_time_s,
                end_time_s=args.end_time_s,
            )
            series = loader.load()
        print(f"{args.ticker1}: {series.n1:,} events  {args.ticker2}: {series.n2:,} events")
        store.save_series_stats(series)

        est = WaveletLeadLagEstimator(
            delta_N=args.delta_N,
            j_max=args.jmax,
            grid_half_width=args.grid_half,
            daub=DaubechiesFilter(args.wavelet),
        )
        with timed("HK fit on LOBSTER") as t_fit:
            result = est.fit(series)

        store.save_result(result)
        store.save_summary(result, delta_N=args.delta_N)
        print("\nSummary:")
        import pandas as pd
        print(pd.read_csv(store.path("summary.csv")).to_string(index=False))

        with timed("plots"):
            plotter = LeadLagPlotter(figsize=(11, 5))
            plotter.plot_heatmap_2d(result, save=store.path("heatmap.png"))
            plotter.plot_contrast(result, save=store.path("contrast.png"))
            plotter.plot_data_overview(
                series.times1 - series.times1[0],
                series.prices1,
                series.times2 - series.times2[0],
                series.prices2,
                labels=(args.ticker1, args.ticker2),
                save=store.path("data_overview.png"),
            )

        store.save_metadata(
            n1=series.n1,
            n2=series.n2,
            duration_s=float(max(series.times1[-1], series.times2[-1])
                             - min(series.times1[0], series.times2[0])),
            elapsed_load_s=t_load["elapsed_s"],
            elapsed_fit_s=t_fit["elapsed_s"],
            theta_hat=result.theta_hat.tolist(),
            levels=result.levels.tolist(),
        )
        print(f"\nAll artefacts saved to: {store.dir}")
        print(f"Reminder: theta>0 means {args.ticker2} (sym2) leads {args.ticker1} (sym1)")
        print(f"          theta<0 means {args.ticker1} (sym1) leads {args.ticker2} (sym2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
