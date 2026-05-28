# HK-LeadLag : Replication of the estimator + crypto adaptation of Hayashi & Koike (2020)

**Multi-scale analysis of lead-lag relationships in high-frequency financial markets**

*arXiv:1708.03992, Independent Python implementation, M2 Quantitative Management II, Paris Dauphine PSL.*

> **Scope, calibrated** : we replicate the **estimator** of Hayashi-Koike (HY non-synchronous
> covariance → Daubechies wavelet convolution → argmax per scale), and adapt it to crypto
> cross-venue / cross-product data. We do **NOT** literally replicate HK's empirical protocol
> (NASDAQ vs BATS quote-update micro-prices, 108 stocks × 21 days, ~0.1 ms resolution,
> SIP/participant timestamps). Specifically:
> - **Price proxy** : aggTrades on crypto (no historical bookTicker on Binance Vision) vs
>   micro-prices in HK. LOBSTER (NASDAQ only) used as quote-midpoint sanity check, not as
>   NASDAQ-vs-BATS replication.
> - **Resolution** : Δ_N = 50 ms on crypto vs ~0.1 ms in HK (limited by crypto inter-tick
>   median ~ 30 ms on Binance BTC spot).
> - **Timestamping** : exchange-API timestamps in crypto, no SIP consolidation. The measured
>   lag may mix true price discovery and reporting conventions.
> - **Panel design** : a handful of cross-venue / cross-product cases vs 108 × 21 = 2268
>   ticker-days in HK. No systematic panel distribution.
> - **Estimator comparison** : we benchmark HK vs HRY (single-scale) ; HK paper also compares
>   to Dobrev-Schaumburg, which we have only in simplified form.

## Overview

Modular OOP framework implementing the wavelet-based, scale-by-scale lead-lag
estimator of Hayashi & Koike (2020) for non-synchronously sampled high-frequency
financial data. Includes:

- Multi-scale `WaveletLeadLagEstimator` and single-scale `HRYEstimator` baseline.
- Numba-accelerated Hayashi-Yoshida cross-covariance (88× speedup vs pure Python).
- Bivariate Brownian + Heston + Lo-MacKinlay sampling for Monte Carlo validation.
- Binance Vision aggTrades loader (auto-detects μs / ms timestamp resolution).
- LOBSTER (TU Berlin) order-book midpoint loader for equity replication.
- Block bootstrap confidence intervals on θ̂_j and multiple-testing helpers.
- Artifact-driven experiment system inspired by reproducible-research conventions.

## Convention de signe (à connaître)

**θ̂_j > 0 ⇔ sym2 leads sym1.** **θ̂_j < 0 ⇔ sym1 leads sym2.**

Locked by `scripts/audit_sign_convention.py` (synthetic test in both directions). The
codebase's HY estimator shifts series-2 intervals by `+τ`; alignment when sym1 leads
requires `τ < 0`, hence the inverted convention vs the paper's narrative.

---

## Quickstart

```bash
# Setup (Windows)
.\setup.bat                           # or: ./setup.sh on Linux/macOS

# Or manually
python -m venv .venv
.venv\Scripts\activate                # source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
pip install -e .

# Verify
.venv\Scripts\python.exe -m pytest -q # 31 tests, all pass

# Sign-convention audit (sanity check before any interpretation)
python scripts/audit_sign_convention.py

# Run the main empirical experiment (BTC vs ETH, calm week)
python scripts/run_empirical.py \
    --start 2026-04-13 --end 2026-04-19 \
    --sym1 BTCUSDT --sym2 ETHUSDT --market spot \
    --delta-N 0.05 --jmax 8 --grid-half 400

# Run on FOMC stress week
python scripts/run_empirical.py \
    --start 2026-04-27 --end 2026-05-03 \
    --sym1 BTCUSDT --sym2 ETHUSDT --market spot \
    --delta-N 0.05 --jmax 8 --grid-half 400

# Run a LOBSTER midpoint pair experiment
python scripts/run_lobster_pair.py \
    --ticker1 SPY --ticker2 AAPL --level 30 \
    --delta-N 0.001 --jmax 8 --grid-half 200

# Bootstrap CI on an existing run
python scripts/run_bootstrap.py \
    --run 2026-04-13to19_btc_eth_spot_full168h \
    --hours-cap 24 --B 100 --block-s 300

# Notebook walkthrough (pedagogical)
jupyter lab notebooks/01_walkthrough.ipynb

# Notebook narrative (soutenance-ready)
jupyter lab notebooks/02_main_narrative.ipynb
```

---

## Key results (selected runs)

### Run 1 : BTC vs ETH, calm week (13-19 Apr 2026, Binance spot)
5.7M ticks, Δ_N = 50 ms, j_max = 8, db10 wavelet.

| j | period (s) | θ̂ (s) | Interpretation |
|---|---|---|---|
| 1-3 | 0.1 - 0.8 | 0.00 | No lead-lag (HF efficiency) |
| 4 | 0.8 - 1.6 | +0.05 | ETH leads BTC 50 ms |
| 5 | 1.6 - 3.2 | +0.05 | ETH leads BTC 50 ms |
| 6 | 3.2 - 6.4 | +0.10 | ETH leads BTC 100 ms |
| 7 | 6.4 - 12.8 | +0.15 | ETH leads BTC 150 ms |
| 8 | 12.8 - 25.6 | +0.25 | ETH leads BTC 250 ms |

→ See `outputs/2026-04-13to19_btc_eth_spot_full168h/heatmap.png`.

### Run 2 : HRY baseline, same data
θ̂ = 0.00 s. The single-scale estimator collapses all scales into one and **misses the multi-scale signal**.
→ See `outputs/comparison_hk_vs_hry.png`.

### Run 3 : FOMC stress week (27 Apr - 3 May 2026)
Same qualitative pattern. Lags **+40% larger at j=8** (250 → 350 ms).
→ See `outputs/comparison_calm_vs_fomc.png`.

### Run 4 : LOBSTER AAPL vs MSFT, 21 Jun 2012, NYSE midpoints
Replication of HK's setup (quote midpoints). 1 day only, noisier than crypto runs.
At j=8 (256-512 ms): AAPL leads MSFT by ~70 ms.

### Run 5 : LOBSTER SPY vs AAPL, 21 Jun 2012, NYSE midpoints
Equity analogue with an aggregate ETF vs a large constituent, level-30 LOBSTER
sample over the first hour. SPY has ~1.03M events vs AAPL ~81k. This run is
resolution-sensitive: with Δ_N = 10 ms, AAPL leads SPY at scales j >= 2; with
Δ_N = 1 ms, SPY leads AAPL over sub-0.5s horizons by roughly 38-119 ms. Use as
an illustration that direction can depend on scale/resolution, not as a single
robust economic direction.

### Run 6 : SOL vs AVAX, same calm week
Strong signal, with a larger lag than BTC/ETH. AVAX has 9× fewer ticks than SOL
yet appears to lead, so the result is presented as an empirical regularity that
requires caution. A controlled tick-density simulation suggests this is not a
pure estimator artefact at the observed tick counts.

---

## Project layout

> See also: **`DATA_LAYOUT.md`** for the `data/` tree and source URLs, and
> **`OUTPUTS_INDEX.md`** for a per-run map (what each `outputs/<run>/` folder represents).

```
hk_leadlag_replication/
├── pyproject.toml              # Package definition (Pydantic, Numba, pywt, pyarrow, …)
├── requirements.txt
├── setup.bat / setup.sh        # One-shot venv + install + tests
├── main.py                     # CLI entry point for YAML-driven experiments
├── main.ipynb                  # Top-level orchestrator notebook
├── DATA_LAYOUT.md              # Expected data/ tree + source URLs + regen commands
├── OUTPUTS_INDEX.md            # Per-run map (config + finding for each outputs/<run>/)
│
├── hk_leadlag/                 # Package source
│   ├── base.py                 # ABCs: BaseLeadLagEstimator, BaseSimulator, NonSyncSeries
│   ├── config.py               # Pydantic ExperimentConfig + sub-configs
│   ├── wavelet.py              # Daubechies filters, autocorrelation wavelet
│   ├── estimators/
│   │   ├── hayashi_yoshida.py  # Numba-jitted HY cross-cov
│   │   ├── wavelet_leadlag.py  # Core HK estimator
│   │   ├── hry.py              # Single-scale baseline
│   │   └── dobrev_schaumburg.py
│   ├── simulation/
│   │   ├── brownian.py         # Bivariate Brownian via circulant embedding
│   │   ├── heston.py
│   │   └── sampling.py         # Regular + Lo-MacKinlay sampling
│   ├── data/
│   │   ├── binance.py          # Binance aggTrades (auto-detect μs/ms)
│   │   ├── lobster.py          # LOBSTER message+orderbook → midpoints
│   │   ├── csv_loader.py
│   │   └── preprocess.py
│   ├── analysis/
│   │   ├── monte_carlo.py
│   │   ├── bootstrap.py        # Moving-block bootstrap on ticks
│   │   ├── multiple_testing.py # Romano-Wolf-style and BH p-value adjustments
│   │   ├── event_study.py
│   │   ├── experiment.py
│   │   └── artifacts.py        # ArtifactStore (reproducible experiment outputs)
│   └── viz/plots.py            # LeadLagPlotter: heatmap, contrast, scalogram, MC boxplots
│
├── scripts/
│   ├── run_empirical.py        # CLI: load data → fit → save artefacts
│   ├── run_bootstrap.py        # CLI: bootstrap CI on an existing run
│   ├── audit_sign_convention.py  # Sign-convention sanity test
│   ├── run_lobster_pair.py     # CLI: LOBSTER midpoint pair experiment
│   └── download_lobster_sample.py
│
├── notebooks/
│   ├── 01_walkthrough.ipynb    # Pedagogical step-by-step
│   └── 02_main_narrative.ipynb # Defense-ready story
│
├── tests/                      # pytest (31 tests)
├── configs/                    # YAML experiment configs
└── outputs/                    # Generated artefacts
    └── <YYYY-MM-DD>_<sym1>_<sym2>_<market>_…/
        ├── config.json         # Experiment config (always tracked in git)
        ├── metadata.json       # Timings, n_ticks, θ̂ (tracked)
        ├── series_stats.json   # Tick density stats (tracked)
        ├── summary.csv         # One row per scale (tracked)
        ├── result.pkl          # Full LeadLagResult (NOT tracked, regenerable)
        ├── heatmap.png         # NOT tracked (regenerable)
        ├── contrast.png        # NOT tracked
        ├── data_overview.png   # NOT tracked
        └── log.txt             # stdout capture
```

---

## Citation

This is an independent academic implementation. The paper:

> Hayashi T. and Koike Y. (2020). "Multi-scale analysis of lead-lag relationships in
> high-frequency financial markets." arXiv:1708.03992. SIAM J. Financial Math.
> companion paper: arXiv:1612.01232.

Companion course material (linked in `notes.md`):

> M. Garcin (2017-2018). *Aspects multifréquentiels du risque*. ESILV course slides.

---

## License

Educational replication. Original paper © Hayashi & Koike, CC-BY arXiv.
LOBSTER sample data © TU Berlin, redistributed under the terms of their academic licence.
Binance Data Vision is a public CDN; no licence issue for academic redistribution.
