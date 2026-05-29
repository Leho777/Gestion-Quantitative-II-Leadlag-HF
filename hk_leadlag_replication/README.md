# HK-LeadLag: estimator replication + crypto adaptation of Hayashi and Koike (2020)

**Multi-scale analysis of lead-lag relationships in high-frequency financial markets.**

Independent Python implementation. M2 272 Quantitative Management II, Paris-Dauphine PSL.

> **Scope (calibrated).** We replicate the **estimator** of Hayashi-Koike (Hayashi-Yoshida
> non-synchronous covariance, then Daubechies wavelet convolution, then argmax per scale) and
> adapt it to crypto cross-venue and cross-product data. We do **not** replicate HK's empirical
> protocol literally (NASDAQ vs BATS quote micro-prices, 108 stocks over 21 days, ~0.1 ms
> resolution, SIP/participant timestamps). Concretely:
> - **Price proxy.** aggTrades on crypto (Binance Vision has no historical bookTicker) instead of
>   quote micro-prices. LOBSTER (NASDAQ only) is used as a quote-midpoint sanity check, not as a
>   NASDAQ vs BATS replication.
> - **Resolution.** Delta_N = 50 ms on crypto vs ~0.1 ms in HK (crypto inter-tick median ~ 30 ms
>   on Binance BTC spot).
> - **Timestamping.** Exchange-API timestamps, no SIP consolidation. The measured lag can mix
>   price discovery and reporting conventions.
> - **Panel.** A handful of cross-venue / cross-product cases vs 108 x 21 = 2268 ticker-days in HK.
> - **Estimator comparison.** We benchmark HK against HRY (single-scale). The paper also uses
>   Dobrev-Schaumburg, which we keep only in simplified form.

## Overview

Modular implementation of the wavelet-based, scale-by-scale lead-lag estimator of Hayashi and
Koike (2020) for non-synchronously sampled high-frequency data. Contents:

- Multi-scale `WaveletLeadLagEstimator` and single-scale `HRYEstimator` baseline.
- Numba-accelerated Hayashi-Yoshida cross-covariance (about 88x faster than pure Python).
- Bivariate Brownian + Heston + Lo-MacKinlay sampling for Monte Carlo validation.
- Loaders for Binance Vision aggTrades, Bybit public archive, Kraken REST, and LOBSTER midpoints.
- Per-day distribution of theta_hat_j across UTC days (analogue of HK's Figure 1).
- Artifact store for reproducible experiment outputs.

## Sign convention

**theta_hat_j > 0 means series 2 leads series 1. theta_hat_j < 0 means series 1 leads series 2.**

Locked by `scripts/audit_sign_convention.py` (synthetic test in both directions). The HY estimator
shifts series-2 intervals by `+tau`; alignment when series 1 leads requires `tau < 0`, hence the
convention is inverted relative to the paper's narrative.

---

## Quickstart

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate                 # source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
pip install -e .

# Tests
.venv\Scripts\python.exe -m pytest -q  # 27 tests

# Sign-convention audit (run before interpreting anything)
python scripts/audit_sign_convention.py

# Core result: same-asset cross-venue (BTC on Binance vs Kraken)
python scripts/run_cross_exchange_kraken.py \
    --start 2026-04-01 --end 2026-04-30 \
    --binance-symbol BTCUSDT --kraken-pair XBTUSD \
    --name 2026-04_full_month_btc_binance_vs_kraken

# Cross-asset BTC vs ETH (exploratory extension)
python scripts/run_empirical.py \
    --start 2026-04-13 --end 2026-04-19 \
    --sym1 BTCUSDT --sym2 ETHUSDT --market spot \
    --delta-N 0.05 --jmax 8 --grid-half 400

# Per-day distribution over a month (analogue of HK Figure 1)
python scripts/run_perday_distribution.py \
    --mode binance_vs_kraken --symbol BTCUSDT \
    --start 2026-01-01 --end 2026-01-31

# LOBSTER midpoint pair (equity sanity check)
python scripts/run_lobster_pair.py \
    --ticker1 AAPL --ticker2 MSFT --level 10 \
    --delta-N 0.001 --jmax 8 --grid-half 200

# Narrative notebook (defense)
jupyter lab notebooks/03_final_narrative.ipynb
```

---

## Key results

The core finding is the **same-asset cross-venue** lead-lag, the closest available analogue of HK's
NASDAQ-vs-BATS setup. Cross-asset (BTC vs ETH) is kept as an exploratory extension because its
direction is sensitive to the price proxy.

### Core: BTC, Binance vs Kraken (April 2026, 30 days, Delta_N = 50 ms, db10)

| j | period (s) | theta_hat (s) | reading |
|---|---|---|---|
| 1 | 0.1 - 0.2 | -0.05 | Binance leads Kraken |
| 4 | 0.8 - 1.6 | -0.30 | Binance leads Kraken |
| 6 | 3.2 - 6.4 | -0.80 | Binance leads Kraken |
| 8 | 12.8 - 25.6 | **-1.95** | Binance leads Kraken |

Monotone profile, Binance ahead at every scale. Direction is invariant across January, February
and April 2026; amplitude is regime-dependent (February ~ -1.15 s vs ~ -2.0 s in January/April).
See `../prez/figures/heatmap_btc_binance_vs_kraken_full_month.png`.

### HK vs HRY (same BTC/ETH 168h data)

HK gives a structured multi-scale profile (0, 0, 0, +50, +50, +100, +150, +250 ms over j=1..8);
the single-scale HRY baseline returns ~0. HRY collapses the scales and misses the signal.
See `../prez/figures/comparison_hk_vs_hry.png`.

### Cross-product, same exchange (30 days)

- BTC and ETH, spot vs USDT-margined perpetual: perp leads spot (+350 ms BTC, +400 ms ETH at j=8).
- BTC, USDT-margined vs coin-margined perpetual: USDT-M leads (-1.70 s at j=8).
- BTC, Binance vs Bybit: quasi-synchronous (two HFT-heavy venues).

### Cross-asset BTC vs ETH (exploratory, fragile)

On aggTrades, ETH appears to lead BTC at coarse scales (+50 to +250 ms). The direction is
**sensitive to the price proxy** (a cleaner synthetic midpoint or micro-price can flip it), so this
is presented as a secondary result, not a core finding.

### LOBSTER AAPL vs MSFT (21 June 2012, NASDAQ midpoints)

Quote-midpoint sanity check on equity (one day, noisier than crypto). At j=8, AAPL leads MSFT by
about 70 ms. Confirms the estimator runs on quote midpoints, not only on crypto trades.

### Monte Carlo

Reduced-scale reproduction of the paper's Section 5 design (constant and Heston volatility,
Lo-MacKinlay non-synchronicity, comparison against previous-tick interpolation). Fine scales are
recovered and the non-synchronous estimator dominates interpolation under strong asynchrony. This
validates the design, not the full Tables 2-3 (which need the paper's n = 30000, 1000 paths).

---

## Project layout

```
hk_leadlag_replication/
├── pyproject.toml              # package definition
├── requirements.txt
│
├── hk_leadlag/                 # package source
│   ├── base.py                 # ABCs: BaseLeadLagEstimator, NonSyncSeries, ...
│   ├── config.py               # Pydantic ExperimentConfig + sub-configs
│   ├── wavelet.py              # Daubechies filters, autocorrelation wavelet, transfer function
│   ├── estimators/
│   │   ├── hayashi_yoshida.py  # Numba-jitted HY cross-covariance
│   │   ├── wavelet_leadlag.py  # core HK estimator
│   │   ├── hry.py              # single-scale baseline
│   │   └── dobrev_schaumburg.py
│   ├── simulation/
│   │   ├── brownian.py         # bivariate Brownian via circulant embedding
│   │   ├── heston.py
│   │   └── sampling.py         # regular + Lo-MacKinlay sampling
│   ├── data/
│   │   ├── binance.py          # Binance aggTrades (auto-detect us/ms)
│   │   ├── binance_midpoint.py # synthetic midpoint / micro-price from aggTrades
│   │   ├── bybit.py            # Bybit public archive
│   │   ├── kraken.py           # Kraken REST trades
│   │   ├── lobster.py          # LOBSTER message + orderbook to midpoints
│   │   └── preprocess.py
│   ├── analysis/
│   │   ├── monte_carlo.py
│   │   ├── event_study.py
│   │   ├── experiment.py       # YAML-config experiment runner
│   │   └── artifacts.py        # ArtifactStore (reproducible outputs)
│   └── viz/plots.py            # LeadLagPlotter: heatmap, contrast, scalogram, MC boxplots
│
├── scripts/
│   ├── run_empirical.py            # cross-asset (config-driven)
│   ├── run_cross_exchange.py       # Binance vs Bybit (same asset)
│   ├── run_cross_exchange_kraken.py# Binance vs Kraken (same asset, core)
│   ├── run_cross_market_btc.py     # spot vs perp (same asset, --symbol generic)
│   ├── run_usdt_vs_coin_perp.py    # USDT-M vs COIN-M perp
│   ├── run_perday_distribution.py  # per-day theta_hat distribution + histograms
│   ├── run_monte_carlo_diagnostic.py
│   ├── run_lobster_pair.py         # LOBSTER midpoint pair
│   ├── audit_sign_convention.py    # sign-convention sanity test
│   ├── verify_remark2_filter_invariance.py  # Remark 2 check (db10 vs sym10)
│   └── download_lobster_sample.py
│
├── notebooks/
│   └── 03_final_narrative.ipynb# defense narrative (from-artifacts mode)
│
├── tests/                      # pytest (27 tests)
├── configs/                    # YAML experiment configs
└── outputs/                    # generated artefacts
    └── <run>/
        ├── config.json         # experiment config (tracked)
        ├── metadata.json       # timings, n_ticks, theta_hat (tracked)
        ├── series_stats.json   # tick-density stats (tracked)
        ├── summary.csv         # one row per scale (tracked)
        ├── result.pkl          # full LeadLagResult (not tracked, regenerable)
        ├── heatmap.png         # not tracked (regenerable)
        └── log.txt             # stdout capture
```

---

## Citation

Independent academic implementation. Paper:

> Hayashi T. and Koike Y. (2020). Multi-scale analysis of lead-lag relationships in high-frequency
> financial markets. arXiv:1708.03992. Companion paper: arXiv:1612.01232.

Course material (linked in the project notes):

> M. Garcin (2017-2018). Aspects multifrequentiels du risque. ESILV course slides.

---

## License

Educational replication. Original paper (c) Hayashi and Koike, arXiv. LOBSTER sample data (c) TU
Berlin, redistributed under their academic licence. Binance Data Vision is a public CDN.
