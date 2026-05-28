# `outputs/` Index : mapping run_name → ce qu'il représente

Pour chaque run : période, setup, paramètres, finding principal.
**Convention de signe** : `θ̂_j < 0 ⇔ sym1 leade sym2` ; `θ̂_j > 0 ⇔ sym2 leade sym1`. Voir `scripts/audit_sign_convention.py`.

## Runs principaux

| Run | Période | Setup | Δ_N | jmax | Finding |
|---|---|---|---|---|---|
| `2026-04-13to19_btc_eth_spot_full168h` | 13-19 avril 2026 | BTC vs ETH spot Binance, aggTrades | 50 ms | 8 | ETH leade BTC : +50 ms (j=4) → +250 ms (j=8) |
| `2026-04-13to19_btc_eth_spot_hry_baseline` | idem | HRY single-scale | 50 ms | (1) | θ̂ ≈ 0 → HRY ne capture pas le signal |
| `2026-04-27to05-03_btc_eth_spot_fomc_week` | 27 avr - 3 mai (FOMC) | idem | 50 ms | 8 | Même structure, lags +40% (j=8 : 350 ms) |
| `2026-04-13to19_btc_binance_vs_kraken` | 13-19 avril | BTC spot, Binance vs Kraken | 50 ms | 8 | **Binance leade Kraken**, -50 ms → **-1.55 s** à j=8 (full-week) ; bootstrap p_RW=0.020 à j=5..8 (sur 24h cap, θ̂ subset = -2.10 s à j=8) |

## Runs cross-venue same-asset (panel)

| Run | Setup | Δ_N | Finding |
|---|---|---|---|
| `2026-04-13to19_btc_spot_vs_perp` | BTC spot Binance vs USDT-M perp Binance | 50 ms | Perp leade spot : 50 → 350 ms |
| `2026-04-13to19_eth_spot_vs_perp` | mirror ETH | 50 ms | Idem (50 → 350 ms) |
| `2026-04-13to19_btc_usdt_perp_vs_coin_perp` | BTC USDT-M perp vs COIN-M perp | 50 ms | USDT-M leade COIN-M, jusqu'à -1.55 s |
| `2026-04-13to19_btc_binance_vs_bybit` | BTC spot Binance vs Bybit | 50 ms | Quasi-synchrone (~150 ms à j=8) |

## Runs de robustesse temporelle BTC/ETH

| Run | Période | Finding |
|---|---|---|
| `2026-02-09to15_btc_eth_spot_week_feb` | 9-15 fév 2026 | θ̂ ≈ 0 partout (régime ≈ calme) |
| `2026-03-16to22_btc_eth_spot_week_mar_fomc` | 16-22 mars (mars FOMC) | +50 à +200 ms (intermédiaire) |
| `2026-04-27to05-03_btc_eth_spot_fomc_week` | avr FOMC | +50 à +350 ms |

→ Pattern régime-dépendant. 

## Runs de fragilité au proxy (BTC/ETH même semaine)

| Run | Proxy de prix | Direction | Magnitude à j=8 |
|---|---|---|---|
| `2026-04-13to19_btc_eth_spot_full168h`            | aggTrades last-trade | ETH leade BTC | +250 ms |
| `2026-04-13to19_btc_eth_synthmid_full168h`        | synth-midpoint (is_buyer_maker)| BTC leade ETH | -100 ms |
| `2026-04-13to19_btc_eth_synthmicroprice_full168h` | synth-microprice (Stoll, trade-volume-weighted)| signal quasi-nul | -0.10 s |

→ **3 proxies → 3 directions** : le signal cross-asset BTC/ETH dépend du proxy. Voir notes §21.

## Runs multi-paires crypto (test de généralité)

| Run | Pair | Finding |
|---|---|---|
| `2026-04-13to19_btc_eth_spot_full168h`     | BTC/ETH | ETH leade BTC, magnitudes modérées |
| `2026-04-13to19_sol_avax_spot_third_pair`  | SOL/AVAX | AVAX leade SOL, magnitudes grandes (jusqu'à 1.15s) |
| `2026-04-13to19_uni_aave_spot_defi`        | UNI/AAVE | UNI leade AAVE, monotone (-50 → -700 ms) |

→ Régularité empirique : **asset moins-quoté apparaît leader**. Test contrôlé d'asymétrie de tick density → l'estimateur reste robuste dans la plage observée.

## Runs LOBSTER (equity midpoint, sanity check méthodologique)

| Run | Tickers | Δ_N | Finding |
|---|---|---|---|
| `2012-06-21_aapl_msft_lobster_lvl10_full` | AAPL vs MSFT, 10 niveaux | 1 ms | AAPL leade MSFT à j=8 (~70 ms) |
| `2012-06-21_aapl_msft_lobster_dN10ms`     | idem, Δ_N relâché | 10 ms | Signal plus net |
| `2012-06-21_spy_aapl_lobster_lvl30_full`  | SPY vs AAPL, 30 niveaux | 1 ms | AAPL leade SPY (mais sign-flip si Δ_N=10 ms < 0.5s) |
| `2012-06-21_spy_aapl_lobster_lvl30`       | idem, Δ_N=10 ms | 10 ms | sensibilité à la résolution |

⚠ HK utilise NASDAQ vs BATS pour le **même** ticker. AAPL vs MSFT et SPY vs AAPL sont des paires cross-asset (pas une réplication stricte). Ces runs servent à **valider que l'estimateur tourne sur quotes-midpoint** (le setup HK), pas à reproduire le résultat empirique du papier.

## Runs grande échelle (avril 2026, 30 jours, 6 setups)

| Run | Setup | θ̂(j=8) (s) | Direction | contrast j=8 |
|---|---|---:|---|---:|
| `2026-04_full_month_btc_binance_vs_kraken`        | BTC, Binance vs Kraken      | **-1.95** | Binance leade Kraken | 0.124 |
| `2026-04_full_month_eth_binance_vs_kraken`        | ETH, Binance vs Kraken      | **-0.30** (peak -0.40 à j=6-7) | Binance leade Kraken (non-monotone) | **0.485** |
| `2026-04_full_month_btc_usdt_perp_vs_coin_perp`   | BTC, USDT-M perp vs COIN-M perp | **-1.70** | USDT-M leade COIN-M | 0.053 |
| `2026-04_full_month_btc_spot_vs_perp`             | BTC, spot vs USDT-M perp    | **+0.35** | perp leade spot | 0.018 |
| `2026-04_full_month_eth_spot_vs_perp`             | ETH, spot vs USDT-M perp    | **+0.40** | perp leade spot | 0.036 |
| `2026-04_full_month_btc_binance_vs_bybit`         | BTC, Binance vs Bybit       | **+0.20** | Bybit leade Binance (faible) | 0.037 |

**Comparaison 7 jours vs 30 jours** : magnitudes globalement stables (différences < 20% par échelle). Le pattern qualitatif est conservé. Sur Binance vs Bybit, on passe de quasi-synchrone (7d) à +200 ms (30d), léger biais Bybit qui apparaît avec plus de sample.

→ Panel : `prez/figures/comparison_panel_full_month.png`.

## Runs obsolètes / pour debug (à ne pas commiter)

| Run | Pourquoi obsolète |
|---|---|
| `2026-04-13to19_btc_eth_synthmid` (1 jour, jmax=6) | Pre-prod du synthmid_full168h. Conservé pour traçabilité, mais non utilisé en défense. |

## Fichiers comparison-plot à la racine de `outputs/`

| Fichier | Contenu |
|---|---|
| `comparison_hk_vs_hry.png`              | HK multi-scale vs HRY single-scale (slide-killer) |
| `comparison_calm_vs_fomc.png`           | BTC/ETH calm week vs FOMC week (panel) |
| `comparison_calm_vs_fomc_with_CI.png`   | idem + 95% CI bootstrap + p_RW |
| `comparison_multipair_crypto.png`       | BTC/ETH + SOL/AVAX + UNI/AAVE |
| `comparison_panel_btc_eth_5runs.png`    | Robustesse temporelle + proxy BTC/ETH |
| `comparison_panel_cross_venue.png`      | 5 setups cross-venue |
| `comparison_proxies_btc_eth.png`        | 3 proxies (aggTrades, synth-mid, synth-microprice) côte à côte |
| `btc_eth_6h_result.pkl` etc.            | Cache exploratoire 6h (à supprimer si jamais on rebuild propre) |

## Quoi commit / quoi pas

| Type de fichier | Commit ? |
|---|---|
| `config.json`, `metadata.json`, `series_stats.json`, `summary.csv`, `bootstrap.json` | **Oui** (petit, reproductibilité) |
| `result.pkl`, `bootstrap_dist.pkl`, `*.png`, `*.npz`, `log.txt` | Non (whitelist dans `.gitignore`) |

Voir `.gitignore` pour les règles exactes.
