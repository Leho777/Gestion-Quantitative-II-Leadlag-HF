# Data layout : `hk_leadlag_replication/data/`

Documentation **versionnée** de la structure attendue de `data/` (le contenu lui-même est gitignored et re-téléchargeable par les loaders).

```
data/
├── raw/                        # downloads bruts, jamais commit
│   ├── binance/
│   │   ├── spot/{SYMBOL}/{SYMBOL}-aggTrades-{YYYY-MM-DD}.parquet
│   │   ├── futures/{SYMBOL}/{SYMBOL}-aggTrades-{YYYY-MM-DD}.parquet   # USDT-M perp
│   │   └── coinm/{SYMBOL}/{SYMBOL}-{YYYY-MM-DD}.parquet               # COIN-M perp
│   ├── bybit/spot/{SYMBOL}/{SYMBOL}-{YYYY-MM-DD}.parquet
│   ├── kraken/{PAIR}/{PAIR}-{YYYY-MM-DD}-{YYYY-MM-DD}.parquet
│   └── live_collect/                                  # polling REST live
│       ├── hyperliquid_l2/{YYYY-MM-DDTHH}.parquet      # L2 book 20 levels
│       ├── binance_bookticker/{YYYY-MM-DDTHH}.parquet  # best bid/ask
│       ├── bybit_trades/{YYYY-MM-DDTHH}.parquet
│       └── dydx_trades/{YYYY-MM-DDTHH}.parquet
├── equity/                     # LOBSTER zips (gitignored; à télécharger)
│   └── LOBSTER_SampleFile_{TICKER}_{YYYY-MM-DD}_{LEVELS}.zip
├── processed/                  # npz cachés, gitignored
│   └── {RUN_NAME}.npz          # arrays t1, p1, t2, p2
└── sample/                     # petit échantillon committable pour tests (whitelisted)
```

## Sources et formats

| Venue | Format raw | Source | Loader |
|---|---|---|---|
| Binance spot       | parquet, aggTrades, 8 cols (no header), timestamps µs post-2024 | `https://data.binance.vision/data/spot/daily/aggTrades/{SYMBOL}/` | `hk_leadlag/data/binance.py::BinanceLoader` |
| Binance USDT-M perp| parquet, aggTrades, 7 cols + header | `https://data.binance.vision/data/futures/um/daily/aggTrades/{SYMBOL}/` | idem (branch sur `market='futures'`) |
| Binance COIN-M perp| parquet, aggTrades | `https://data.binance.vision/data/futures/cm/daily/aggTrades/{SYMBOL}/` | idem (branch sur `market='coinm'`) |
| Bybit spot         | parquet, trades CSV.gz | `https://public.bybit.com/spot/{SYMBOL}/` | `hk_leadlag/data/bybit.py::BybitTradesLoader` |
| Kraken             | parquet, REST paginated `since` | `https://api.kraken.com/0/public/Trades` | `hk_leadlag/data/kraken.py::KrakenTradesLoader` |
| LOBSTER equity     | zip → CSV (message + orderbook) | `https://lobsterdata.com/info/DataSamples.php` | `hk_leadlag/data/lobster.py::LobsterLoader` |
| Hyperliquid (live) | parquet snapshots L2 | REST `https://api.hyperliquid.xyz/info` (l2Book) | `scripts/collect_live_data.py` |
| dYdX v4 (live)     | parquet snapshots trades | REST `https://indexer.dydx.trade/v4/trades/...` | idem |

## Schémas

### Binance spot aggTrades (`agg_trade_id, price, qty, first_trade_id, last_trade_id, timestamp, is_buyer_maker, is_best_match`)
- `timestamp` : entiers ; **µs** depuis 2024, **ms** avant. Le loader détecte la magnitude.
- `is_buyer_maker = True` ⇒ trade exécuté contre un ordre passif d'achat ⇒ **prix au bid**.
- Reconstruction synth-mid : `mid = price` si `not is_buyer_maker` else `price` ; `mid_smooth = ema(mid)`. Voir `binance_midpoint.py`.

### Binance futures aggTrades (header présent, 7 cols, `is_best_match` absent, booleens stringifiés)
Le loader branche sur `market`.

### Bybit trades
Schema CSV : `timestamp, symbol, side, size, price, tickDirection, ...`.

### Kraken trades
Schema REST : `[price, volume, time, side, ord_type, misc]`.

### LOBSTER
- `*_message_*.csv` : `time, event_type, order_id, size, price, direction`.
- `*_orderbook_*.csv` : snapshots prix/taille en colonnes bid/ask × niveaux.

### Hyperliquid L2 snapshot (poll)
```python
{
    "ts": <unix_seconds>,
    "bid_px": float, "bid_sz": float,
    "ask_px": float, "ask_sz": float,
    "mid": (ask+bid)/2,
    "micro": (ask*vbid + bid*vask)/(vbid+vask),   # Stoll inversion
    "n_bid_levels": int, "n_ask_levels": int,
}
```

## Commandes de régénération

### Crypto Binance : week calm BTC/ETH (run principal)
```bash
python scripts/run_empirical.py \
    --sym1 BTCUSDT --sym2 ETHUSDT \
    --start 2026-04-13 --end 2026-04-19 \
    --venue binance --market spot \
    --run-name 2026-04-13to19_btc_eth_spot_full168h
```

### Cross-venue Binance vs Kraken (BTC)
```bash
python scripts/run_cross_exchange.py \
    --symbol BTC --start 2026-04-13 --end 2026-04-19 \
    --venue1 binance --venue2 kraken \
    --run-name 2026-04-13to19_btc_binance_vs_kraken
```

### Bootstrap CI sur un run existant
```bash
python scripts/run_bootstrap.py \
    --run 2026-04-13to19_btc_binance_vs_kraken \
    --hours-cap 24 --B 50 --block-s 300
```

### Live data collector (polling Hyperliquid + autres)
```bash
python scripts/collect_live_data.py --duration-h 12 --poll-s 1.0
```

## Politique git

Tout `data/` est gitignored sauf :
- `data/sample/` (whitelisted), petits échantillons pour les tests
- le présent `DATA_LAYOUT.md`

`outputs/` : seuls les petits fichiers de reproductibilité sont commit (`config.json`, `metadata.json`, `series_stats.json`, `summary.csv`, `bootstrap.json`). Les `.pkl`, `.png`, `.npz` sont ignorés. Voir `.gitignore`.
