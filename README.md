# Multi-scale lead-lag à haute fréquence : réplication de Hayashi & Koike (2020)

Réplication méthodologique et extensions de :

> Hayashi T. & Koike Y. (2020). *Multi-scale analysis of lead-lag relationships
> in high-frequency financial markets.* arXiv:1708.03992.

**Master 2 272, Ingénierie Économique et Financière, parcours Finance Quantitative,
Université Paris-Dauphine PSL.**
Auteurs : Léo Renault, Théo Verdelhan, Ziad El Arari.

---

## Portée du travail 

Nous répliquons **l'estimateur** de Hayashi-Koike (covariance non-synchrone de
Hayashi-Yoshida, convolution par autocorrélation d'ondelette de Daubechies,
argmax par échelle) et nous l'**adaptons** à des données haute fréquence
2025-2026 (crypto cross-venue/cross-product et equity LOBSTER).

Nous ne reproduisons **pas** le protocole empirique exact du papier (NASDAQ vs
BATS, août 2015, 108 actions, micro-prices de quotes, résolution ~0.1 ms),
faute d'accès libre aux quotes historiques correspondantes. Le détail des écarts
(prix, résolution temporelle, timestamping, panel) est documenté dans le rapport,
section « Portée de la réplication ».

---

## Structure du dépôt

```
.
├── hk_leadlag_replication/   # Code : package OOP + scripts + tests
│   ├── hk_leadlag/           #   estimateurs (HK, HRY, DS), wavelet, loaders data
│   ├── scripts/              #   CLIs : run_empirical, run_bootstrap, cross-venue...
│   ├── tests/                #   suite pytest (31 tests)
│   ├── notebooks/            #   walkthrough pédagogique + notebook narratif
│   ├── outputs/              #   artefacts de repro (config, summary, stats par run)
│   ├── README.md             #   doc détaillée du package
│   ├── DATA_LAYOUT.md        #   structure attendue de data/ + sources
│   └── OUTPUTS_INDEX.md      #   mapping run → finding pour chaque expérience
│
├── rendu/                    # Rapport LaTeX (article de synthèse)
│   └── main.tex
│
├── prez/                     # Support de présentation (Beamer) + figures
│   ├── slide.tex
│   └── figures/
│
└── papier_retenu_3_Hayashi_Koike/  # Article original + décorticage
```

---

## Convention de signe (piège à connaître)

Dans ce codebase :
- **θ̂ⱼ > 0  ⇔  série 2 mène série 1**
- **θ̂ⱼ < 0  ⇔  série 1 mène série 2**

Convention inversée par rapport à la lecture naïve. Verrouillée par un audit
synthétique reproductible (`scripts/audit_sign_convention.py`) et un test pytest.

---

## Démarrage rapide

```bash
cd hk_leadlag_replication

# Setup (crée le venv, installe les dépendances, lance les tests)
./setup.sh          # Linux / macOS
# ou
setup.bat           # Windows

# Vérifier la convention de signe
python scripts/audit_sign_convention.py

# Lancer l'expérience principale (BTC vs ETH, semaine calme)
python scripts/run_empirical.py \
    --sym1 BTCUSDT --sym2 ETHUSDT \
    --start 2026-04-13 --end 2026-04-19 \
    --venue binance --market spot
```

Python 3.11. Dépendances dans `hk_leadlag_replication/requirements.txt`.

---

## Données

Les données brutes (≈ 1.5 GB de parquet) ne sont **pas** versionnées (cf.
`.gitignore`). Elles sont **re-téléchargeables** automatiquement par les loaders
(Binance Vision, Bybit, Kraken ; voir `DATA_LAYOUT.md`). Aucune clé API n'est
requise : tous les loaders utilisent des archives ou endpoints publics.

---

## Résultats principaux (résumé)

1. **HK révèle ce que HRY masque** : sur BTC/ETH, l'estimateur single-scale HRY
   renvoie θ̂ ≈ 0 alors que HK montre un profil multi-échelle structuré.
2. **Cross-venue same-asset (le cœur)** : sur BTC, Binance précède Kraken de
   façon monotone, jusqu'à ≈ −1.95 s à l'échelle la plus grossière (30 jours
   d'avril 2026), avec bootstrap p_RW = 0.020. Pattern confirmé sur ETH.
3. **Fragilité au proxy de prix** : sur BTC/ETH cross-asset, trois reconstructions
   de prix (aggTrades, synth-midpoint, synth-microprice) donnent trois directions
   différentes → finding à présenter avec ce caveat.
4. **Robustesse temporelle** : pattern qualitatif stable en régime actif, mais
   magnitude dépendante du régime de marché.

Détails dans le rapport (`rendu/main.tex`) et `hk_leadlag_replication/OUTPUTS_INDEX.md`.

---

## Note académique

Travail universitaire de réplication. L'article original est © Hayashi & Koike,
diffusé sur arXiv. Le code est une implémentation indépendante (le code des
auteurs n'a pas été utilisé).
