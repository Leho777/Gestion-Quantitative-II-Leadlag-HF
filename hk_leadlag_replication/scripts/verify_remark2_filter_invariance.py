"""Remark 2 (Hayashi & Koike 2020) : a longueur L fixee, l'autocorrelation
wavelet Psi_j(l) ne depend que de la power transfer function du filtre, pas de
sa phase. Donc extremal-phase (dbN) et least-asymmetric (symN) du meme ordre
donnent le meme Psi_j, donc le meme estimateur theta_hat_j.

Verification numerique : on compare Psi_j pour db10 et sym10 (tous deux L=20),
niveaux j=1..8.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hk_leadlag.wavelet import DaubechiesFilter, autocorrelation_wavelet


def main() -> int:
    db = DaubechiesFilter("db10")
    sym = DaubechiesFilter("sym10")
    print(f"db10  : L = {db.L}")
    print(f"sym10 : L = {sym.L}")
    print()
    print(f"{'j':>2} {'len Psi_j':>10} {'max|db-sym|':>14}")
    worst = 0.0
    for j in range(1, 9):
        psi_db = autocorrelation_wavelet(db, j)
        psi_sym = autocorrelation_wavelet(sym, j)
        if len(psi_db) != len(psi_sym):
            print(f"{j:>2}  LENGTH MISMATCH {len(psi_db)} vs {len(psi_sym)}")
            continue
        d = float(np.max(np.abs(psi_db - psi_sym)))
        worst = max(worst, d)
        print(f"{j:>2} {len(psi_db):>10} {d:>14.2e}")
    print()
    print(f"max ecart toutes echelles : {worst:.2e}")
    print("=> Psi_j identique (Remark 2 verifie)" if worst < 1e-10
          else "=> ECART NON NEGLIGEABLE, revoir l'hypothese")
    return 0


if __name__ == "__main__":
    sys.exit(main())
