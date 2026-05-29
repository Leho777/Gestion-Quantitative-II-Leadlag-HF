"""Bundle data/ + heavy outputs/ artifacts into one shareable tarball.

Use case
--------
The git repo is intentionally lightweight (raw/processed data and heavyweight
artefacts are gitignored). To onboard a collaborator, we need to ship them the
data so they don't have to re-download Binance / Kraken (Kraken alone is ~40
min of REST pagination for one month).

This script builds ``hk_leadlag_data.tar.gz`` (~1.5 GB compressed) ready to
upload on Google Drive / OneDrive / etc.

Usage::

    python scripts/make_data_archive.py
    python scripts/make_data_archive.py --out /path/to/output.tar.gz
    python scripts/make_data_archive.py --skip-raw    # only ship npz + outputs

Default contents
----------------
* ``data/raw/binance/`` (~1.5 GB, parquet aggTrades)
* ``data/raw/bybit/`` (~38 MB)
* ``data/raw/kraken/`` (~13 MB)
* ``data/raw/live_collect/`` (Hyperliquid L2 + Binance bookTicker snapshots)
* ``data/processed/`` (~69 MB, npz pré-construits)
* ``data/equity/*.zip`` (~82 MB LOBSTER zips)
* ``outputs/**/*.pkl`` (HK fit results)
* ``outputs/**/*.png`` (heatmaps, contrast plots)
* ``outputs/**/log.txt`` (stdout captures)

NOT included (tracked in git already, redundant)
* ``outputs/**/config.json``
* ``outputs/**/metadata.json``
* ``outputs/**/series_stats.json``
* ``outputs/**/summary.csv``
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# Patterns to include relative to ROOT.
INCLUDE_PATTERNS_RAW = [
    "data/raw/binance/**/*.parquet",
    "data/raw/bybit/**/*.parquet",
    "data/raw/kraken/**/*.parquet",
    "data/raw/live_collect/**/*.parquet",
]
INCLUDE_PATTERNS_LIGHT = [
    "data/processed/*.npz",
    "data/equity/*.zip",
    "data/MANIFEST.md",
    "outputs/**/*.pkl",
    "outputs/**/*.png",
    "outputs/**/log.txt",
]


def collect_files(patterns: list[str]) -> list[Path]:
    seen: set[Path] = set()
    for pat in patterns:
        for p in ROOT.glob(pat):
            if p.is_file():
                seen.add(p.resolve())
    return sorted(seen)


def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path,
                   default=ROOT.parent / "hk_leadlag_data.tar.gz",
                   help="Output tarball path (default: parent dir).")
    p.add_argument("--skip-raw", action="store_true",
                   help="Skip data/raw/ (large). Ship only processed npz + outputs.")
    args = p.parse_args(argv)

    patterns = list(INCLUDE_PATTERNS_LIGHT)
    if not args.skip_raw:
        patterns = INCLUDE_PATTERNS_RAW + patterns

    print(f"Bundling files matching {len(patterns)} patterns...")
    files = collect_files(patterns)
    if not files:
        print("No files matched. Nothing to bundle.", file=sys.stderr)
        return 1

    total = sum(f.stat().st_size for f in files)
    print(f"{len(files):,} files, {human_size(total)} (uncompressed)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.out, "w:gz", compresslevel=6) as tar:
        for f in files:
            arcname = f.relative_to(ROOT)
            tar.add(f, arcname=str(arcname))
            if len(files) > 100 and tar.fileobj.tell() > 0:
                pass  # too noisy, skip per-file logging
    final_size = args.out.stat().st_size
    print(f"Created {args.out}  ({human_size(final_size)} compressed)")
    print()
    print("Next steps:")
    print(f"  1. Upload {args.out.name} to Google Drive / OneDrive / Dauphine filer.")
    print(f"  2. Get the shareable link.")
    print(f"  3. Share the link with collaborators who need the cached data.")
    print(f"  4. Collaborator extracts with:")
    print(f"     tar -xzf {args.out.name} -C hk_leadlag_replication/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
