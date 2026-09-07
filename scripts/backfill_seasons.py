"""Backfill full-season ingestion for a range of historical seasons.

This is what you'd run once, up front, to populate enough history for
Phase 4/5 (model training + walk-forward backtesting) -- e.g. 2020-2025.
Skips play-by-play (fetched separately, see Phase 3) to keep this fast.

Usage:
    python scripts/backfill_seasons.py --start 2020 --end 2025
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.run_ingestion import run, logger  # noqa: E402
from src.providers.nflverse_provider import NFLverseProvider  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    provider = NFLverseProvider()
    for season in range(args.start, args.end + 1):
        logger.info("=== Backfilling season %s ===", season)
        try:
            run(provider, season=season)
        except Exception:
            logger.error("Season %s failed, continuing with next season", season)
            continue
