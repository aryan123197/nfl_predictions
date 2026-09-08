"""Backfill the full pipeline (bronze -> silver -> gold) for a range of
historical seasons.

This is what you'd run once, up front, to populate enough history for
Phase 4/5 (model training + walk-forward backtesting) -- e.g. 2020-2025.
Runs play-by-play too (--skip-plays to opt out, e.g. for a quick smoke
test) -- Phase 4's feature table needs EPA rolling stats, which don't
exist without it.

gold_transform.py and game_results_transform.py recompute from ALL of
silver.games/injuries/plays/odds on every run (see gold_transform.py's
own docstring for why), so they only need to run once at the end, not
once per season.

Usage:
    python scripts/backfill_seasons.py --start 2020 --end 2025
    python scripts/backfill_seasons.py --start 2024 --end 2025 --skip-plays  # faster, no EPA features
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.run_ingestion import run as run_ingestion, logger  # noqa: E402
from src.ingest.run_pbp_ingestion import run as run_pbp_ingestion  # noqa: E402
from src.providers.nflverse_provider import NFLverseProvider  # noqa: E402
from src.transform import game_results_transform, gold_transform, plays_transform, silver_transform  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--skip-plays", action="store_true",
                         help="Skip play-by-play ingestion (faster, but gold.game_features EPA "
                              "columns stay NULL for these seasons -- fine for a quick smoke test, "
                              "not for real model training).")
    args = parser.parse_args()

    provider = NFLverseProvider()
    for season in range(args.start, args.end + 1):
        logger.info("=== Backfilling season %s ===", season)
        try:
            run_ingestion(provider, season=season)
            silver_transform.run(season=season)
            if not args.skip_plays:
                run_pbp_ingestion(provider, season=season)
                plays_transform.run(season=season)
        except Exception:
            logger.error("Season %s failed, continuing with next season", season)
            continue

    logger.info("=== Rebuilding gold layer across all backfilled seasons ===")
    gold_transform.run()
    game_results_transform.run()
    logger.info("=== Backfill complete ===")
