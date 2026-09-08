"""
Phase 6: Automated Data & Prediction Pipeline (design doc sections 29-31).

End-to-end orchestration workflow:
1. Ingests raw data (games, odds, injuries, players, play-by-play) into Bronze.
2. Normalizes into analytics-ready Silver tables.
3. Builds Gold feature tables (Elo, rolling EPA, injury impact, QB features).
4. Generates point-in-time predictions for upcoming games using the active Champion model.
5. Records execution status and metrics to metadata.pipeline_runs.

Usage:
    python -m src.pipeline.run_pipeline --season 2025
    python -m src.pipeline.run_pipeline --season 2025 --week 1
    python -m src.pipeline.run_pipeline --season 2025 --skip-plays
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from src.db import get_engine, init_schema
from src.ingest.pipeline_metadata import finish_run, start_run
from src.ingest.run_ingestion import run as run_ingestion
from src.ingest.run_pbp_ingestion import run as run_pbp_ingestion
from src.ml.registry import predict_upcoming_games
from src.providers.base import NFLDataProvider
from src.providers.nflverse_provider import NFLverseProvider
from src.transform import (
    game_results_transform,
    gold_transform,
    plays_transform,
    silver_transform,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline_runner")


def run(
    season: int | None = None,
    week: int | None = None,
    skip_plays: bool = False,
    provider: NFLDataProvider | None = None,
) -> dict:
    """Execute the automated data ingestion, transformation, and prediction pipeline."""
    if season is None:
        season = datetime.now(timezone.utc).year

    if provider is None:
        provider = NFLverseProvider()

    init_schema()
    engine = get_engine()
    run_id = start_run("nfl_data_pipeline")
    total_records = 0

    logger.info("=== Starting Automated Data Pipeline [run_id=%s, season=%s, week=%s] ===", run_id, season, week)

    try:
        # 1. Ingest Bronze (games, injuries, players, odds)
        logger.info("[Step 1/5] Ingesting Bronze data...")
        run_ingestion(provider, season=season, week=week)

        # 2. Transform Silver (games, teams, players, injuries, odds)
        logger.info("[Step 2/5] Transforming Silver layer...")
        silver_counts = silver_transform.run(season=season)
        total_records += sum(silver_counts.values())

        # 3. Ingest & Transform Play-by-Play (if not skipped)
        if not skip_plays:
            logger.info("[Step 3/5] Ingesting and transforming Play-by-Play...")
            pbp_ingested = run_pbp_ingestion(provider, season=season)
            pbp_transformed = plays_transform.run(season=season)
            total_records += pbp_ingested + pbp_transformed
        else:
            logger.info("[Step 3/5] Skipping Play-by-Play per configuration")

        # 4. Rebuild Gold Layer (Elo, rolling stats, injury impact, game_features, game_results)
        logger.info("[Step 4/5] Rebuilding Gold feature tables...")
        gold_counts = gold_transform.run()
        results_counts = game_results_transform.run()
        total_records += sum(gold_counts.values()) + results_counts

        # 5. Generate Upcoming Predictions with Active Champion
        logger.info("[Step 5/5] Generating predictions with active Champion model...")
        predictions_written = predict_upcoming_games(
            engine=engine,
            season=season,
            week=week,
        )
        total_records += predictions_written

        finish_run(run_id, "success", records_processed=total_records)
        logger.info(
            "=== Automated Data Pipeline Completed Successfully [run_id=%s, records=%d, predictions=%d] ===",
            run_id,
            total_records,
            predictions_written,
        )
        return {
            "run_id": run_id,
            "status": "success",
            "records_processed": total_records,
            "predictions_written": predictions_written,
        }

    except Exception as e:
        logger.exception("Pipeline failed with exception: %s", e)
        finish_run(run_id, "failed", records_processed=total_records, error_message=str(e))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the automated NFL data ingestion, feature build, and prediction pipeline.")
    parser.add_argument("--season", type=int, default=None, help="Target season (default: current year)")
    parser.add_argument("--week", type=int, default=None, help="Target week (default: all weeks in season)")
    parser.add_argument("--skip-plays", action="store_true", help="Skip play-by-play ingestion and transformation")
    args = parser.parse_args()

    run(season=args.season, week=args.week, skip_plays=args.skip_plays)


if __name__ == "__main__":
    main()
