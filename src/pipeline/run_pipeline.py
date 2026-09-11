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
import json
import logging
from datetime import datetime, timezone

from src.db import get_engine, init_schema, update_pipeline_state
from src.ingest.pipeline_metadata import finish_run, start_run
from src.ingest.run_ingestion import run as run_ingestion
from src.ingest.run_pbp_ingestion import run as run_pbp_ingestion
from src.ml.registry import predict_upcoming_games
from src.pipeline.health_check import run_health_check
from src.pipeline.schedule_resolver import get_active_week, get_current_season
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
    check_health: bool = False,
) -> dict:
    """Execute the automated data ingestion, transformation, and prediction pipeline."""
    init_schema()
    engine = get_engine()

    if season is None:
        season = get_current_season()

    if week is None:
        resolved_week = get_active_week(engine=engine, season=season)
    else:
        resolved_week = week

    if provider is None:
        provider = NFLverseProvider()

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
            pbp_ingested = run_pbp_ingestion(provider, season=season) or 0
            pbp_transformed = plays_transform.run(season=season) or 0
            total_records += (pbp_ingested or 0) + (pbp_transformed or 0)
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
        
        # Phase 7: Update pipeline state
        update_pipeline_state(
            pipeline_name="nfl_data_pipeline",
            current_season=season,
            current_week=resolved_week,
            last_successful_run_id=run_id,
            status="idle",
            metadata_json=json.dumps({
                "predictions_written": predictions_written,
                "total_records": total_records,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }),
        )

        logger.info(
            "=== Automated Data Pipeline Completed Successfully [run_id=%s, records=%d, predictions=%d] ===",
            run_id,
            total_records,
            predictions_written,
        )

        health_summary = None
        if check_health:
            logger.info("Running post-pipeline health check...")
            health_res = run_health_check(season=season, week=resolved_week, engine=engine)
            health_summary = health_res.to_dict()
            logger.info("Health check status: %s", health_res.status)

        return {
            "run_id": run_id,
            "status": "success",
            "records_processed": total_records,
            "predictions_written": predictions_written,
            "health": health_summary,
        }

    except Exception as e:
        logger.exception("Pipeline failed with exception: %s", e)
        finish_run(run_id, "failed", records_processed=total_records, error_message=str(e))
        update_pipeline_state(
            pipeline_name="nfl_data_pipeline",
            current_season=season,
            current_week=resolved_week,
            status="error",
            metadata_json=json.dumps({
                "error": str(e),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }),
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the automated NFL data ingestion, feature build, and prediction pipeline.")
    parser.add_argument("--season", type=int, default=None, help="Target season (default: dynamically determined current season)")
    parser.add_argument("--week", type=int, default=None, help="Target week (default: dynamically determined active week)")
    parser.add_argument("--skip-plays", action="store_true", help="Skip play-by-play ingestion and transformation")
    parser.add_argument("--check-health", action="store_true", help="Run system health and data quality check after completion")
    args = parser.parse_args()

    run(
        season=args.season,
        week=args.week,
        skip_plays=args.skip_plays,
        check_health=args.check_health,
    )


if __name__ == "__main__":
    main()
