"""
Phase 3 Slice B: play-by-play ingestion (bronze layer).

Separate from run_ingestion.py's games/injuries/players entrypoint, per
that module's own docstring note -- play-by-play files are 40-80MB per
season and take much longer to fetch/parse than the other sources, so
this runs on its own cadence rather than bundling into every ingestion
run.

Ingests a FULL SEASON at a time -- nflverse ships one file per season,
not per week, so get_plays() downloads and parses the whole file
regardless of any week filter. Bronze storage matches that: each run
deletes this season+source's existing rows and bulk-inserts the fresh
pull (not the per-row upsert loop games_raw/injuries_raw use -- a
season is ~45k+ play rows, and a per-row SELECT-then-branch loop
doesn't scale to that volume the way it's fine for a few hundred games
or injury reports).

Usage:
    python -m src.ingest.run_pbp_ingestion --season 2025
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
from sqlalchemy import text

from src.db import get_engine, init_schema, qualified_table, to_sql_target
from src.ingest.pipeline_metadata import finish_run, start_run
from src.providers.base import NFLDataProvider, ProviderError
from src.providers.nflverse_provider import NFLverseProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("pbp_ingestion")


def _replace_plays(df: pd.DataFrame, source: str, run_id: int, season: int) -> int:
    if df.empty:
        return 0
    engine = get_engine()
    table = qualified_table("bronze", "plays_raw")
    df = df.copy()
    df["source"] = source
    df["pipeline_run_id"] = run_id

    with engine.begin() as conn:
        conn.execute(
            text(f"DELETE FROM {table} WHERE season = :season AND source = :source"),
            {"season": season, "source": source},
        )
        df.to_sql(con=conn, if_exists="append", index=False, **to_sql_target("bronze", "plays_raw"))
    return len(df)


def run(provider: NFLDataProvider, season: int) -> None:
    init_schema()
    run_id = start_run("nfl_pbp_pipeline")

    try:
        logger.info("Fetching play-by-play for season=%s from %s (this can take a couple minutes)",
                     season, provider.name)
        plays = provider.get_plays(season=season)
        count = _replace_plays(plays, source=provider.name, run_id=run_id, season=season)
        finish_run(run_id, status="success", records_processed=count)
        logger.info("Pipeline run %s completed successfully (%d play rows)", run_id, count)

    except ProviderError as exc:
        logger.error("Provider error: %s", exc)
        finish_run(run_id, status="failed", records_processed=0, error_message=str(exc))
        raise
    except Exception as exc:  # noqa: BLE001 - top-level pipeline boundary
        logger.exception("Unexpected pipeline failure")
        finish_run(run_id, status="failed", records_processed=0, error_message=str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest play-by-play data into the bronze layer.")
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    provider = NFLverseProvider()
    try:
        run(provider, season=args.season)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
