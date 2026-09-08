"""
Phase 1 ingestion entrypoint.

Pulls games, injuries, and players for a given season from a provider
and upserts them into the bronze layer, wrapping the whole run in a
metadata.pipeline_runs record (per design doc section 31).

Usage:
    python -m src.ingest.run_ingestion --season 2025
    python -m src.ingest.run_ingestion --season 2025 --week 3
    python -m src.ingest.run_ingestion --season 2025 --skip-injuries

This intentionally does NOT ingest play-by-play here -- that's a
separate, heavier job (src/ingest/run_pbp_ingestion.py, Phase 3 Slice B)
since play-by-play files are 40-80MB per season and weren't needed for
the first game-level model.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys

import pandas as pd
from sqlalchemy import text

from src.db import get_engine, init_schema, qualified_table
from src.ingest.pipeline_metadata import finish_run, start_run
from src.providers.base import NFLDataProvider, ProviderError
from src.providers.nflverse_provider import NFLverseProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ingestion")


def _row_hash(row: pd.Series) -> str:
    return hashlib.sha256(row.to_json().encode("utf-8")).hexdigest()[:16]


def _upsert_games(df: pd.DataFrame, source: str, run_id: int) -> int:
    if df.empty:
        return 0
    engine = get_engine()
    table = qualified_table("bronze", "games_raw")
    df = df.copy()
    df["source"] = source
    df["pipeline_run_id"] = run_id
    df["raw_payload_hash"] = df.apply(_row_hash, axis=1)

    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    is_postgres = engine.dialect.name != "sqlite"

    with engine.begin() as conn:
        if is_postgres:
            from psycopg2.extras import execute_values
            cur = conn.connection.cursor()
            cols = [c for c in records[0].keys() if c != "id"]
            col_names = ", ".join(cols)
            update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in ("game_id", "source"))
            sql = f"""
            INSERT INTO {table} ({col_names})
            VALUES %s
            ON CONFLICT (game_id, source) DO UPDATE SET
                {update_clause}
            """
            data = [tuple(r[c] for c in cols) for r in records]
            execute_values(cur, sql, data, page_size=2000)
        else:
            for payload in records:
                existing = conn.execute(
                    text(f"SELECT id FROM {table} WHERE game_id = :game_id AND source = :source"),
                    {"game_id": payload["game_id"], "source": source},
                ).fetchone()
                if existing:
                    set_clause = ", ".join(f"{col} = :{col}" for col in payload if col not in ("game_id", "source"))
                    payload["id"] = existing[0]
                    conn.execute(text(f"UPDATE {table} SET {set_clause} WHERE id = :id"), payload)
                else:
                    cols = ", ".join(payload.keys())
                    placeholders = ", ".join(f":{c}" for c in payload.keys())
                    conn.execute(text(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"), payload)
    return len(df)


def _upsert_injuries(df: pd.DataFrame, source: str, run_id: int) -> int:
    if df.empty:
        return 0
    engine = get_engine()
    table = qualified_table("bronze", "injuries_raw")
    df = df.copy()
    df["source"] = source
    df["pipeline_run_id"] = run_id
    df["raw_payload_hash"] = df.apply(_row_hash, axis=1)

    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    is_postgres = engine.dialect.name != "sqlite"

    with engine.begin() as conn:
        if is_postgres:
            from psycopg2.extras import execute_values
            cur = conn.connection.cursor()
            cols = [c for c in records[0].keys() if c != "id"]
            col_names = ", ".join(cols)
            sql = f"INSERT INTO {table} ({col_names}) VALUES %s"
            data = [tuple(r[c] for c in cols) for r in records]
            execute_values(cur, sql, data, page_size=2000)
        else:
            cols = ", ".join(records[0].keys())
            placeholders = ", ".join(f":{c}" for c in records[0].keys())
            conn.execute(text(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"), records)
    return len(df)


def _upsert_players(df: pd.DataFrame, source: str, run_id: int, season: int) -> int:
    if df.empty:
        return 0
    engine = get_engine()
    table = qualified_table("bronze", "players_raw")
    df = df.copy()
    df["source"] = source
    df["pipeline_run_id"] = run_id
    df["season"] = season

    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    is_postgres = engine.dialect.name != "sqlite"

    with engine.begin() as conn:
        if is_postgres:
            from psycopg2.extras import execute_values
            cur = conn.connection.cursor()
            cols = [c for c in records[0].keys() if c != "id"]
            col_names = ", ".join(cols)
            update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in ("player_id", "season", "source"))
            sql = f"""
            INSERT INTO {table} ({col_names})
            VALUES %s
            ON CONFLICT (player_id, season, source) DO UPDATE SET
                {update_clause}
            """
            data = [tuple(r[c] for c in cols) for r in records]
            execute_values(cur, sql, data, page_size=2000)
        else:
            for payload in records:
                existing = conn.execute(
                    text(f"SELECT id FROM {table} WHERE player_id = :player_id AND season = :season AND source = :source"),
                    {"player_id": payload["player_id"], "season": season, "source": source},
                ).fetchone()
                if existing:
                    set_clause = ", ".join(f"{col} = :{col}" for col in payload if col not in ("player_id", "season", "source"))
                    payload["id"] = existing[0]
                    conn.execute(text(f"UPDATE {table} SET {set_clause} WHERE id = :id"), payload)
                else:
                    cols = ", ".join(payload.keys())
                    placeholders = ", ".join(f":{c}" for c in payload.keys())
                    conn.execute(text(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"), payload)
    return len(df)


def run(provider: NFLDataProvider, season: int, week: int | None = None,
        skip_injuries: bool = False, skip_players: bool = False) -> None:
    init_schema()
    run_id = start_run("nfl_data_pipeline")
    total_records = 0

    try:
        logger.info("Fetching games for season=%s week=%s from %s", season, week, provider.name)
        games = provider.get_games(season=season, week=week)
        total_records += _upsert_games(games, source=provider.name, run_id=run_id)
        logger.info("Upserted %d game rows", len(games))

        if not skip_injuries:
            logger.info("Fetching injuries for season=%s week=%s", season, week)
            injuries = provider.get_injuries(season=season, week=week)
            total_records += _upsert_injuries(injuries, source=provider.name, run_id=run_id)
            logger.info("Inserted %d injury rows", len(injuries))

        if not skip_players:
            logger.info("Fetching players for season=%s", season)
            players = provider.get_players(season=season)
            total_records += _upsert_players(players, source=provider.name, run_id=run_id, season=season)
            logger.info("Upserted %d player rows", len(players))

        finish_run(run_id, status="success", records_processed=total_records)
        logger.info("Pipeline run %s completed successfully (%d records)", run_id, total_records)

    except ProviderError as exc:
        logger.error("Provider error: %s", exc)
        finish_run(run_id, status="failed", records_processed=total_records, error_message=str(exc))
        raise
    except Exception as exc:  # noqa: BLE001 - top-level pipeline boundary
        logger.exception("Unexpected pipeline failure")
        finish_run(run_id, status="failed", records_processed=total_records, error_message=str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NFL data into the bronze layer.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--skip-injuries", action="store_true")
    parser.add_argument("--skip-players", action="store_true")
    args = parser.parse_args()

    provider = NFLverseProvider()
    try:
        run(provider, season=args.season, week=args.week,
            skip_injuries=args.skip_injuries, skip_players=args.skip_players)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
