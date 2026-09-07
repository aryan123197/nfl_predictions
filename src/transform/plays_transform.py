"""
Phase 3 Slice B: bronze -> silver for play-by-play (design doc section 11).

Bulk delete-then-insert per season, like run_pbp_ingestion.py's bronze
write -- a season is ~45k+ rows, too many for the per-row upsert loop
silver_transform.py uses for games/injuries/players/odds.

Usage:
    python -m src.transform.plays_transform --season 2025
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from sqlalchemy import text

from src.db import get_engine, init_schema, qualified_table, to_sql_target
from src.transform.team_aliases import canonical_team_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("plays_transform")

BOOLEAN_COLUMNS = [
    "success", "pass_attempt", "rush_attempt", "interception", "fumble_lost",
    "touchdown", "sack", "qb_hit", "first_down", "third_down_converted", "third_down_failed",
]
INT_COLUMNS = ["qtr", "down", "ydstogo", "yardline_100"]
SILVER_COLUMNS = (
    ["play_id", "game_id", "season", "week", "qtr", "down", "ydstogo", "yardline_100",
     "posteam", "defteam", "play_type", "yards_gained", "epa"]
    + BOOLEAN_COLUMNS
    + ["passer_player_id", "rusher_player_id", "receiver_player_id"]
)


def _read_bronze_plays(engine, season: int) -> pd.DataFrame:
    table = qualified_table("bronze", "plays_raw")
    return pd.read_sql(text(f"SELECT * FROM {table} WHERE season = :season"), engine, params={"season": season})


def _transform(bronze_plays: pd.DataFrame) -> pd.DataFrame:
    df = bronze_plays.copy()
    df["posteam"] = df["posteam"].apply(canonical_team_id)
    df["defteam"] = df["defteam"].apply(canonical_team_id)
    for col in BOOLEAN_COLUMNS:
        df[col] = df[col].apply(lambda v: None if pd.isna(v) else bool(v))
    for col in INT_COLUMNS:
        df[col] = df[col].apply(lambda v: None if pd.isna(v) else int(v))

    out = df[SILVER_COLUMNS].reset_index(drop=True)
    # Belt-and-suspenders: any other NaN that slipped through (e.g. a
    # missing yards_gained/epa) becomes a real SQL NULL, not a float NaN
    # that could get bound into a non-float column -- see DECISIONS.md /
    # the Slice A bug review for why this matters on Postgres specifically.
    out = out.where(pd.notna(out), None)
    return out


def run(season: int) -> int:
    init_schema()
    engine = get_engine()

    bronze_plays = _read_bronze_plays(engine, season)
    if bronze_plays.empty:
        logger.warning("bronze.plays_raw has no rows for season=%s -- run run_pbp_ingestion first.", season)
        return 0

    silver_plays = _transform(bronze_plays)
    table = qualified_table("silver", "plays")
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {table} WHERE season = :season"), {"season": season})
        silver_plays.to_sql(con=conn, if_exists="append", index=False, **to_sql_target("silver", "plays"))
    logger.info("Rebuilt silver.plays for season=%s: %d rows", season, len(silver_plays))
    return len(silver_plays)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transform bronze play-by-play into the silver layer.")
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()
    run(season=args.season)


if __name__ == "__main__":
    main()
