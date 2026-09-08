"""
Phase 2 transform: bronze -> silver.

Reads a season's raw rows out of bronze.games_raw / injuries_raw /
players_raw and writes normalized, deduplicated rows into the silver
tables (schema/002_silver.sql), applying the rules from design doc
section 11:

    - type normalization      (scores/lines to proper numeric types)
    - deduplication            (players collapsed to one row per ID)
    - team ID normalization    (see team_aliases.py)
    - timestamp normalization  (gameday -> game_date)
    - null handling            (missing scores -> status='scheduled')

Unlike bronze, this is idempotently re-runnable: re-running the
transform for a season updates existing silver rows in place rather
than duplicating them. silver.injuries is the one exception to
"one row per entity" -- it stays point-in-time (one row per bronze
report row, upserted on bronze_id) by design, so injury trend features
stay possible later (see README design notes).

Usage:
    python -m src.transform.silver_transform --season 2025
    python -m src.transform.silver_transform --season 2025 --week 3
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db import get_engine, init_schema, qualified_table
from src.transform.team_aliases import canonical_team_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("silver_transform")


def _read_bronze(engine: Engine, table: str, season: int, week: Optional[int] = None) -> pd.DataFrame:
    qualified = qualified_table("bronze", table)
    query = f"SELECT * FROM {qualified} WHERE season = :season"
    params = {"season": season}
    if week is not None:
        query += " AND week = :week"
        params["week"] = week
    return pd.read_sql(text(query), engine, params=params)


def _upsert_teams(engine: Engine, team_ids: set[str], season: int) -> None:
    table = qualified_table("silver", "teams")
    with engine.begin() as conn:
        for team_id in sorted(team_ids):
            existing = conn.execute(
                text(f"SELECT team_id, first_season, last_season FROM {table} WHERE team_id = :team_id"),
                {"team_id": team_id},
            ).fetchone()
            if existing is None:
                conn.execute(
                    text(f"INSERT INTO {table} (team_id, first_season, last_season) "
                         f"VALUES (:team_id, :season, :season)"),
                    {"team_id": team_id, "season": season},
                )
            else:
                first_season = min(existing[1], season) if existing[1] is not None else season
                last_season = max(existing[2], season) if existing[2] is not None else season
                conn.execute(
                    text(f"UPDATE {table} SET first_season = :first_season, last_season = :last_season "
                         f"WHERE team_id = :team_id"),
                    {"first_season": first_season, "last_season": last_season, "team_id": team_id},
                )


def _game_status(row: pd.Series) -> str:
    home_score = row.get("home_score")
    away_score = row.get("away_score")
    if pd.notna(home_score) and pd.notna(away_score):
        return "final"
    return "scheduled"


def _upsert_games(engine: Engine, games: pd.DataFrame) -> int:
    if games.empty:
        return 0
    table = qualified_table("silver", "games")
    is_postgres = engine.dialect.name != "sqlite"
    records = []
    for _, row in games.iterrows():
        records.append({
            "game_id": row["game_id"],
            "season": int(row["season"]),
            "week": int(row["week"]) if pd.notna(row.get("week")) else None,
            "game_date": row.get("gameday"),
            "home_team_id": canonical_team_id(row.get("home_team")),
            "away_team_id": canonical_team_id(row.get("away_team")),
            "home_score": int(row["home_score"]) if pd.notna(row.get("home_score")) else None,
            "away_score": int(row["away_score"]) if pd.notna(row.get("away_score")) else None,
            "home_rest_days": int(row["home_rest"]) if pd.notna(row.get("home_rest")) else None,
            "away_rest_days": int(row["away_rest"]) if pd.notna(row.get("away_rest")) else None,
            "status": _game_status(row),
        })

    with engine.begin() as conn:
        if is_postgres:
            sql = f"""
            INSERT INTO {table} (game_id, season, week, game_date, home_team_id, away_team_id,
                                 home_score, away_score, home_rest_days, away_rest_days, status)
            VALUES (:game_id, :season, :week, :game_date, :home_team_id, :away_team_id,
                    :home_score, :away_score, :home_rest_days, :away_rest_days, :status)
            ON CONFLICT (game_id) DO UPDATE SET
                season = EXCLUDED.season,
                week = EXCLUDED.week,
                game_date = EXCLUDED.game_date,
                home_team_id = EXCLUDED.home_team_id,
                away_team_id = EXCLUDED.away_team_id,
                home_score = EXCLUDED.home_score,
                away_score = EXCLUDED.away_score,
                home_rest_days = EXCLUDED.home_rest_days,
                away_rest_days = EXCLUDED.away_rest_days,
                status = EXCLUDED.status,
                updated_at = now()
            """
            conn.execute(text(sql), records)
        else:
            for payload in records:
                existing = conn.execute(
                    text(f"SELECT game_id FROM {table} WHERE game_id = :game_id"),
                    {"game_id": payload["game_id"]},
                ).fetchone()
                if existing:
                    conn.execute(
                        text(f"UPDATE {table} SET season=:season, week=:week, game_date=:game_date, "
                             f"home_team_id=:home_team_id, away_team_id=:away_team_id, "
                             f"home_score=:home_score, away_score=:away_score, "
                             f"home_rest_days=:home_rest_days, away_rest_days=:away_rest_days, status=:status "
                             f"WHERE game_id=:game_id"),
                        payload,
                    )
                else:
                    conn.execute(
                        text(f"INSERT INTO {table} (game_id, season, week, game_date, home_team_id, "
                             f"away_team_id, home_score, away_score, home_rest_days, away_rest_days, status) "
                             f"VALUES (:game_id, :season, :week, :game_date, :home_team_id, "
                             f":away_team_id, :home_score, :away_score, :home_rest_days, :away_rest_days, :status)"),
                        payload,
                    )
    return len(records)


def _upsert_odds(engine: Engine, games: pd.DataFrame, source: str) -> int:
    if games.empty:
        return 0
    table = qualified_table("silver", "odds")
    sportsbook = "consensus"
    is_postgres = engine.dialect.name != "sqlite"
    records = []
    for _, row in games.iterrows():
        records.append({
            "game_id": row["game_id"],
            "sportsbook": sportsbook,
            "source": source,
            "spread": float(row["spread_line"]) if pd.notna(row.get("spread_line")) else None,
            "moneyline_home": float(row["home_moneyline"]) if pd.notna(row.get("home_moneyline")) else None,
            "moneyline_away": float(row["away_moneyline"]) if pd.notna(row.get("away_moneyline")) else None,
            "total": float(row["total_line"]) if pd.notna(row.get("total_line")) else None,
        })

    with engine.begin() as conn:
        if is_postgres:
            sql = f"""
            INSERT INTO {table} (game_id, sportsbook, source, spread, moneyline_home, moneyline_away, total)
            VALUES (:game_id, :sportsbook, :source, :spread, :moneyline_home, :moneyline_away, :total)
            ON CONFLICT (game_id, sportsbook, source) DO UPDATE SET
                spread = EXCLUDED.spread,
                moneyline_home = EXCLUDED.moneyline_home,
                moneyline_away = EXCLUDED.moneyline_away,
                total = EXCLUDED.total
            """
            conn.execute(text(sql), records)
        else:
            for payload in records:
                existing = conn.execute(
                    text(f"SELECT odds_id FROM {table} WHERE game_id=:game_id AND sportsbook=:sportsbook "
                         f"AND source=:source"),
                    payload,
                ).fetchone()
                if existing:
                    conn.execute(
                        text(f"UPDATE {table} SET spread=:spread, moneyline_home=:moneyline_home, "
                             f"moneyline_away=:moneyline_away, total=:total "
                             f"WHERE game_id=:game_id AND sportsbook=:sportsbook AND source=:source"),
                        payload,
                    )
                else:
                    conn.execute(
                        text(f"INSERT INTO {table} (game_id, sportsbook, source, spread, "
                             f"moneyline_home, moneyline_away, total) "
                             f"VALUES (:game_id, :sportsbook, :source, :spread, "
                             f":moneyline_home, :moneyline_away, :total)"),
                        payload,
                    )
    return len(records)


def _upsert_players(engine: Engine, players: pd.DataFrame) -> int:
    if players.empty:
        return 0
    table = qualified_table("silver", "players")
    is_postgres = engine.dialect.name != "sqlite"

    data = []
    records = []
    for _, row in players.iterrows():
        t_id = canonical_team_id(row.get("team"))
        p_name = str(row["full_name"]) if pd.notna(row.get("full_name")) else None
        p_pos = str(row["position"]) if pd.notna(row.get("position")) else None
        p_status = str(row["status"]) if pd.notna(row.get("status")) else None
        p_id = str(row["player_id"])

        data.append((p_id, p_name, p_pos, t_id, p_status))
        records.append({
            "player_id": p_id,
            "name": p_name,
            "position": p_pos,
            "team_id": t_id,
            "status": p_status,
        })

    with engine.begin() as conn:
        if is_postgres:
            from psycopg2.extras import execute_values
            raw_conn = conn.connection.dbapi_connection
            cur = raw_conn.cursor()
            sql = f"""
            INSERT INTO {table} (player_id, name, position, team_id, status)
            VALUES %s
            ON CONFLICT (player_id) DO UPDATE SET
                name = EXCLUDED.name,
                position = EXCLUDED.position,
                team_id = EXCLUDED.team_id,
                status = EXCLUDED.status,
                updated_at = now()
            """
            execute_values(cur, sql, data, page_size=2000)
        else:
            for payload in records:
                existing = conn.execute(
                    text(f"SELECT player_id FROM {table} WHERE player_id = :player_id"),
                    {"player_id": payload["player_id"]},
                ).fetchone()
                if existing:
                    conn.execute(
                        text(f"UPDATE {table} SET name=:name, position=:position, team_id=:team_id, "
                             f"status=:status WHERE player_id=:player_id"),
                        payload,
                    )
                else:
                    conn.execute(
                        text(f"INSERT INTO {table} (player_id, name, position, team_id, status) "
                             f"VALUES (:player_id, :name, :position, :team_id, :status)"),
                        payload,
                    )
    return len(data)


def _upsert_injuries(engine: Engine, injuries: pd.DataFrame, source: str) -> int:
    if injuries.empty:
        return 0
    table = qualified_table("silver", "injuries")
    is_postgres = engine.dialect.name != "sqlite"

    data = []
    records = []
    for _, row in injuries.iterrows():
        b_id = int(row["id"])
        p_id = str(row["player_id"]) if pd.notna(row.get("player_id")) else None
        t_id = canonical_team_id(row.get("team"))
        s_val = int(row["season"]) if pd.notna(row.get("season")) else None
        w_val = int(row["week"]) if pd.notna(row.get("week")) else None
        st_val = str(row["report_status"]) if pd.notna(row.get("report_status")) else None
        inj_type = str(row["report_primary_injury"]) if pd.notna(row.get("report_primary_injury")) else None
        rep_at = str(row["date_modified"]) if pd.notna(row.get("date_modified")) else None

        data.append((b_id, p_id, t_id, s_val, w_val, st_val, inj_type, rep_at, source))
        records.append({
            "bronze_id": b_id,
            "player_id": p_id,
            "team_id": t_id,
            "season": s_val,
            "week": w_val,
            "status": st_val,
            "injury_type": inj_type,
            "reported_at": rep_at,
            "source": source,
        })

    with engine.begin() as conn:
        if is_postgres:
            from psycopg2.extras import execute_values
            raw_conn = conn.connection.dbapi_connection
            cur = raw_conn.cursor()
            sql = f"""
            INSERT INTO {table} (bronze_id, player_id, team_id, season, week, status, injury_type, reported_at, source)
            VALUES %s
            ON CONFLICT (bronze_id) DO UPDATE SET
                player_id = EXCLUDED.player_id,
                team_id = EXCLUDED.team_id,
                season = EXCLUDED.season,
                week = EXCLUDED.week,
                status = EXCLUDED.status,
                injury_type = EXCLUDED.injury_type,
                reported_at = EXCLUDED.reported_at,
                source = EXCLUDED.source
            """
            execute_values(cur, sql, data, page_size=2000)
        else:
            for payload in records:
                existing = conn.execute(
                    text(f"SELECT injury_id FROM {table} WHERE bronze_id = :bronze_id"),
                    {"bronze_id": payload["bronze_id"]},
                ).fetchone()
                if existing:
                    conn.execute(
                        text(f"UPDATE {table} SET player_id=:player_id, team_id=:team_id, season=:season, "
                             f"week=:week, status=:status, injury_type=:injury_type, reported_at=:reported_at, "
                             f"source=:source WHERE bronze_id=:bronze_id"),
                        payload,
                    )
                else:
                    conn.execute(
                        text(f"INSERT INTO {table} (bronze_id, player_id, team_id, season, week, status, "
                             f"injury_type, reported_at, source) "
                             f"VALUES (:bronze_id, :player_id, :team_id, :season, :week, :status, :injury_type, "
                             f":reported_at, :source)"),
                        payload,
                    )
    return len(data)


def run(season: int, week: Optional[int] = None) -> dict[str, int]:
    """Run the bronze -> silver transform for one season (optionally one week).

    Returns a dict of row counts written per silver table, for logging /
    pipeline-metadata purposes.
    """
    init_schema()
    engine = get_engine()

    games = _read_bronze(engine, "games_raw", season, week)
    injuries = _read_bronze(engine, "injuries_raw", season, week)
    players = _read_bronze(engine, "players_raw", season)

    team_ids = {
        canonical_team_id(t)
        for t in pd.concat([games.get("home_team", pd.Series(dtype=object)),
                             games.get("away_team", pd.Series(dtype=object))])
        if pd.notna(t)
    }
    _upsert_teams(engine, team_ids, season)
    logger.info("Upserted %d teams", len(team_ids))

    games_count = _upsert_games(engine, games)
    logger.info("Upserted %d silver.games rows", games_count)

    odds_source = games["source"].iloc[0] if not games.empty else "unknown"
    odds_count = _upsert_odds(engine, games, source=odds_source)
    logger.info("Upserted %d silver.odds rows", odds_count)

    players_count = _upsert_players(engine, players)
    logger.info("Upserted %d silver.players rows", players_count)

    injuries_source = injuries["source"].iloc[0] if not injuries.empty else "unknown"
    injuries_count = _upsert_injuries(engine, injuries, source=injuries_source)
    logger.info("Upserted %d silver.injuries rows", injuries_count)

    return {
        "teams": len(team_ids),
        "games": games_count,
        "odds": odds_count,
        "players": players_count,
        "injuries": injuries_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Transform bronze data into the silver layer.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, default=None)
    args = parser.parse_args()
    counts = run(season=args.season, week=args.week)
    logger.info("Silver transform complete: %s", counts)


if __name__ == "__main__":
    main()
