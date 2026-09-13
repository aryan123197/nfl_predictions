"""
Read-only queries over silver + gold for the API.

All SQL lives here rather than in the route handlers, so the routes stay a
thin HTTP layer and the queries can be tested directly. Every statement is a
SELECT: the API never writes, since the pipelines own all writes and a
read-only surface is one less way to corrupt the warehouse from the web.

Table references go through qualified_table() so the same code serves
Postgres (`gold.game_features`) and the SQLite dev/test database
(`gold_game_features`) -- the same dialect-portability the rest of the
project already relies on.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import inspect, text

from src.db import get_engine, qualified_table

# silver.games is the source of truth for what a game IS (teams, date,
# score, status); gold.game_features is the source of truth for what the
# model sees. A LEFT JOIN keeps games visible even before the gold
# transform has run for them -- a scheduled game with no features yet is a
# real state, not an error.
_GAME_SELECT = """
    SELECT
        g.game_id, g.season, g.week, g.game_date, g.status,
        g.home_team_id, g.away_team_id, g.home_score, g.away_score,
        g.home_rest_days, g.away_rest_days,
        f.home_elo, f.away_elo, f.elo_difference,
        f.home_injury_impact, f.away_injury_impact,
        f.home_recent_form, f.away_recent_form,
        f.home_off_epa, f.away_off_epa, f.home_def_epa, f.away_def_epa,
        f.opening_spread, f.current_spread, f.spread_movement
    FROM {games} g
    LEFT JOIN {features} f ON f.game_id = g.game_id
"""


def _game_select() -> str:
    return _GAME_SELECT.format(
        games=qualified_table("silver", "games"),
        features=qualified_table("gold", "game_features"),
    )


def _rows(sql: str, params: dict | None = None) -> list[dict]:
    with get_engine().begin() as conn:
        return [dict(r) for r in conn.execute(text(sql), params or {}).mappings().all()]


def warehouse_ready() -> bool:
    """False when the database has no silver.games table at all.

    Distinguishes "the pipeline has never run here" from "the pipeline ran
    and found nothing", so /health can say which -- a fresh clone hitting an
    empty database should get a clear answer, not an opaque 500.
    """
    engine = get_engine()
    if engine.dialect.name == "sqlite":
        return inspect(engine).has_table(qualified_table("silver", "games"))
    return inspect(engine).has_table("games", schema="silver")



def list_games(season: Optional[int] = None, week: Optional[int] = None) -> list[dict]:
    where, params = [], {}
    if season is not None:
        where.append("g.season = :season")
        params["season"] = season
    if week is not None:
        where.append("g.week = :week")
        params["week"] = week

    sql = _game_select()
    if where:
        sql += " WHERE " + " AND ".join(where)
    # game_id in the sort key keeps ordering deterministic for games sharing
    # a kickoff slot, so the week view doesn't reshuffle between requests.
    sql += " ORDER BY g.season, g.week, g.game_date, g.game_id"
    return _rows(sql, params)


def get_game(game_id: str) -> Optional[dict]:
    sql = _game_select() + " WHERE g.game_id = :game_id"
    rows = _rows(sql, {"game_id": game_id})
    return rows[0] if rows else None


def list_seasons() -> list[int]:
    """Seasons that actually have games, newest first -- lets the UI populate
    its season picker from the data instead of hardcoding a range that goes
    stale or offers seasons this database has never ingested."""
    sql = f"SELECT DISTINCT season FROM {qualified_table('silver', 'games')} ORDER BY season DESC"
    return [r["season"] for r in _rows(sql)]


def list_weeks(season: int) -> list[int]:
    sql = (
        f"SELECT DISTINCT week FROM {qualified_table('silver', 'games')} "
        "WHERE season = :season AND week IS NOT NULL ORDER BY week"
    )
    return [r["week"] for r in _rows(sql, {"season": season})]


def list_teams() -> list[dict]:
    sql = (
        f"SELECT team_id, first_season, last_season "
        f"FROM {qualified_table('silver', 'teams')} ORDER BY team_id"
    )
    return _rows(sql)


def get_team(team_id: str) -> Optional[dict]:
    sql = (
        f"SELECT team_id, first_season, last_season "
        f"FROM {qualified_table('silver', 'teams')} WHERE team_id = :team_id"
    )
    rows = _rows(sql, {"team_id": team_id})
    return rows[0] if rows else None


def get_team_rating(team_id: str, season: Optional[int] = None) -> Optional[dict]:
    """Most recent Elo rating for a team, optionally within one season.

    gold.team_ratings is point-in-time (one row per team per game), so
    "current rating" is the latest row -- ordered by season/week rather than
    by generated_at, which reflects when the transform ran, not what the
    rating describes.
    """
    table = qualified_table("gold", "team_ratings")
    if not inspect(get_engine()).has_table(table):
        return None

    sql = f"SELECT * FROM {table} WHERE team_id = :team_id"
    params: dict = {"team_id": team_id}
    if season is not None:
        sql += " AND season = :season"
        params["season"] = season
    sql += " ORDER BY season DESC, week DESC LIMIT 1"
    rows = _rows(sql, params)
    return rows[0] if rows else None
