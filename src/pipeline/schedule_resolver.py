"""
Phase 7: Schedule & Dynamic Week Resolver.

Provides automatic point-in-time season and week determination for zero-touch
continuous execution across the live NFL season.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db import get_engine, qualified_table


def get_current_season(dt: datetime | date | None = None) -> int:
    """Return the active NFL season year for a given date.
    
    NFL league year rolls over in March (offseason/draft), season runs Sept-Feb.
    Dates in Jan/Feb belong to the previous calendar year's season.
    """
    if dt is None:
        dt = datetime.now(timezone.utc).date()
    elif isinstance(dt, datetime):
        dt = dt.date()

    if dt.month < 3:
        return dt.year - 1
    return dt.year


def get_active_week(
    engine: Engine | None = None,
    season: int | None = None,
    as_of_date: date | None = None,
) -> int:
    """Determine the current/upcoming NFL week.
    
    First checks database games if available for the given season.
    Falls back to a calendar-based calculation starting from September.
    """
    if as_of_date is None:
        as_of_date = datetime.now(timezone.utc).date()

    if season is None:
        season = get_current_season(as_of_date)

    if engine is None:
        engine = get_engine()

    # 1. Attempt DB-backed resolution from silver.games
    try:
        games_tbl = qualified_table("silver", "games")
        with engine.connect() as conn:
            # Look for the week of upcoming or in-progress games around the target date
            query = text(f"""
                SELECT week
                FROM {games_tbl}
                WHERE season = :season
                  AND game_date >= :window_start
                  AND game_date <= :window_end
                ORDER BY game_date ASC
                LIMIT 1
            """)
            window_start = as_of_date - timedelta(days=2)
            window_end = as_of_date + timedelta(days=6)
            row = conn.execute(query, {
                "season": season,
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
            }).fetchone()
            if row and row[0] is not None:
                return int(row[0])

            # If no games in immediate window, find the earliest uncompleted week
            uncompleted_query = text(f"""
                SELECT week
                FROM {games_tbl}
                WHERE season = :season
                  AND (home_score IS NULL OR away_score IS NULL)
                ORDER BY week ASC
                LIMIT 1
            """)
            uncompleted_row = conn.execute(uncompleted_query, {"season": season}).fetchone()
            if uncompleted_row and uncompleted_row[0] is not None:
                return int(uncompleted_row[0])

            # If all games completed, return max week
            max_week_row = conn.execute(
                text(f"SELECT MAX(week) FROM {games_tbl} WHERE season = :season"),
                {"season": season},
            ).fetchone()
            if max_week_row and max_week_row[0] is not None:
                return int(max_week_row[0])
    except Exception:
        # If DB query fails or tables don't exist yet, fall through to calendar logic
        pass

    # 2. Calendar-based fallback heuristic
    # Week 1 typically kicks off the Thursday after first Monday in September
    # Estimate week 1 as ~Sept 5 of the season year
    sept_1 = date(season, 9, 1)
    # Day of week for Sept 1 (0=Mon, 6=Sun)
    # Labor Day is first Monday in Sept
    first_mon_offset = (0 - sept_1.weekday()) % 7
    labor_day = sept_1 + timedelta(days=first_mon_offset)
    week_1_thursday = labor_day + timedelta(days=3)

    if as_of_date < week_1_thursday:
        return 1

    days_since_w1 = (as_of_date - week_1_thursday).days
    week_num = 1 + (days_since_w1 // 7)
    # Cap between 1 and 22 (18 regular season weeks + 4 playoff rounds)
    return max(1, min(week_num, 22))


def get_upcoming_games(
    engine: Engine,
    season: int,
    week: int | None = None,
) -> list[dict]:
    """Retrieve upcoming (unplayed or pending) games for prediction."""
    games_tbl = qualified_table("silver", "games")
    with engine.connect() as conn:
        where_clause = "WHERE season = :season"
        params = {"season": season}
        if week is not None:
            where_clause += " AND week = :week"
            params["week"] = week

        query = text(f"""
            SELECT game_id, season, week, game_date, home_team_id, away_team_id, home_score, away_score
            FROM {games_tbl}
            {where_clause}
            ORDER BY game_date ASC, game_id ASC
        """)
        rows = conn.execute(query, params).fetchall()
        return [dict(r._mapping) for r in rows]
