"""
Phase 3 Slice A: silver -> gold (feature engineering, non-EPA slice).

Builds:
    gold.team_ratings      Elo, one row per team per game
    gold.team_game_stats   points for/against, win/loss, margin
    gold.injury_impact     injury_impact_v0 heuristic, one row per team per game
    gold.game_odds         reshaped from silver.odds
    gold.game_features     one row per game, assembled from all of the above

Unlike silver_transform (which processes one season/week at a time),
this recomputes gold from ALL of silver.games on every run -- Elo and
recent-form need full chronological history to stay point-in-time
correct, and at V1 data volumes (thousands of games at most)
recomputing from scratch is simpler and safer than maintaining
incremental state that could drift from silver. Every gold table here
is fully replaced (DELETE + re-INSERT) each run, not upserted row by
row.

Point-in-time correctness (design doc section 19) is the load-bearing
requirement of this whole module:
    - elo_pre/elo_post come from src/features/elo.py, which processes
      games strictly in (season, week) order.
    - injury_impact only ever sees silver.injuries rows with
      reported_at < that game's game_date.
    - recent_form only looks at a team's games strictly before the
      current one.
See tests/test_gold_transform.py for the tests that enforce this.

Usage:
    python -m src.transform.gold_transform
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db import get_engine, init_schema, qualified_table
from src.features.elo import compute_elo_ratings
from src.features.game_status import is_final_game
from src.features.injury_impact import compute_team_injury_impact, load_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("gold_transform")

RECENT_FORM_WINDOW = 4  # games


def _read_table(engine: Engine, schema: str, table: str) -> pd.DataFrame:
    qualified = qualified_table(schema, table)
    return pd.read_sql(text(f"SELECT * FROM {qualified}"), engine)


def _replace_table(engine: Engine, schema: str, table: str, df: pd.DataFrame) -> int:
    """Delete all rows and bulk-insert `df`. Simplest correct way to keep
    a fully-recomputed gold table in sync with its inputs each run."""
    qualified = qualified_table(schema, table)
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {qualified}"))
        if df.empty:
            return 0
        cols = list(df.columns)
        placeholders = ", ".join(f":{c}" for c in cols)
        col_list = ", ".join(cols)
        records = df.to_dict(orient="records")
        conn.execute(text(f"INSERT INTO {qualified} ({col_list}) VALUES ({placeholders})"), records)
    return len(df)


def _int_or_none(value):
    """int(value) if it's a real number, else None -- guards against
    pandas turning a NULL INTEGER column into float NaN on read (verified:
    pd.read_sql upcasts a nullable int column to float64/NaN), which then
    fails the INSERT against an INTEGER column on Postgres (SQLite silently
    coerces NaN to NULL, which is why this only shows up in production)."""
    return int(value) if pd.notna(value) else None


def _compute_team_game_stats(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in games.iterrows():
        final = is_final_game(g)
        home_win = away_win = None
        home_margin = away_margin = None
        if final:
            home_margin = int(g["home_score"] - g["away_score"])
            away_margin = -home_margin
            if home_margin > 0:
                home_win, away_win = True, False
            elif home_margin < 0:
                home_win, away_win = False, True
            # else: tie -- leave both win flags None rather than a false "loss"
            # for both teams; _compute_recent_form scores it 0.5 via margin.
        home_points_for = _int_or_none(g["home_score"])
        away_points_for = _int_or_none(g["away_score"])
        season = _int_or_none(g["season"])
        week = _int_or_none(g["week"])
        rows.append({"team_id": g["home_team_id"], "opponent_id": g["away_team_id"], "game_id": g["game_id"],
                     "season": season, "week": week, "is_home": True,
                     "points_for": home_points_for, "points_against": away_points_for,
                     "win": home_win, "margin": home_margin})
        rows.append({"team_id": g["away_team_id"], "opponent_id": g["home_team_id"], "game_id": g["game_id"],
                     "season": season, "week": week, "is_home": False,
                     "points_for": away_points_for, "points_against": home_points_for,
                     "win": away_win, "margin": away_margin})
    return pd.DataFrame(rows)


def _compute_recent_form(team_game_stats: pd.DataFrame) -> pd.DataFrame:
    """Win pct over the last RECENT_FORM_WINDOW *completed* games strictly
    before this one, per team (a tie counts as 0.5, matching Elo's
    treatment of ties -- see src/features/elo.py's _actual_score).
    Returns team_id, game_id, recent_form."""
    rows = []
    ordered = team_game_stats.sort_values(["team_id", "season", "week", "game_id"], kind="stable")
    for team_id, team_games in ordered.groupby("team_id", sort=False):
        history: list[float] = []
        for _, g in team_games.iterrows():
            recent = history[-RECENT_FORM_WINDOW:]
            recent_form = (sum(recent) / len(recent)) if recent else None
            rows.append({"team_id": team_id, "game_id": g["game_id"], "recent_form": recent_form})
            margin = g["margin"]
            if pd.notna(margin):
                history.append(1.0 if margin > 0 else 0.0 if margin < 0 else 0.5)
    return pd.DataFrame(rows)


def _compute_injury_impact_rows(games: pd.DataFrame, injuries: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    weights = load_weights()
    injuries = injuries.copy()
    injuries["reported_at"] = pd.to_datetime(injuries["reported_at"], errors="coerce")

    # Prefer the real report timestamp when nflverse provides one. It often
    # doesn't -- nflverse dropped `date_modified` from the 2025 injuries
    # file (see README design notes), so reported_at is NULL for real 2025
    # data. Fall back to (season, week) <= this game's (season, week):
    # injury reports are filed during the days leading into that week's
    # games, so a report from the same week always precedes kickoff.
    # Coarser than a timestamp, but still point-in-time safe -- it can
    # only ever look backward, never forward.
    has_timestamp = injuries["reported_at"].notna()
    timestamped = injuries[has_timestamp]
    fallback_pool = injuries[~has_timestamp & injuries["season"].notna() & injuries["week"].notna()]

    # The by-week fallback only depends on (season, week), not the exact
    # game_date, so it's computed once per distinct (season, week) instead
    # of rescanning the whole injuries table for every game -- a typical
    # week has ~16 games sharing the same cutoff.
    by_week_cache: dict[tuple, pd.DataFrame] = {}

    def _by_week_pool(season, week) -> pd.DataFrame:
        key = (season, week)
        if key not in by_week_cache:
            by_week_cache[key] = fallback_pool[
                (fallback_pool["season"] < season)
                | ((fallback_pool["season"] == season) & (fallback_pool["week"] <= week))
            ]
        return by_week_cache[key]

    rows = []
    for _, g in games.iterrows():
        cutoff = pd.to_datetime(g["game_date"], errors="coerce")
        by_timestamp = timestamped[timestamped["reported_at"] < cutoff] if pd.notna(cutoff) else timestamped.iloc[0:0]
        as_of = pd.concat([by_timestamp, _by_week_pool(g["season"], g["week"])], ignore_index=True)

        impact_by_team = compute_team_injury_impact(as_of, players, weights=weights)
        for team_id in (g["home_team_id"], g["away_team_id"]):
            rows.append({
                "team_id": team_id,
                "game_id": g["game_id"],
                "season": _int_or_none(g["season"]),
                "week": _int_or_none(g["week"]),
                "injury_impact": impact_by_team.get(team_id, 0.0),
                "config_version": weights["version"],
            })
    return pd.DataFrame(rows)


def _compute_game_odds(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    if odds.empty:
        return pd.DataFrame(columns=["game_id", "sportsbook", "source", "opening_spread", "current_spread",
                                      "spread_movement", "total", "moneyline_home", "moneyline_away"])
    merged = odds.merge(games[["game_id"]], on="game_id", how="inner")
    # gold.game_odds is one row per game (PK on game_id); if/when a second
    # sportsbook source is added, this needs the precedence rule from
    # DECISIONS.md #2 instead of "first row wins" -- fine for V1's single source.
    merged = merged.drop_duplicates(subset="game_id", keep="first")
    out = pd.DataFrame({
        "game_id": merged["game_id"],
        "sportsbook": merged["sportsbook"],
        "source": merged["source"],
        "opening_spread": merged["spread"],
        "current_spread": merged["spread"],
        "spread_movement": 0.0,  # only one snapshot exists in V1 -- see DECISIONS.md #2
        "total": merged["total"],
        "moneyline_home": merged["moneyline_home"],
        "moneyline_away": merged["moneyline_away"],
    })
    return out


def run() -> dict[str, int]:
    init_schema()
    engine = get_engine()

    games = _read_table(engine, "silver", "games")
    if games.empty:
        logger.warning("silver.games is empty -- nothing to build. Run silver_transform first.")
        return {"team_ratings": 0, "team_game_stats": 0, "injury_impact": 0, "game_odds": 0, "game_features": 0}

    injuries = _read_table(engine, "silver", "injuries")
    players = _read_table(engine, "silver", "players")
    odds = _read_table(engine, "silver", "odds")

    elo = compute_elo_ratings(games)
    elo_count = _replace_table(engine, "gold", "team_ratings", elo)
    logger.info("Rebuilt gold.team_ratings: %d rows", elo_count)

    team_game_stats = _compute_team_game_stats(games)
    stats_count = _replace_table(engine, "gold", "team_game_stats", team_game_stats)
    logger.info("Rebuilt gold.team_game_stats: %d rows", stats_count)

    recent_form = _compute_recent_form(team_game_stats)

    injury_rows = _compute_injury_impact_rows(games, injuries, players)
    injury_count = _replace_table(engine, "gold", "injury_impact", injury_rows)
    logger.info("Rebuilt gold.injury_impact: %d rows", injury_count)

    game_odds = _compute_game_odds(games, odds)
    odds_count = _replace_table(engine, "gold", "game_odds", game_odds)
    logger.info("Rebuilt gold.game_odds: %d rows", odds_count)

    features = _assemble_game_features(games, elo, injury_rows, recent_form, game_odds)
    features_count = _replace_table(engine, "gold", "game_features", features)
    logger.info("Rebuilt gold.game_features: %d rows", features_count)

    return {
        "team_ratings": elo_count,
        "team_game_stats": stats_count,
        "injury_impact": injury_count,
        "game_odds": odds_count,
        "game_features": features_count,
    }


def _assemble_game_features(games: pd.DataFrame, elo: pd.DataFrame, injury_rows: pd.DataFrame,
                             recent_form: pd.DataFrame, game_odds: pd.DataFrame) -> pd.DataFrame:
    elo_by_team_game = {(r["team_id"], r["game_id"]): r["elo_pre"] for _, r in elo.iterrows()}
    injury_by_team_game = {(r["team_id"], r["game_id"]): r["injury_impact"] for _, r in injury_rows.iterrows()}
    form_by_team_game = {(r["team_id"], r["game_id"]): r["recent_form"] for _, r in recent_form.iterrows()}
    odds_by_game = {r["game_id"]: r for _, r in game_odds.iterrows()}

    rows = []
    for _, g in games.iterrows():
        game_id = g["game_id"]
        home_id, away_id = g["home_team_id"], g["away_team_id"]
        home_elo = elo_by_team_game.get((home_id, game_id))
        away_elo = elo_by_team_game.get((away_id, game_id))
        odds_row = odds_by_game.get(game_id)

        rows.append({
            "game_id": game_id,
            "season": _int_or_none(g["season"]),
            "week": _int_or_none(g["week"]),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": (home_elo - away_elo) if home_elo is not None and away_elo is not None else None,
            "home_injury_impact": injury_by_team_game.get((home_id, game_id)),
            "away_injury_impact": injury_by_team_game.get((away_id, game_id)),
            "home_rest_days": _int_or_none(g.get("home_rest_days")),
            "away_rest_days": _int_or_none(g.get("away_rest_days")),
            "home_recent_form": form_by_team_game.get((home_id, game_id)),
            "away_recent_form": form_by_team_game.get((away_id, game_id)),
            "temperature": None,
            "wind": None,
            "opening_spread": odds_row["opening_spread"] if odds_row is not None else None,
            "current_spread": odds_row["current_spread"] if odds_row is not None else None,
            "spread_movement": odds_row["spread_movement"] if odds_row is not None else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    counts = run()
    logger.info("Gold transform complete: %s", counts)


if __name__ == "__main__":
    main()
