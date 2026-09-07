"""
Elo team ratings -- design doc section 17.

A standard Elo implementation with a home-field advantage adjustment and
between-season mean reversion (the same "regress ratings toward the mean
in the offseason" technique used by public NFL Elo models like 538's),
so a team's rating from a strong prior season doesn't fully carry over
into a new one where key players may have left.

Point-in-time correctness (design doc section 19): ratings are computed
by walking games in chronological (season, week) order. Every game in a
given week is assigned its elo_pre from the rating state *after* the
previous week -- never from other games in the same week, and never
from any later week. Only completed ("final") games actually move a
team's rating; scheduled games get elo_post == elo_pre (no update,
since the outcome doesn't exist yet).

This recomputes full history from scratch every call rather than
persisting incremental state -- fine at V1 data volumes (a season is
~270 games), and it means there's no risk of the persisted state and
silver.games silently drifting out of sync. Revisit if/when this
becomes a performance problem.
"""

from __future__ import annotations

import pandas as pd

from src.features.game_status import is_final_game

INITIAL_RATING = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 55.0  # elo points added to the home team's rating before computing expected score
SEASON_REGRESSION = 1.0 / 3.0  # fraction of the way each team's rating regresses back to INITIAL_RATING each new season


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _actual_score(team_score: float, opponent_score: float) -> float:
    if team_score > opponent_score:
        return 1.0
    if team_score < opponent_score:
        return 0.0
    return 0.5  # tie


def compute_elo_ratings(games: pd.DataFrame) -> pd.DataFrame:
    """Compute point-in-time Elo ratings for every team-game in `games`.

    `games` must have columns: game_id, season, week, home_team_id,
    away_team_id, home_score, away_score, status. Order is not assumed
    -- this function sorts by (season, week) itself.

    Returns a long-format DataFrame, one row per team per game:
        team_id, game_id, season, week, elo_pre, elo_post
    """
    if games.empty:
        return pd.DataFrame(columns=["team_id", "game_id", "season", "week", "elo_pre", "elo_post"])

    ratings: dict[str, float] = {}
    last_season_seen: dict[str, int] = {}
    rows: list[dict] = []

    ordered = games.sort_values(["season", "week", "game_id"], kind="stable")
    for (season, week), week_games in ordered.groupby(["season", "week"], sort=False):
        # apply between-season regression once per team, before this week's games,
        # the first time we see that team in a new season
        for team_id in pd.concat([week_games["home_team_id"], week_games["away_team_id"]]).unique():
            prior_season = last_season_seen.get(team_id)
            if prior_season is not None and season > prior_season and team_id in ratings:
                ratings[team_id] += SEASON_REGRESSION * (INITIAL_RATING - ratings[team_id])
            last_season_seen[team_id] = season

        # snapshot elo_pre for every game this week from state as of the end of the previous week
        week_pre = {}
        for _, row in week_games.iterrows():
            home_id, away_id = row["home_team_id"], row["away_team_id"]
            home_pre = ratings.get(home_id, INITIAL_RATING)
            away_pre = ratings.get(away_id, INITIAL_RATING)
            week_pre[row["game_id"]] = (home_pre, away_pre)

        # compute updates from this week's final games, applied only after
        # the whole week's elo_pre values have been captured
        updates: dict[str, float] = {}
        for _, row in week_games.iterrows():
            game_id = row["game_id"]
            home_id, away_id = row["home_team_id"], row["away_team_id"]
            home_pre, away_pre = week_pre[game_id]

            if is_final_game(row):
                expected_home = _expected_score(home_pre + HOME_ADVANTAGE, away_pre)
                actual_home = _actual_score(row["home_score"], row["away_score"])
                delta = K_FACTOR * (actual_home - expected_home)
                home_post = home_pre + delta
                away_post = away_pre - delta
                updates[home_id] = home_post
                updates[away_id] = away_post
            else:
                home_post, away_post = home_pre, away_pre

            rows.append({"team_id": home_id, "game_id": game_id, "season": season, "week": week,
                         "elo_pre": home_pre, "elo_post": home_post})
            rows.append({"team_id": away_id, "game_id": game_id, "season": season, "week": week,
                         "elo_pre": away_pre, "elo_post": away_post})

        for team_id, new_rating in updates.items():
            ratings[team_id] = new_rating

    return pd.DataFrame(rows)
