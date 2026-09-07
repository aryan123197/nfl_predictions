"""Unit tests for src/features/elo.py -- no database needed."""

from __future__ import annotations

import pandas as pd

from src.features.elo import INITIAL_RATING, compute_elo_ratings


def _game(game_id, season, week, home, away, home_score=None, away_score=None, status="scheduled"):
    return {"game_id": game_id, "season": season, "week": week, "home_team_id": home, "away_team_id": away,
            "home_score": home_score, "away_score": away_score, "status": status}


def test_week1_games_start_at_initial_rating():
    games = pd.DataFrame([_game("g1", 2025, 1, "A", "B", 24, 17, "final")])
    elo = compute_elo_ratings(games)

    assert elo[elo["team_id"] == "A"]["elo_pre"].iloc[0] == INITIAL_RATING
    assert elo[elo["team_id"] == "B"]["elo_pre"].iloc[0] == INITIAL_RATING


def test_winner_gains_rating_loser_loses_rating():
    games = pd.DataFrame([_game("g1", 2025, 1, "A", "B", 24, 17, "final")])
    elo = compute_elo_ratings(games)

    winner_post = elo[elo["team_id"] == "A"]["elo_post"].iloc[0]
    loser_post = elo[elo["team_id"] == "B"]["elo_post"].iloc[0]

    assert winner_post > INITIAL_RATING
    assert loser_post < INITIAL_RATING


def test_scheduled_game_does_not_change_rating():
    games = pd.DataFrame([_game("g1", 2025, 1, "A", "B", status="scheduled")])
    elo = compute_elo_ratings(games)

    row_a = elo[elo["team_id"] == "A"].iloc[0]
    assert row_a["elo_pre"] == row_a["elo_post"] == INITIAL_RATING


def test_rating_carries_forward_across_weeks_point_in_time():
    games = pd.DataFrame([
        _game("g1", 2025, 1, "A", "B", 24, 17, "final"),
        _game("g2", 2025, 2, "A", "C", status="scheduled"),
    ])
    elo = compute_elo_ratings(games)

    week1_post_a = elo[(elo["team_id"] == "A") & (elo["game_id"] == "g1")]["elo_post"].iloc[0]
    week2_pre_a = elo[(elo["team_id"] == "A") & (elo["game_id"] == "g2")]["elo_pre"].iloc[0]

    assert week2_pre_a == week1_post_a  # continuity: pre-week-2 rating is exactly post-week-1
    assert week2_pre_a > INITIAL_RATING  # reflects the week 1 win


def test_same_week_games_do_not_affect_each_other():
    games = pd.DataFrame([
        _game("g1", 2025, 1, "A", "B", 24, 17, "final"),
        _game("g2", 2025, 1, "C", "D", status="scheduled"),
    ])
    elo = compute_elo_ratings(games)

    c_pre = elo[(elo["team_id"] == "C") & (elo["game_id"] == "g2")]["elo_pre"].iloc[0]
    assert c_pre == INITIAL_RATING  # unaffected by the simultaneous A/B result


def test_season_regression_pulls_rating_toward_initial():
    games = pd.DataFrame([
        _game("g1", 2025, 1, "A", "B", 24, 0, "final"),  # big win inflates A's rating
        _game("g2", 2026, 1, "A", "C", status="scheduled"),
    ])
    elo = compute_elo_ratings(games)

    season2025_post_a = elo[(elo["team_id"] == "A") & (elo["game_id"] == "g1")]["elo_post"].iloc[0]
    season2026_pre_a = elo[(elo["team_id"] == "A") & (elo["game_id"] == "g2")]["elo_pre"].iloc[0]

    assert season2025_post_a > INITIAL_RATING
    assert INITIAL_RATING < season2026_pre_a < season2025_post_a  # regressed toward, not all the way to, 1500
