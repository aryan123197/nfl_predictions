"""
API tests -- offline, against a real (SQLite) database.

Uses the same in-memory-ish SQLite approach the transform tests use, so
these run in CI with no network and no Postgres. The point is to exercise
the actual SQL against a real schema, not to mock the database away: a
mocked query would still pass if a column name were wrong, which is exactly
the bug class that has bitten this project before (elo_rating vs elo_post).
"""

from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("fastapi", reason="requirements-api.txt not installed")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    """A TestClient bound to a fresh, fully-migrated SQLite database.

    src.db caches its engine in a module global, so the env var must be set
    AND the cache cleared before anything touches the database.
    """
    db_path = os.path.join(tempfile.mkdtemp(), "api_test.db")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    import src.db as db

    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_SessionLocal", None)
    db.init_schema()
    _seed(db)

    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def _seed(db) -> None:
    engine = db.get_engine()
    games = db.qualified_table("silver", "games")
    teams = db.qualified_table("silver", "teams")
    features = db.qualified_table("gold", "game_features")
    ratings = db.qualified_table("gold", "team_ratings")

    with engine.begin() as conn:
        for team in ("KC", "BUF", "SF"):
            conn.execute(
                text(f"INSERT INTO {teams} (team_id, first_season, last_season) VALUES (:t, 2020, 2025)"),
                {"t": team},
            )
        # A completed game and a scheduled one -- the scheduled game is the
        # interesting case: null score, and (before Phase 4) null prediction.
        conn.execute(
            text(
                f"INSERT INTO {games} (game_id, season, week, game_date, home_team_id, away_team_id, "
                "home_score, away_score, home_rest_days, away_rest_days, status) VALUES "
                "('2025_01_BUF_KC', 2025, 1, '2025-09-07', 'KC', 'BUF', 27, 23, 7, 7, 'final')"
            )
        )
        conn.execute(
            text(
                f"INSERT INTO {games} (game_id, season, week, game_date, home_team_id, away_team_id, "
                "status) VALUES ('2025_02_SF_KC', 2025, 2, '2025-09-14', 'KC', 'SF', 'scheduled')"
            )
        )
        conn.execute(
            text(
                f"INSERT INTO {features} (game_id, season, week, home_team_id, away_team_id, "
                "home_elo, away_elo, elo_difference, current_spread) VALUES "
                "('2025_01_BUF_KC', 2025, 1, 'KC', 'BUF', 1650.0, 1600.0, 50.0, -2.5)"
            )
        )
        conn.execute(
            text(
                f"INSERT INTO {ratings} (team_id, game_id, season, week, elo_pre, elo_post) VALUES "
                "('KC', '2025_01_BUF_KC', 2025, 1, 1650.0, 1665.0)"
            )
        )


# -- health / meta ----------------------------------------------------------

def test_health_reports_warehouse_ready_and_no_predictions_yet(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["warehouse_ready"] is True
    # The whole point of the Phase 4 boundary: the API runs fine without it.
    assert body["predictions_available"] is False
    assert body["seasons"] == [2025]


def test_seasons_and_weeks_come_from_the_data(client):
    assert client.get("/seasons").json() == [2025]
    assert client.get("/seasons/2025/weeks").json() == [1, 2]


# -- games ------------------------------------------------------------------

def test_list_games_joins_features_onto_games(client):
    games = client.get("/games", params={"season": 2025}).json()
    assert [g["game_id"] for g in games] == ["2025_01_BUF_KC", "2025_02_SF_KC"]

    played = games[0]
    assert played["status"] == "final"
    assert played["home_score"] == 27
    assert played["features"]["home_elo"] == 1650.0
    assert played["features"]["elo_difference"] == 50.0


def test_scheduled_game_has_null_score_not_zero(client):
    """A scheduled game must not report 0-0 -- that reads as a real result."""
    game = client.get("/games/2025_02_SF_KC").json()
    assert game["status"] == "scheduled"
    assert game["home_score"] is None
    assert game["away_score"] is None


def test_game_without_gold_features_still_returned(client):
    """LEFT JOIN, not INNER: a game the gold transform hasn't covered yet is
    a real state and must not vanish from the week view."""
    game = client.get("/games/2025_02_SF_KC").json()
    assert game["features"]["home_elo"] is None


def test_week_endpoint_filters_to_that_week(client):
    games = client.get("/games/week/1", params={"season": 2025}).json()
    assert [g["game_id"] for g in games] == ["2025_01_BUF_KC"]


def test_unknown_game_is_404(client):
    assert client.get("/games/does_not_exist").status_code == 404


# -- teams ------------------------------------------------------------------

def test_team_elo_uses_post_game_rating(client):
    """elo_post (1665), not elo_pre (1650) -- current strength is the rating
    after the most recent rated game."""
    team = client.get("/teams/KC").json()
    assert team["elo"] == 1665.0


def test_team_without_ratings_has_null_elo(client):
    assert client.get("/teams/SF").json()["elo"] is None


def test_unknown_team_is_404(client):
    assert client.get("/teams/XXX").status_code == 404


# -- the Phase 4 boundary ---------------------------------------------------

def test_prediction_absent_is_404_naming_the_missing_phase(client):
    response = client.get("/predictions/2025_01_BUF_KC")
    assert response.status_code == 404
    # The message must distinguish "no model yet" from "no such game",
    # otherwise a frontend can't tell a pipeline gap from a bad URL.
    assert "Phase 4" in response.json()["detail"]


def test_games_carry_null_prediction_before_phase_4(client):
    game = client.get("/games/2025_01_BUF_KC").json()
    assert game["prediction"] is None


def test_model_performance_reports_unavailable_not_zeroed_metrics(client):
    body = client.get("/model/performance").json()
    assert body["available"] is False
    assert body["accuracy"] is None  # not 0.0 -- that would read as a terrible model
    assert "Phase 4" in body["reason"]


def test_predictions_light_up_when_phase_4_lands_the_contract_table(client):
    """The boundary's real test: create ml.predictions exactly as design doc
    section 21 specifies, and predictions should appear with no code change.
    """
    import src.db as db

    table = db.qualified_table("ml", "predictions")
    with db.get_engine().begin() as conn:
        conn.execute(
            text(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                "prediction_id INTEGER PRIMARY KEY AUTOINCREMENT, game_id TEXT, "
                "model_version TEXT, prediction_timestamp TEXT, "
                "home_win_probability REAL, away_win_probability REAL, "
                "predicted_home_score REAL, predicted_away_score REAL, "
                "predicted_margin REAL, market_spread REAL)"
            )
        )
        conn.execute(
            text(
                f"INSERT INTO {table} (game_id, model_version, prediction_timestamp, "
                "home_win_probability, away_win_probability, predicted_home_score, "
                "predicted_away_score, predicted_margin, market_spread) VALUES "
                "('2025_01_BUF_KC', 'v0.1', '2025-09-06T12:00:00', 0.68, 0.32, 27.3, 23.1, 4.2, -2.5)"
            )
        )

    assert client.get("/health").json()["predictions_available"] is True

    prediction = client.get("/predictions/2025_01_BUF_KC").json()
    assert prediction["home_win_probability"] == 0.68
    assert prediction["model_version"] == "v0.1"

    # ...and it should be nested onto the game, not only on its own endpoint.
    game = client.get("/games/2025_01_BUF_KC").json()
    assert game["prediction"]["predicted_margin"] == 4.2


def test_latest_prediction_wins_when_a_game_is_repredicted(client):
    """Weekly retraining (design doc section 26) means one game accumulates
    several predictions; the UI must show the newest, not an arbitrary one."""
    import src.db as db

    table = db.qualified_table("ml", "predictions")
    with db.get_engine().begin() as conn:
        conn.execute(
            text(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                "prediction_id INTEGER PRIMARY KEY AUTOINCREMENT, game_id TEXT, "
                "model_version TEXT, prediction_timestamp TEXT, "
                "home_win_probability REAL, away_win_probability REAL, "
                "predicted_home_score REAL, predicted_away_score REAL, "
                "predicted_margin REAL, market_spread REAL)"
            )
        )
        for version, ts, prob in (("v0.1", "2025-09-01T12:00:00", 0.55),
                                  ("v0.2", "2025-09-06T12:00:00", 0.68)):
            conn.execute(
                text(
                    f"INSERT INTO {table} (game_id, model_version, prediction_timestamp, "
                    "home_win_probability, away_win_probability) VALUES (:g, :v, :ts, :p, :away_p)"
                ),
                {"g": "2025_01_BUF_KC", "v": version, "ts": ts, "p": prob, "away_p": 1.0 - prob},
            )

    assert client.get("/predictions/2025_01_BUF_KC").json()["model_version"] == "v0.2"
    games = client.get("/games", params={"season": 2025}).json()
    predicted = [g for g in games if g["prediction"]][0]
    assert predicted["prediction"]["model_version"] == "v0.2"


def test_model_performance_with_evaluated_games(client):
    """When predictions exist on completed games, /model/performance computes live metrics."""
    import src.db as db

    table = db.qualified_table("ml", "predictions")
    with db.get_engine().begin() as conn:
        conn.execute(
            text(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                "prediction_id INTEGER PRIMARY KEY AUTOINCREMENT, game_id TEXT, "
                "model_version TEXT, prediction_timestamp TEXT, "
                "home_win_probability REAL, away_win_probability REAL, "
                "predicted_home_score REAL, predicted_away_score REAL, "
                "predicted_margin REAL, market_spread REAL)"
            )
        )
        # 2025_01_BUF_KC has home_score=27, away_score=23 (KC won by 4)
        conn.execute(
            text(
                f"INSERT INTO {table} (game_id, model_version, prediction_timestamp, "
                "home_win_probability, away_win_probability, predicted_home_score, "
                "predicted_away_score, predicted_margin, market_spread) VALUES "
                "('2025_01_BUF_KC', 'v0.1', '2025-09-06T12:00:00', 0.68, 0.32, 27.0, 23.0, 4.0, -2.5)"
            )
        )

    res = client.get("/model/performance")
    assert res.status_code == 200
    perf = res.json()
    assert perf["available"] is True
    assert perf["games_evaluated"] == 1
    assert perf["accuracy"] == 1.0
    assert perf["model_version"] == "v0.1"
    assert perf["mae_margin"] == 0.0


def test_monitoring_health_endpoint(client):
    res = client.get("/monitoring/health")
    assert res.status_code == 200
    health = res.json()
    assert "status" in health
    assert "passed_count" in health
    assert "warnings_count" in health
    assert "errors_count" in health


def test_explanation_endpoint_returns_404_when_no_game(client):
    res = client.get("/model/explanation/unknown_game_id")
    assert res.status_code == 404


def test_unbuilt_player_endpoint_returns_501(client):
    assert client.get("/players/00-0033873").status_code == 501

