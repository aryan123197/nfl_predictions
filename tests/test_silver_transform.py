"""
Tests for the bronze -> silver transform, using an in-memory SQLite
database (offline, no network, no Postgres) -- same pattern as
tests/test_ingestion.py.

Seeds bronze tables via the real ingestion path (run_ingestion.run with
a FakeProvider) rather than hand-writing bronze rows, so these tests
exercise the actual bronze schema the transform reads from.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import text

import src.db as db_module
from src.ingest import run_ingestion
from src.providers.base import NFLDataProvider
from src.transform import silver_transform


class FakeProvider(NFLDataProvider):
    def __init__(self, games: pd.DataFrame, injuries: pd.DataFrame | None = None,
                 players: pd.DataFrame | None = None):
        self._games = games
        self._injuries = injuries if injuries is not None else pd.DataFrame(
            columns=["season", "week", "team", "player_id", "full_name",
                     "position", "report_status", "report_primary_injury",
                     "practice_status", "date_modified"]
        )
        self._players = players if players is not None else pd.DataFrame(
            columns=["player_id", "full_name", "position", "team", "status"]
        )

    def get_games(self, season, week=None):
        df = self._games[self._games["season"] == season]
        if week is not None:
            df = df[df["week"] == week]
        return df.reset_index(drop=True)

    def get_plays(self, season, week=None):
        raise NotImplementedError

    def get_players(self, season):
        return self._players.reset_index(drop=True)

    def get_injuries(self, season, week=None):
        df = self._injuries[self._injuries["season"] == season]
        if week is not None:
            df = df[df["week"] == week]
        return df.reset_index(drop=True)

    def get_odds(self, season, week=None):
        raise NotImplementedError


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    yield db_path


@pytest.fixture()
def sample_games():
    return pd.DataFrame({
        "game_id": ["2025_01_LV_LAC"],
        "season": [2025],
        "week": [1],
        "game_type": ["REG"],
        "gameday": ["2025-09-07"],
        "gametime": ["16:25"],
        "home_team": ["LAC"],
        "away_team": ["OAK"],  # deliberately the old abbreviation: exercises team alias normalization
        "home_score": [24.0],
        "away_score": [17.0],
        "location": ["Home"],
        "spread_line": [-2.5],
        "total_line": [44.0],
        "home_moneyline": [-130.0],
        "away_moneyline": [110.0],
        "home_rest": [7.0],
        "away_rest": [7.0],
        "overtime": [0.0],
        "result": [7.0],
    })


def _seed_bronze(games: pd.DataFrame, injuries: pd.DataFrame | None = None,
                  players: pd.DataFrame | None = None, season: int = 2025, week: int | None = 1):
    provider = FakeProvider(games=games, injuries=injuries, players=players)
    run_ingestion.run(provider, season=season, week=week,
                       skip_injuries=injuries is None, skip_players=players is None)


def test_silver_games_normalizes_team_aliases_and_status(isolated_db, sample_games):
    _seed_bronze(sample_games)
    silver_transform.run(season=2025, week=1)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT home_team_id, away_team_id, status FROM silver_games WHERE game_id = '2025_01_LV_LAC'")
        ).fetchone()

    assert row[0] == "LAC"
    assert row[1] == "LV"  # OAK normalized to canonical LV
    assert row[2] == "final"  # both scores present


def test_silver_games_scheduled_status_when_score_missing(isolated_db, sample_games):
    unplayed = sample_games.copy()
    unplayed.loc[0, "home_score"] = None
    unplayed.loc[0, "away_score"] = None
    _seed_bronze(unplayed)
    silver_transform.run(season=2025, week=1)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        status = conn.execute(
            text("SELECT status FROM silver_games WHERE game_id = '2025_01_LV_LAC'")
        ).scalar()

    assert status == "scheduled"


def test_silver_transform_is_idempotent_for_games(isolated_db, sample_games):
    _seed_bronze(sample_games)
    silver_transform.run(season=2025, week=1)
    silver_transform.run(season=2025, week=1)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM silver_games")).scalar()

    assert count == 1


def test_silver_teams_tracks_season_range(isolated_db, sample_games):
    _seed_bronze(sample_games, season=2025, week=1)
    silver_transform.run(season=2025, week=1)

    games_2024 = sample_games.copy()
    games_2024["season"] = 2024
    games_2024["game_id"] = "2024_01_LV_LAC"
    _seed_bronze(games_2024, season=2024, week=1)
    silver_transform.run(season=2024, week=1)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT first_season, last_season FROM silver_teams WHERE team_id = 'LV'")
        ).fetchone()

    assert row[0] == 2024
    assert row[1] == 2025


def test_silver_odds_derived_from_games(isolated_db, sample_games):
    _seed_bronze(sample_games)
    silver_transform.run(season=2025, week=1)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT sportsbook, source, spread, moneyline_home, moneyline_away, total "
                 "FROM silver_odds WHERE game_id = '2025_01_LV_LAC'")
        ).fetchone()

    assert row[0] == "consensus"
    assert row[1] == "FakeProvider"
    assert row[2] == -2.5
    assert row[3] == -130.0
    assert row[4] == 110.0
    assert row[5] == 44.0


def test_silver_injuries_stay_point_in_time_across_reports(isolated_db, sample_games):
    week1_injury = pd.DataFrame({
        "season": [2025], "week": [1], "team": ["OAK"], "player_id": ["p1"],
        "full_name": ["Test Player"], "position": ["WR"], "report_status": ["Questionable"],
        "report_primary_injury": ["Ankle"], "practice_status": ["Limited"], "date_modified": ["2025-09-04"],
    })
    _seed_bronze(sample_games, injuries=week1_injury, season=2025, week=1)
    silver_transform.run(season=2025, week=1)

    week1_injury_updated = week1_injury.copy()
    week1_injury_updated.loc[0, "report_status"] = "Out"
    week1_injury_updated.loc[0, "date_modified"] = "2025-09-05"
    _seed_bronze(sample_games, injuries=week1_injury_updated, season=2025, week=1)
    silver_transform.run(season=2025, week=1)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT status, team_id FROM silver_injuries ORDER BY reported_at")
        ).fetchall()

    assert len(rows) == 2  # both report snapshots kept, not collapsed to "current status"
    assert [r[0] for r in rows] == ["Questionable", "Out"]
    assert rows[0][1] == "LV"  # team alias normalized here too


def test_silver_injuries_transform_is_idempotent(isolated_db, sample_games):
    injury = pd.DataFrame({
        "season": [2025], "week": [1], "team": ["OAK"], "player_id": ["p1"],
        "full_name": ["Test Player"], "position": ["WR"], "report_status": ["Questionable"],
        "report_primary_injury": ["Ankle"], "practice_status": ["Limited"], "date_modified": ["2025-09-04"],
    })
    _seed_bronze(sample_games, injuries=injury, season=2025, week=1)
    silver_transform.run(season=2025, week=1)
    silver_transform.run(season=2025, week=1)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM silver_injuries")).scalar()

    assert count == 1  # same bronze row re-transformed, not duplicated


def test_silver_players_normalizes_team_and_upserts(isolated_db, sample_games):
    players = pd.DataFrame({
        "player_id": ["p1"], "full_name": ["Test Player"], "position": ["WR"],
        "team": ["OAK"], "status": ["ACT"],
    })
    _seed_bronze(sample_games, players=players, season=2025, week=1)
    silver_transform.run(season=2025, week=1)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT team_id, status FROM silver_players WHERE player_id = 'p1'")
        ).fetchone()

    assert row[0] == "LV"
    assert row[1] == "ACT"
