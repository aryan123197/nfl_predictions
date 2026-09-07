"""
Tests for the ingestion pipeline logic, using an in-memory SQLite
database so these run fast and offline (no network, no Postgres).

Uses a fake provider so we can test upsert/append semantics precisely,
without depending on nflverse's live data or network access.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import text

import src.db as db_module
from src.ingest import run_ingestion
from src.providers.base import NFLDataProvider


class FakeProvider(NFLDataProvider):
    """Deterministic in-memory provider for testing ingestion logic."""

    def __init__(self, games: pd.DataFrame, injuries: pd.DataFrame | None = None):
        self._games = games
        self._injuries = injuries if injuries is not None else pd.DataFrame(
            columns=["season", "week", "team", "player_id", "full_name",
                     "position", "report_status", "report_primary_injury",
                     "practice_status", "date_modified"]
        )

    def get_games(self, season, week=None):
        df = self._games[self._games["season"] == season]
        if week is not None:
            df = df[df["week"] == week]
        return df.reset_index(drop=True)

    def get_plays(self, season, week=None):
        raise NotImplementedError

    def get_players(self, season):
        return pd.DataFrame(columns=["player_id", "full_name", "position", "team", "status"])

    def get_injuries(self, season, week=None):
        df = self._injuries[self._injuries["season"] == season]
        if week is not None:
            df = df[df["week"] == week]
        return df.reset_index(drop=True)

    def get_odds(self, season, week=None):
        raise NotImplementedError


@pytest.fixture()
def sample_games():
    return pd.DataFrame({
        "game_id": ["2025_01_KC_LAC"],
        "season": [2025],
        "week": [1],
        "game_type": ["REG"],
        "gameday": ["2025-09-05"],
        "gametime": ["20:20"],
        "home_team": ["LAC"],
        "away_team": ["KC"],
        "home_score": [27.0],
        "away_score": [21.0],
        "location": ["Home"],
        "spread_line": [-3.0],
        "total_line": [45.5],
        "home_moneyline": [-150.0],
        "away_moneyline": [130.0],
        "home_rest": [7.0],
        "away_rest": [7.0],
        "overtime": [0.0],
        "result": [6.0],
    })


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """Point src.db at a throwaway SQLite file for the duration of the test."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    # reset module-level singletons so the new DATABASE_URL takes effect
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    yield db_path


def test_ingestion_creates_schema_and_inserts_games(isolated_db, sample_games):
    provider = FakeProvider(games=sample_games)
    run_ingestion.run(provider, season=2025, week=1, skip_injuries=True, skip_players=True)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT game_id, home_score, away_score FROM bronze_games_raw")).fetchall()

    assert len(rows) == 1
    assert rows[0][0] == "2025_01_KC_LAC"
    assert rows[0][1] == 27.0


def test_ingestion_is_idempotent_for_games(isolated_db, sample_games):
    provider = FakeProvider(games=sample_games)
    run_ingestion.run(provider, season=2025, week=1, skip_injuries=True, skip_players=True)
    run_ingestion.run(provider, season=2025, week=1, skip_injuries=True, skip_players=True)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM bronze_games_raw")).scalar()

    assert count == 1  # upsert, not duplicate


def test_ingestion_updates_score_on_rerun(isolated_db, sample_games):
    provider = FakeProvider(games=sample_games)
    run_ingestion.run(provider, season=2025, week=1, skip_injuries=True, skip_players=True)

    updated_games = sample_games.copy()
    updated_games.loc[0, "home_score"] = 30.0  # simulate a late score correction
    provider_v2 = FakeProvider(games=updated_games)
    run_ingestion.run(provider_v2, season=2025, week=1, skip_injuries=True, skip_players=True)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        score = conn.execute(text("SELECT home_score FROM bronze_games_raw WHERE game_id = '2025_01_KC_LAC'")).scalar()

    assert score == 30.0


def test_pipeline_run_recorded_as_success(isolated_db, sample_games):
    provider = FakeProvider(games=sample_games)
    run_ingestion.run(provider, season=2025, week=1, skip_injuries=True, skip_players=True)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("SELECT status, records_processed FROM metadata_pipeline_runs")).fetchone()

    assert row[0] == "success"
    assert row[1] == 1


def test_pipeline_run_recorded_as_failed_on_provider_error(isolated_db):
    class BrokenProvider(FakeProvider):
        def get_games(self, season, week=None):
            from src.providers.base import ProviderError
            raise ProviderError("simulated upstream failure")

    provider = BrokenProvider(games=pd.DataFrame())
    with pytest.raises(Exception):
        run_ingestion.run(provider, season=2025, week=1, skip_injuries=True, skip_players=True)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("SELECT status, error_message FROM metadata_pipeline_runs")).fetchone()

    assert row[0] == "failed"
    assert "simulated upstream failure" in row[1]


def test_injuries_are_append_only_across_runs(isolated_db, sample_games):
    injuries = pd.DataFrame({
        "season": [2025], "week": [1], "team": ["KC"], "player_id": ["p1"],
        "full_name": ["Test Player"], "position": ["WR"], "report_status": ["Questionable"],
        "report_primary_injury": ["Ankle"], "practice_status": ["Limited"], "date_modified": ["2025-09-04"],
    })
    provider = FakeProvider(games=sample_games, injuries=injuries)
    run_ingestion.run(provider, season=2025, week=1, skip_players=True)
    run_ingestion.run(provider, season=2025, week=1, skip_players=True)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM bronze_injuries_raw")).scalar()

    assert count == 2  # append-only: same report inserted twice = 2 rows
