"""
Tests for Phase 7: State management, Dynamic schedule resolver, and Health monitoring.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from src.db import (
    apply_ddl_file,
    get_pipeline_state,
    init_schema,
    update_pipeline_state,
)
from src.pipeline.health_check import HealthCheckResult, run_health_check
from src.pipeline.schedule_resolver import (
    get_active_week,
    get_current_season,
    get_upcoming_games,
)


@pytest.fixture
def sqlite_engine(monkeypatch, tmp_path):
    db_file = tmp_path / "test_state.db"
    db_url = f"sqlite:///{db_file}"
    engine = create_engine(db_url)
    monkeypatch.setenv("DATABASE_URL", db_url)
    import src.db as db_module
    monkeypatch.setattr(db_module, "_engine", engine)
    init_schema(force=True)
    return engine


def test_get_current_season():
    # September 2026 -> Season 2026
    assert get_current_season(date(2026, 9, 10)) == 2026
    # December 2026 -> Season 2026
    assert get_current_season(date(2026, 12, 25)) == 2026
    # January 2027 (Playoffs of 2026 season) -> Season 2026
    assert get_current_season(date(2027, 1, 15)) == 2026
    # February 2027 (Super Bowl) -> Season 2026
    assert get_current_season(date(2027, 2, 8)) == 2026
    # March 2027 (New league year) -> Season 2027
    assert get_current_season(date(2027, 3, 15)) == 2027


def test_get_active_week_calendar_fallback():
    # Before season starts (August) -> Week 1
    assert get_active_week(season=2026, as_of_date=date(2026, 8, 20)) == 1
    # Week 1 (~Sept 10, 2026) -> Week 1
    w1 = get_active_week(season=2026, as_of_date=date(2026, 9, 10))
    assert w1 == 1
    # 4 weeks later (~Oct 8, 2026) -> Week 5
    w5 = get_active_week(season=2026, as_of_date=date(2026, 10, 8))
    assert w5 == 5


def test_pipeline_state_crud(sqlite_engine):
    # Initially none
    state = get_pipeline_state("nfl_data_pipeline")
    assert state is None

    # Insert state
    update_pipeline_state(
        pipeline_name="nfl_data_pipeline",
        current_season=2026,
        current_week=1,
        last_successful_run_id=1,
        status="idle",
        metadata_json=json.dumps({"test": "value"}),
    )

    state = get_pipeline_state("nfl_data_pipeline")
    assert state is not None
    assert state["pipeline_name"] == "nfl_data_pipeline"
    assert state["current_season"] == 2026
    assert state["current_week"] == 1
    assert state["last_successful_run_id"] == 1
    assert state["status"] == "idle"

    # Update state
    update_pipeline_state(
        pipeline_name="nfl_data_pipeline",
        current_week=2,
        last_successful_run_id=2,
        status="running",
    )

    updated = get_pipeline_state("nfl_data_pipeline")
    assert updated["current_week"] == 2
    assert updated["last_successful_run_id"] == 2
    assert updated["status"] == "running"
    assert updated["current_season"] == 2026


def test_health_check_clean(sqlite_engine):
    # Seed 32 teams and clean game / prediction data
    with sqlite_engine.begin() as conn:
        for i in range(32):
            conn.execute(
                text(f"INSERT INTO silver_teams (team_id, first_season, last_season) VALUES ('T{i}', 2020, 2026)")
            )
        conn.execute(text("""
            INSERT INTO silver_games (game_id, season, week, game_date, home_team_id, away_team_id, status)
            VALUES ('2026_01_T0_T1', 2026, 1, '2026-09-10', 'T0', 'T1', 'scheduled')
        """))
        conn.execute(text("""
            INSERT INTO ml_predictions (game_id, model_version, home_win_probability, away_win_probability, cover_probability)
            VALUES ('2026_01_T0_T1', 'v1', 0.60, 0.40, 0.52)
        """))
        conn.execute(text("""
            INSERT INTO metadata_pipeline_state (pipeline_name, current_season, current_week, status)
            VALUES ('nfl_data_pipeline', 2026, 1, 'idle')
        """))

    res = run_health_check(season=2026, week=1, engine=sqlite_engine)
    assert res.status in ("HEALTHY", "WARNING")  # No hard errors
    assert len(res.errors) == 0
    assert "Silver teams count is exactly 32" in res.passed
    assert "All 1 predictions satisfy P(Home) + P(Away) = 1.0" in res.passed


def test_health_check_catches_invalid_probabilities(sqlite_engine):
    with sqlite_engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO silver_games (game_id, season, week, game_date, home_team_id, away_team_id, status)
            VALUES ('2026_01_T0_T1', 2026, 1, '2026-09-10', 'T0', 'T1', 'scheduled')
        """))
        # Invalid: probabilities don't sum to 1.0 and exceed [0, 1]
        conn.execute(text("""
            INSERT INTO ml_predictions (game_id, model_version, home_win_probability, away_win_probability, cover_probability)
            VALUES ('2026_01_T0_T1', 'v1', 1.50, 0.40, 0.52)
        """))

    res = run_health_check(season=2026, week=1, engine=sqlite_engine)
    assert res.status == "UNHEALTHY"
    assert any("out-of-bounds" in e for e in res.errors)
