"""
Phase 6 Tests: Automated Data Pipeline & ML Retraining Orchestration.

Tests pipeline execution, champion registry management, prediction serving,
and metadata.pipeline_runs tracking using in-memory SQLite databases and
deterministic mock providers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

import src.db as db_module
from src.ml import registry
from src.ml.features import FEATURE_COLUMNS
from src.ml.train import NFLPredictionModel
from src.pipeline import run_pipeline, run_retrain
from src.providers.base import NFLDataProvider
from src.transform import game_results_transform, gold_transform, silver_transform
import xgboost as xgb


class FakeProvider(NFLDataProvider):
    """Deterministic in-memory provider for pipeline tests."""

    def __init__(self, games: pd.DataFrame, injuries: pd.DataFrame | None = None, plays: pd.DataFrame | None = None):
        self._games = games
        self._injuries = injuries if injuries is not None else pd.DataFrame(
            columns=["season", "week", "team", "player_id", "full_name",
                     "position", "report_status", "report_primary_injury",
                     "practice_status", "date_modified"]
        )
        self._plays = plays if plays is not None else pd.DataFrame(
            columns=["play_id", "game_id", "season", "week", "qtr", "down",
                     "ydstogo", "yardline_100", "play_type", "yards_gained",
                     "epa", "success", "pass", "rush", "posteam", "defteam",
                     "sack", "qb_hit", "first_down", "third_down_converted",
                     "third_down_failed", "touchdown", "interception", "fumble_lost"]
        )

    def get_games(self, season, week=None):
        df = self._games[self._games["season"] == season]
        if week is not None:
            df = df[df["week"] == week]
        return df.reset_index(drop=True)

    def get_plays(self, season, week=None):
        df = self._plays[self._plays["season"] == season]
        if week is not None:
            df = df[df["week"] == week]
        return df.reset_index(drop=True)

    def get_players(self, season):
        return pd.DataFrame(columns=["player_id", "full_name", "position", "team", "status"])

    def get_injuries(self, season, week=None):
        df = self._injuries[self._injuries["season"] == season]
        if week is not None:
            df = df[df["week"] == week]
        return df.reset_index(drop=True)

    def get_odds(self, season, week=None):
        raise NotImplementedError


class BrokenProvider(NFLDataProvider):
    """Provider that raises an error to test failure handling."""

    def get_games(self, season, week=None):
        raise ConnectionError("Upstream API is unreachable")

    def get_plays(self, season, week=None):
        raise ConnectionError("Upstream API is unreachable")

    def get_players(self, season):
        raise ConnectionError("Upstream API is unreachable")

    def get_injuries(self, season, week=None):
        raise ConnectionError("Upstream API is unreachable")

    def get_odds(self, season, week=None):
        raise ConnectionError("Upstream API is unreachable")


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    yield db_path


@pytest.fixture()
def sample_season_games():
    """Generates two seasons of games with balanced winners and scheduled games."""
    rows = []
    teams = [("KC", "LV"), ("BUF", "MIA"), ("DAL", "PHI"), ("SF", "LAR")]
    # Season 2024: 4 weeks of completed games (balanced home/away winners)
    for w in range(1, 5):
        for idx, (home, away) in enumerate(teams):
            home_won = (w + idx) % 2 == 0
            rows.append({
                "game_id": f"2024_{w:02d}_{away}_{home}",
                "season": 2024,
                "week": w,
                "game_type": "REG",
                "gameday": f"2024-09-{(w * 7):02d}",
                "home_team": home,
                "away_team": away,
                "home_score": 27 + idx if home_won else 17,
                "away_score": 17 if home_won else 24 + idx,
                "spread_line": -3.5,
                "total_line": 45.5,
                "home_moneyline": -180 if home_won else 150,
                "away_moneyline": 150 if home_won else -180,
            })

    # Season 2025: Week 1 completed, Week 2 scheduled (no scores)
    for idx, (home, away) in enumerate(teams):
        home_won = idx % 2 == 0
        rows.append({
            "game_id": f"2025_01_{away}_{home}",
            "season": 2025,
            "week": 1,
            "game_type": "REG",
            "gameday": "2025-09-07",
            "home_team": home,
            "away_team": away,
            "home_score": 27 + idx if home_won else 17,
            "away_score": 17 if home_won else 24 + idx,
            "spread_line": -4.0,
            "total_line": 48.0,
            "home_moneyline": -190,
            "away_moneyline": 160,
        })
        rows.append({
            "game_id": f"2025_02_{away}_{home}",
            "season": 2025,
            "week": 2,
            "game_type": "REG",
            "gameday": "2025-09-14",
            "home_team": home,
            "away_team": away,
            "home_score": None,
            "away_score": None,
            "spread_line": -3.0,
            "total_line": 44.0,
            "home_moneyline": -150,
            "away_moneyline": 130,
        })
    return pd.DataFrame(rows)


def _build_dummy_model() -> NFLPredictionModel:
    X_dummy = pd.DataFrame(np.random.randn(20, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
    y_win = pd.Series([1, 0] * 10)
    y_margin = pd.Series([7.0, -3.0] * 10)
    y_total = pd.Series([45.0, 41.0] * 10)

    win = xgb.XGBClassifier(n_estimators=5, max_depth=2)
    win.fit(X_dummy, y_win)
    margin = xgb.XGBRegressor(n_estimators=5, max_depth=2)
    margin.fit(X_dummy, y_margin)
    total = xgb.XGBRegressor(n_estimators=5, max_depth=2)
    total.fit(X_dummy, y_total)

    return NFLPredictionModel(
        win_model=win,
        margin_model=margin,
        total_model=total,
    )


def test_champion_registration_and_loading(tmp_path, monkeypatch):
    """Verifies register_champion saves artifact bundle and load_champion_model reads it back."""
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(registry, "CHAMPION_DIR", tmp_path / "models" / "champion")

    dummy_model = _build_dummy_model()
    metrics = {"accuracy": 0.625, "log_loss": 0.650, "brier_score": 0.220}

    # Before registration: load returns (None, None)
    m, meta = registry.load_champion_model()
    assert m is None
    assert meta is None

    # Register
    version_dir = registry.register_champion(
        model=dummy_model,
        metrics=metrics,
        model_version="test_champion_v1",
        training_metadata={"split": "test_split"},
    )
    assert version_dir.exists()

    # After registration: load returns model and metadata
    loaded_model, loaded_meta = registry.load_champion_model()
    assert loaded_model is not None
    assert loaded_meta["model_version"] == "test_champion_v1"
    assert loaded_meta["metrics"]["accuracy"] == 0.625


def test_run_pipeline_end_to_end(sample_season_games, isolated_db, tmp_path, monkeypatch):
    """Verifies run_pipeline orchestrates ingestion -> silver -> gold -> predictions with pipeline_runs tracking."""
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(registry, "CHAMPION_DIR", tmp_path / "models" / "champion")

    provider = FakeProvider(sample_season_games)

    # Seed model
    dummy_model = _build_dummy_model()
    registry.register_champion(dummy_model, {"accuracy": 0.65}, "test_pipeline_champ")

    result = run_pipeline.run(season=2025, skip_plays=True, provider=provider)
    assert result["status"] == "success"
    assert result["records_processed"] > 0
    assert result["predictions_written"] > 0

    # Verify metadata.pipeline_runs
    engine = db_module.get_engine()
    with engine.begin() as conn:
        run_row = conn.execute(
            text("SELECT pipeline_name, status, records_processed, error_message FROM metadata_pipeline_runs WHERE run_id = :id"),
            {"id": result["run_id"]},
        ).mappings().first()
        assert run_row["pipeline_name"] == "nfl_data_pipeline"
        assert run_row["status"] == "success"
        assert run_row["error_message"] is None

        # Verify predictions were recorded
        pred_count = conn.execute(text("SELECT COUNT(*) FROM ml_predictions")).scalar_one()
        assert pred_count > 0


def test_run_pipeline_records_failure_on_exception(isolated_db):
    """Verifies that an error in the pipeline marks the run as failed in metadata.pipeline_runs."""
    broken_provider = BrokenProvider()

    with pytest.raises(ConnectionError):
        run_pipeline.run(season=2025, provider=broken_provider)

    engine = db_module.get_engine()
    with engine.begin() as conn:
        run_row = conn.execute(
            text("SELECT pipeline_name, status, error_message FROM metadata_pipeline_runs ORDER BY run_id DESC LIMIT 1")
        ).mappings().first()
        assert run_row["pipeline_name"] == "nfl_data_pipeline"
        assert run_row["status"] == "failed"
        assert "Upstream API is unreachable" in run_row["error_message"]


def test_run_retrain_cadence_skip(isolated_db):
    """Verifies that run_retrain skips model training during early weeks per Decision #4."""
    result = run_retrain.run(season=2025, week=2, force_retrain=False)

    assert result["status"] == "skipped_cadence_gate"
    assert result["promoted"] is False
    assert "Skipping model weight retraining for Week 2" in result["message"]

    engine = db_module.get_engine()
    with engine.begin() as conn:
        run_row = conn.execute(
            text("SELECT status FROM metadata_pipeline_runs WHERE run_id = :id"),
            {"id": result["run_id"]},
        ).mappings().first()
        assert run_row["status"] == "success"


def test_run_retrain_end_to_end(sample_season_games, isolated_db, tmp_path, monkeypatch):
    """Verifies run_retrain trains candidate, evaluates via promotion gate, and records run."""
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(registry, "CHAMPION_DIR", tmp_path / "models" / "champion")

    provider = FakeProvider(sample_season_games)

    # Ingest and prepare gold layer for multiple seasons
    from src.ingest.run_ingestion import run as run_ingestion
    run_ingestion(provider, season=2024)
    silver_transform.run(season=2024)
    run_ingestion(provider, season=2025)
    silver_transform.run(season=2025)
    gold_transform.run()
    game_results_transform.run()

    # Execute retraining with force_retrain=True
    result = run_retrain.run(season=2025, week=2, force_retrain=True)
    assert result["status"] == "success"
    assert "candidate_version" in result
    assert "candidate_metrics" in result

    # Check pipeline runs table
    engine = db_module.get_engine()
    with engine.begin() as conn:
        run_row = conn.execute(
            text("SELECT status, records_processed FROM metadata_pipeline_runs WHERE run_id = :id"),
            {"id": result["run_id"]},
        ).mappings().first()
        assert run_row["status"] == "success"
        assert run_row["records_processed"] > 0
