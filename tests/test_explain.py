"""
Tests for Phase 8: Model Explainability Engine (TreeSHAP feature attributions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sqlalchemy import text

import src.db as db_module
from src.ml.explain import _human_readable_impact, explain_game_prediction
from src.ml.features import FEATURE_COLUMNS
from src.ml.train import NFLPredictionModel


@pytest.fixture
def isolated_engine(tmp_path, monkeypatch):
    db_path = tmp_path / "explain_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    engine = db_module.get_engine()
    db_module.init_schema()
    return engine


@pytest.fixture
def trained_mock_model():
    """Create and train a minimal fast NFLPredictionModel for testing."""
    np.random.seed(42)
    n_samples = 40
    X = pd.DataFrame(np.random.randn(n_samples, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
    y_win = (X["elo_difference"] > 0).astype(int)
    y_margin = X["elo_difference"] * 0.1 + np.random.randn(n_samples) * 2
    y_total = 45.0 + np.random.randn(n_samples) * 5

    win_model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42, eval_metric="logloss")
    win_model.fit(X, y_win)

    margin_model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42, eval_metric="mae")
    margin_model.fit(X, y_margin)

    total_model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42, eval_metric="mae")
    total_model.fit(X, y_total)

    model = NFLPredictionModel(
        win_model=win_model,
        margin_model=margin_model,
        total_model=total_model,
        margin_std=3.5,
    )
    return model



def test_human_readable_impact():
    assert "Higher Elo strength for KC" in _human_readable_impact("elo_difference", 45.0, 0.35, "KC", "BAL")
    assert "Stronger Elo rating for BAL" in _human_readable_impact("elo_difference", -45.0, -0.35, "KC", "BAL")
    assert "rest days" in _human_readable_impact("home_rest_days", 7.0, 0.1, "KC", "BAL")
    assert "Clean injury report" in _human_readable_impact("home_injury_impact", 0.0, 0.0, "KC", "BAL")
    assert "Injury impact penalty" in _human_readable_impact("home_injury_impact", -2.5, -0.2, "KC", "BAL")


def test_explain_game_prediction_structure(isolated_engine, trained_mock_model):
    engine = isolated_engine

    gf_table = db_module.qualified_table("gold", "game_features")
    sg_table = db_module.qualified_table("silver", "games")

    with engine.begin() as conn:
        conn.execute(
            text(f"""
                INSERT INTO {sg_table} (game_id, season, week, home_team_id, away_team_id, status)
                VALUES ('2024_01_BAL_KC', 2024, 1, 'KC', 'BAL', 'final')
            """)
        )
        col_list = ["game_id", "season", "week", "home_team_id", "away_team_id"] + FEATURE_COLUMNS
        col_sql = ", ".join(col_list)
        placeholders = ", ".join(f":{c}" for c in col_list)
        data = {
            "game_id": "2024_01_BAL_KC",
            "season": 2024,
            "week": 1,
            "home_team_id": "KC",
            "away_team_id": "BAL",
            **{c: 0.5 for c in FEATURE_COLUMNS},
        }
        data["elo_difference"] = 65.0
        data["home_elo"] = 1680.0
        data["away_elo"] = 1615.0
        data["current_spread"] = -3.0

        conn.execute(text(f"INSERT INTO {gf_table} ({col_sql}) VALUES ({placeholders})"), data)

    explanation = explain_game_prediction("2024_01_BAL_KC", engine=engine, model=trained_mock_model)

    assert explanation is not None
    assert explanation["game_id"] == "2024_01_BAL_KC"
    assert explanation["home_team_id"] == "KC"
    assert explanation["away_team_id"] == "BAL"
    assert "favored_team" in explanation
    assert "win_probability" in explanation
    assert "base_probability" in explanation
    assert 0.0 <= explanation["base_probability"] <= 1.0
    assert "top_positive_factors" in explanation
    assert "top_negative_factors" in explanation

    # Check factor schema
    for f in explanation["top_positive_factors"]:
        assert "feature" in f
        assert "display_name" in f
        assert "contribution" in f
        assert f["contribution"] > 0
        assert "description" in f

    for f in explanation["top_negative_factors"]:
        assert "feature" in f
        assert "contribution" in f
        assert f["contribution"] <= 0


def test_explain_nonexistent_game(isolated_engine, trained_mock_model):
    explanation = explain_game_prediction("2099_99_NON_EXISTENT", engine=isolated_engine, model=trained_mock_model)
    assert explanation is None
