"""
Tests for Phase 5: Walk-Forward Backtesting and Model Promotion.

Tests include:
- Pure unit tests for src/ml/promotion.py (tolerance boundaries, regression gates, improvements)
- Full walk-forward simulation tests against an in-memory SQLite database seeded through
  the real ingestion/transform path.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import text

import src.db as db_module
from src.ingest import run_ingestion
from src.ml import backtest as ml_backtest
from src.ml.promotion import evaluate_candidate_promotion, DEFAULT_TOLERANCES
from src.providers.base import NFLDataProvider
from src.transform import game_results_transform, gold_transform, silver_transform


# -- Unit tests: Candidate Model Promotion Gate (Decision #1) ----------------


def test_promotion_when_improving_log_loss_within_tolerances():
    champ = {"accuracy": 0.65, "log_loss": 0.640, "brier_score": 0.220}
    cand = {"accuracy": 0.65, "log_loss": 0.620, "brier_score": 0.218}

    decision = evaluate_candidate_promotion(champ, cand)
    assert decision.promoted is True
    assert "log_loss -0.0200" in decision.reason


def test_promotion_when_improving_accuracy_within_tolerances():
    champ = {"accuracy": 0.60, "log_loss": 0.650, "brier_score": 0.230}
    cand = {"accuracy": 0.65, "log_loss": 0.655, "brier_score": 0.232}  # slight loss increase within tolerance

    decision = evaluate_candidate_promotion(champ, cand)
    assert decision.promoted is True
    assert "accuracy +5.00%" in decision.reason


def test_rejection_when_log_loss_regresses_beyond_tolerance():
    champ = {"accuracy": 0.65, "log_loss": 0.620, "brier_score": 0.210}
    # Log loss regresses by +0.030 (exceeds default tolerance of 0.015) even if accuracy went up
    cand = {"accuracy": 0.70, "log_loss": 0.650, "brier_score": 0.210}

    decision = evaluate_candidate_promotion(champ, cand)
    assert decision.promoted is False
    assert "Log loss regressed" in decision.reason


def test_rejection_when_brier_score_regresses_beyond_tolerance():
    champ = {"accuracy": 0.65, "log_loss": 0.620, "brier_score": 0.210}
    # Brier regresses by +0.020 (exceeds default tolerance of 0.010)
    cand = {"accuracy": 0.68, "log_loss": 0.615, "brier_score": 0.230}

    decision = evaluate_candidate_promotion(champ, cand)
    assert decision.promoted is False
    assert "Brier score regressed" in decision.reason


def test_rejection_when_accuracy_drops_beyond_tolerance():
    champ = {"accuracy": 0.65, "log_loss": 0.620, "brier_score": 0.210}
    # Accuracy drops by 5% (exceeds default tolerance of 2%) even if log loss improved slightly
    cand = {"accuracy": 0.58, "log_loss": 0.610, "brier_score": 0.208}

    decision = evaluate_candidate_promotion(champ, cand)
    assert decision.promoted is False
    assert "Accuracy dropped" in decision.reason


def test_rejection_when_no_metric_improves():
    champ = {"accuracy": 0.65, "log_loss": 0.620, "brier_score": 0.210}
    cand = {"accuracy": 0.65, "log_loss": 0.620, "brier_score": 0.210}

    decision = evaluate_candidate_promotion(champ, cand)
    assert decision.promoted is False
    assert "failed to improve" in decision.reason


# -- Full walk-forward integration tests (SQLite) ----------------------------


class FakeProvider(NFLDataProvider):
    def __init__(self, games: pd.DataFrame):
        self._games = games

    def get_games(self, season: int, week: int | None = None) -> pd.DataFrame:
        df = self._games[self._games["season"] == season]
        if week is not None:
            df = df[df["week"] == week]
        return df.reset_index(drop=True)

    def get_injuries(self, season: int, week: int | None = None) -> pd.DataFrame:
        return pd.DataFrame(columns=["season", "week", "team", "player_id", "full_name",
                                      "position", "report_status", "report_primary_injury",
                                      "practice_status", "date_modified"])

    def get_players(self, season: int | None = None) -> pd.DataFrame:
        return pd.DataFrame(columns=["player_id", "full_name", "position", "team", "status"])

    def get_odds(self, season: int, week: int | None = None) -> pd.DataFrame:
        raise NotImplementedError

    def get_plays(self, season: int) -> pd.DataFrame:
        raise NotImplementedError


def _make_game(season: int, week: int, home: str, away: str, h_score: int, a_score: int, date: str) -> dict:
    return {
        "game_id": f"{season}_{week:02d}_{away}_{home}",
        "season": season,
        "week": week,
        "game_type": "REG",
        "gameday": date,
        "gametime": "13:00",
        "home_team": home,
        "away_team": away,
        "home_score": h_score,
        "away_score": a_score,
        "location": "Home",
        "spread_line": -3.0,
        "total_line": 45.0,
        "home_moneyline": -150.0,
        "away_moneyline": 130.0,
        "home_rest": 7.0,
        "away_rest": 7.0,
        "overtime": 0.0,
        "result": None,
    }


def _build_full_pipeline(games: pd.DataFrame) -> None:
    provider = FakeProvider(games)
    for season in sorted(games["season"].unique()):
        run_ingestion.run(provider, season=int(season), week=None, skip_injuries=True, skip_players=True)
        silver_transform.run(season=int(season), week=None)
    gold_transform.run()
    game_results_transform.run()


@pytest.fixture()
def isolated_db(monkeypatch, tmp_path):
    db_file = tmp_path / "test_backtest.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    db_module.init_schema()
    return db_file


def test_walk_forward_backtest_end_to_end(isolated_db):
    """Seed multi-season data (2024 and 2025 weeks 1-5) and run walk-forward simulation."""
    games = []
    # Seed 2024 prior season (16 games with mixed outcomes)
    for w in range(1, 5):
        games.append(_make_game(2024, w, "KC", "LV", 28, 14, f"2024-09-{w:02d}"))
        games.append(_make_game(2024, w, "BUF", "MIA", 17, 24, f"2024-09-{w:02d}"))
        games.append(_make_game(2024, w, "SF", "LAR", 20, 17, f"2024-09-{w:02d}"))
        games.append(_make_game(2024, w, "BAL", "CIN", 21, 28, f"2024-09-{w:02d}"))

    # Seed 2025 target season (Weeks 1 to 5 with mixed outcomes)
    for w in range(1, 6):
        games.append(_make_game(2025, w, "KC", "LV", 27, 20, f"2025-09-{w:02d}"))
        games.append(_make_game(2025, w, "BUF", "MIA", 14, 24, f"2025-09-{w:02d}"))
        games.append(_make_game(2025, w, "SF", "LAR", 30, 13, f"2025-09-{w:02d}"))
        games.append(_make_game(2025, w, "BAL", "CIN", 20, 24, f"2025-09-{w:02d}"))

    df_games = pd.DataFrame(games)
    _build_full_pipeline(df_games)

    engine = db_module.get_engine()

    # Run backtest with retrain_start_week=4
    summary = ml_backtest.run_walk_forward_backtest(
        season=2025,
        start_week=1,
        end_week=5,
        retrain_start_week=4,
        write_db=True,
        engine=engine,
    )

    assert summary["season"] == 2025
    assert summary["start_week"] == 1
    assert summary["end_week"] == 5
    assert len(summary["weekly_results"]) == 5
    assert summary["cumulative_metrics"]["n_games"] == 20
    assert 0.0 <= summary["cumulative_metrics"]["accuracy"] <= 1.0

    # Verify Decision #4: weeks 1-3 lineage skipped retraining
    early_lineage = [entry for entry in summary["lineage_log"] if entry["week"] < 4]
    for entry in early_lineage:
        assert entry["action"] == "retained"
        assert "Decision #4" in entry["reason"]

    # Verify predictions were written to ml.predictions
    table = db_module.qualified_table("ml", "predictions")
    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        sample_row = conn.execute(
            text(f"SELECT predicted_home_score, predicted_away_score, predicted_margin, cover_probability FROM {table} LIMIT 1")
        ).fetchone()
        assert count == 20
        assert sample_row[0] is not None
        assert sample_row[1] is not None
        assert sample_row[2] is not None
        assert sample_row[3] is not None
        assert sample_row[0] - sample_row[1] == pytest.approx(sample_row[2], abs=0.2)


def test_walk_forward_backtest_with_save_models(isolated_db, monkeypatch, tmp_path):
    """Verify that save_models=True writes model artifacts to MODELS_DIR."""
    import src.ml.train as ml_train
    fake_models_dir = tmp_path / "models"
    monkeypatch.setattr(ml_backtest, "MODELS_DIR", fake_models_dir)
    monkeypatch.setattr(ml_backtest, "BACKTESTS_DIR", fake_models_dir / "backtests")
    monkeypatch.setattr(ml_train, "MODELS_DIR", fake_models_dir)

    games = []
    for w in range(1, 4):
        games.append(_make_game(2024, w, "KC", "LV", 28, 14, f"2024-09-{w:02d}"))
        games.append(_make_game(2024, w, "BUF", "MIA", 17, 24, f"2024-09-{w:02d}"))
        games.append(_make_game(2025, w, "KC", "LV", 27, 20, f"2025-09-{w:02d}"))
        games.append(_make_game(2025, w, "BUF", "MIA", 14, 24, f"2025-09-{w:02d}"))

    df_games = pd.DataFrame(games)
    _build_full_pipeline(df_games)

    summary = ml_backtest.run_walk_forward_backtest(
        season=2025,
        start_week=1,
        end_week=2,
        retrain_start_week=2,
        write_db=False,
        save_models=True,
    )

    # Base model artifact directory should exist
    saved_dirs = [p for p in fake_models_dir.iterdir() if p.is_dir() and p.name != "backtests"]
    assert len(saved_dirs) >= 1
    assert (saved_dirs[0] / "model.joblib").exists()
    assert (saved_dirs[0] / "metadata.json").exists()
