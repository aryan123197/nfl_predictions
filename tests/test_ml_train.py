"""
Tests for src/ml/train.py's split logic (pure, no DB) and the full
training pipeline against an in-memory SQLite database seeded through
the real ingestion/transform path -- same pattern as
tests/test_gold_transform.py.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import text

import src.db as db_module
from src.ingest import run_ingestion
from src.ml import train as ml_train
from src.ml.features import FEATURE_COLUMNS, load_training_frame, split_features_target
from src.providers.base import NFLDataProvider
from src.transform import game_results_transform, gold_transform, silver_transform

# -- split_train_holdout: pure logic, no DB ---------------------------------


def _row(season, week, game_id="g"):
    return {"season": season, "week": week, "game_id": f"{game_id}{season}_{week}"}


def test_season_based_split_when_multiple_seasons_present():
    df = pd.DataFrame([_row(2023, w) for w in range(1, 5)] + [_row(2024, w) for w in range(1, 5)])
    train, holdout, description = ml_train.split_train_holdout(df, holdout_season=2024)

    assert set(train["season"]) == {2023}
    assert set(holdout["season"]) == {2024}
    assert description == "season<2024"


def test_default_holdout_is_most_recent_season():
    df = pd.DataFrame([_row(2022, 1), _row(2023, 1), _row(2024, 1)])
    train, holdout, _ = ml_train.split_train_holdout(df, holdout_season=None)

    assert set(holdout["season"]) == {2024}
    assert set(train["season"]) == {2022, 2023}


def test_falls_back_to_week_split_with_only_one_season():
    df = pd.DataFrame([_row(2025, w) for w in range(1, 9)])  # weeks 1-8, one season only
    train, holdout, description = ml_train.split_train_holdout(df, holdout_season=None)

    assert not train.empty
    assert not holdout.empty
    assert train["week"].max() < holdout["week"].min()  # still strictly chronological
    assert "weeks<" in description


# -- full pipeline integration -----------------------------------------------


class FakeProvider(NFLDataProvider):
    def __init__(self, games: pd.DataFrame):
        self._games = games

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
        return pd.DataFrame(columns=["season", "week", "team", "player_id", "full_name",
                                      "position", "report_status", "report_primary_injury",
                                      "practice_status", "date_modified"])

    def get_odds(self, season, week=None):
        raise NotImplementedError


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    yield db_path


def _game(game_id, season, week, home, away, home_score, away_score, gameday):
    return {
        "game_id": game_id, "season": season, "week": week, "game_type": "REG",
        "gameday": gameday, "gametime": "13:00", "home_team": home, "away_team": away,
        "home_score": home_score, "away_score": away_score, "location": "Home",
        "spread_line": -3.0, "total_line": 45.0, "home_moneyline": -150.0, "away_moneyline": 130.0,
        "home_rest": 7.0, "away_rest": 7.0, "overtime": 0.0, "result": None,
    }


@pytest.fixture()
def two_season_games():
    rows = []
    for season, base_day in [(2023, "2023-09"), (2024, "2024-09")]:
        for week in range(1, 6):
            rows.append(_game(f"{season}_{week:02d}_A_B", season, week, "A", "B",
                               24.0 + week, 17.0, f"{base_day}-{week:02d}"))
            rows.append(_game(f"{season}_{week:02d}_C_D", season, week, "C", "D",
                               20.0, 23.0 + week, f"{base_day}-{week:02d}"))
    return pd.DataFrame(rows)


def _build_full_pipeline(games: pd.DataFrame):
    provider = FakeProvider(games)
    for season in sorted(games["season"].unique()):
        run_ingestion.run(provider, season=int(season), week=None, skip_injuries=True, skip_players=True)
        silver_transform.run(season=int(season), week=None)
    gold_transform.run()
    game_results_transform.run()


def test_load_training_frame_excludes_ties(isolated_db, two_season_games):
    games_with_tie = pd.concat([
        two_season_games,
        pd.DataFrame([_game("2024_06_E_F", 2024, 6, "E", "F", 20.0, 20.0, "2024-09-06")]),
    ], ignore_index=True)
    _build_full_pipeline(games_with_tie)

    engine = db_module.get_engine()
    df = load_training_frame(engine)

    assert "2024_06_E_F" not in set(df["game_id"])  # tie excluded -- no well-defined win label


def test_train_and_write_predictions_end_to_end(isolated_db, two_season_games):
    _build_full_pipeline(two_season_games)
    engine = db_module.get_engine()

    result = ml_train.run(holdout_season=2024)

    assert result["predictions_written"] == 10  # 2 games/week x 5 weeks in the 2024 holdout
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert 0.0 <= result["metrics"]["brier_score"] <= 1.0
    assert result["metrics"]["mae_margin"] >= 0.0

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM ml_predictions")).scalar()
        row = conn.execute(
            text("SELECT home_win_probability, away_win_probability, predicted_home_score, "
                 "predicted_away_score, predicted_margin, cover_probability FROM ml_predictions LIMIT 1")
        ).fetchone()

    assert count == 10
    assert row[0] == pytest.approx(1.0 - row[1])  # probabilities are complementary
    assert row[2] is not None and row[3] is not None  # predicted scores present
    assert row[4] is not None  # predicted margin present
    assert row[5] is not None  # cover probability present
    assert row[2] - row[3] == pytest.approx(row[4], abs=0.2)


def test_predictions_are_never_overwritten_across_retrains(isolated_db, two_season_games):
    _build_full_pipeline(two_season_games)
    engine = db_module.get_engine()

    ml_train.run(holdout_season=2024)
    ml_train.run(holdout_season=2024)  # a second "retrain" -- must add rows, not replace them

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM ml_predictions")).scalar()
        versions = conn.execute(text("SELECT COUNT(DISTINCT model_version) FROM ml_predictions")).scalar()

    assert count == 20  # both runs' predictions coexist
    assert versions == 2  # under two distinct model_version values


def test_feature_target_split_produces_binary_labels(isolated_db, two_season_games):
    _build_full_pipeline(two_season_games)
    engine = db_module.get_engine()
    df = load_training_frame(engine)

    X, y = split_features_target(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(y.unique()).issubset({0, 1})
