"""
Integration tests for the silver -> gold transform, using an in-memory
SQLite database (offline, no network) -- same pattern as
tests/test_silver_transform.py. These exercise the point-in-time
correctness guarantees (design doc section 19) end to end through the
real bronze -> silver -> gold pipeline, not just the unit-level
guarantees already covered by test_elo.py / test_injury_impact.py.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import text

import src.db as db_module
from src.ingest import run_ingestion
from src.providers.base import NFLDataProvider
from src.transform import gold_transform, silver_transform


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


def _base_game(game_id, week, home, away, home_score=None, away_score=None,
                gameday="2025-09-07", home_rest=7.0, away_rest=7.0, spread=-3.0):
    return {
        "game_id": game_id, "season": 2025, "week": week, "game_type": "REG",
        "gameday": gameday, "gametime": "13:00", "home_team": home, "away_team": away,
        "home_score": home_score, "away_score": away_score, "location": "Home",
        "spread_line": spread, "total_line": 45.0, "home_moneyline": -130.0, "away_moneyline": 110.0,
        "home_rest": home_rest, "away_rest": away_rest, "overtime": 0.0, "result": None,
    }


@pytest.fixture()
def two_week_games():
    return pd.DataFrame([
        _base_game("2025_01_KC_LAC", 1, "KC", "LAC", home_score=27.0, away_score=21.0, gameday="2025-09-07"),
        _base_game("2025_02_KC_DEN", 2, "DEN", "KC", gameday="2025-09-14"),  # scheduled, not yet played
    ])


@pytest.fixture()
def kc_qb_roster():
    return pd.DataFrame({"player_id": ["qb1"], "full_name": ["Test QB"], "position": ["QB"],
                          "team": ["KC"], "status": ["ACT"]})


@pytest.fixture()
def tie_then_game():
    return pd.DataFrame([
        _base_game("2025_01_KC_LAC", 1, "KC", "LAC", home_score=20.0, away_score=20.0, gameday="2025-09-07"),
        _base_game("2025_02_KC_DEN", 2, "DEN", "KC", gameday="2025-09-14"),
    ])


def _run_full_pipeline(games, injuries=None, players=None):
    provider = FakeProvider(games=games, injuries=injuries, players=players)
    run_ingestion.run(provider, season=2025, week=None,
                       skip_injuries=injuries is None, skip_players=players is None)
    silver_transform.run(season=2025, week=None)
    return gold_transform.run()


def test_gold_transform_is_idempotent(isolated_db, two_week_games):
    _run_full_pipeline(two_week_games)
    gold_transform.run()  # second run should replace, not duplicate

    engine = db_module.get_engine()
    with engine.connect() as conn:
        features_count = conn.execute(text("SELECT COUNT(*) FROM gold_game_features")).scalar()
        ratings_count = conn.execute(text("SELECT COUNT(*) FROM gold_team_ratings")).scalar()

    assert features_count == 2  # one row per game
    assert ratings_count == 4  # two teams per game x two games


def test_elo_reflects_prior_week_result_only(isolated_db, two_week_games):
    _run_full_pipeline(two_week_games)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        week1_home_elo = conn.execute(
            text("SELECT home_elo FROM gold_game_features WHERE game_id = '2025_01_KC_LAC'")
        ).scalar()
        week2_away_elo = conn.execute(  # KC is away in week 2
            text("SELECT away_elo FROM gold_game_features WHERE game_id = '2025_02_KC_DEN'")
        ).scalar()

    assert week1_home_elo == 1500.0  # KC's first-ever game: no history yet
    assert week2_away_elo > 1500.0   # KC won week 1 as the home team, rating should carry forward


def test_injury_reported_after_game_date_is_excluded(isolated_db, two_week_games, kc_qb_roster):
    late_injury = pd.DataFrame({
        "season": [2025], "week": [1], "team": ["KC"], "player_id": ["qb1"],
        "full_name": ["Test QB"], "position": ["QB"], "report_status": ["Out"],
        "report_primary_injury": ["Knee"], "practice_status": ["DNP"],
        "date_modified": ["2025-09-08"],  # AFTER the 2025-09-07 game_date -- must not count
    })
    _run_full_pipeline(two_week_games, injuries=late_injury, players=kc_qb_roster)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        home_impact = conn.execute(  # KC is home in week 1
            text("SELECT home_injury_impact FROM gold_game_features WHERE game_id = '2025_01_KC_LAC'")
        ).scalar()

    assert home_impact == 0.0  # the late-reported injury must not leak into this game's features


def test_injury_reported_before_game_date_is_included(isolated_db, two_week_games, kc_qb_roster):
    early_injury = pd.DataFrame({
        "season": [2025], "week": [1], "team": ["KC"], "player_id": ["qb1"],
        "full_name": ["Test QB"], "position": ["QB"], "report_status": ["Out"],
        "report_primary_injury": ["Knee"], "practice_status": ["DNP"],
        "date_modified": ["2025-09-05"],  # BEFORE the 2025-09-07 game_date
    })
    _run_full_pipeline(two_week_games, injuries=early_injury, players=kc_qb_roster)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        home_impact = conn.execute(
            text("SELECT home_injury_impact FROM gold_game_features WHERE game_id = '2025_01_KC_LAC'")
        ).scalar()

    assert home_impact == pytest.approx(-10.0)  # QB Out counts in full


def test_recent_form_uses_only_games_strictly_before(isolated_db, two_week_games):
    _run_full_pipeline(two_week_games)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        week1_home_form = conn.execute(  # KC's first game -- no prior history
            text("SELECT home_recent_form FROM gold_game_features WHERE game_id = '2025_01_KC_LAC'")
        ).scalar()
        week2_away_form = conn.execute(  # KC's second game (now away) -- should reflect the week 1 win
            text("SELECT away_recent_form FROM gold_game_features WHERE game_id = '2025_02_KC_DEN'")
        ).scalar()

    assert week1_home_form is None
    assert week2_away_form == 1.0  # 1 win / 1 completed prior game


def test_game_odds_mirrors_silver_odds(isolated_db, two_week_games):
    _run_full_pipeline(two_week_games)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT opening_spread, current_spread, spread_movement, total "
                 "FROM gold_game_odds WHERE game_id = '2025_01_KC_LAC'")
        ).fetchone()

    assert row[0] == row[1] == -3.0  # single snapshot: opening == current
    assert row[2] == 0.0
    assert row[3] == 45.0


def test_rest_days_carried_from_silver_games(isolated_db, two_week_games):
    _run_full_pipeline(two_week_games)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT home_rest_days, away_rest_days FROM gold_game_features "
                 "WHERE game_id = '2025_01_KC_LAC'")
        ).fetchone()

    assert row[0] == 7
    assert row[1] == 7


def test_tie_counts_as_half_win_in_recent_form(isolated_db, tie_then_game):
    # Regression test: a tie used to be recorded as win=False for BOTH
    # teams (home_margin > 0 is False when margin is 0), so recent_form
    # scored a tied game as an outright loss. Elo already treats a tie as
    # 0.5 (see elo.py's _actual_score) -- recent_form should match.
    _run_full_pipeline(tie_then_game)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        tie_win_flag = conn.execute(
            text("SELECT win FROM gold_team_game_stats WHERE team_id = 'KC' AND game_id = '2025_01_KC_LAC'")
        ).scalar()
        week2_away_form = conn.execute(
            text("SELECT away_recent_form FROM gold_game_features WHERE game_id = '2025_02_KC_DEN'")
        ).scalar()

    assert tie_win_flag is None  # neither a win nor a loss
    assert week2_away_form == 0.5  # the tie counts as half a win, not a full loss


def test_scheduled_games_do_not_crash_the_pipeline_on_missing_scores(isolated_db, two_week_games):
    # Regression test: pd.read_sql upcasts a NULL INTEGER column to float
    # NaN, and inserting that NaN straight into an INTEGER column fails on
    # Postgres (SQLite silently coerces it to NULL, which is why this only
    # showed up against a real Postgres-shaped check). two_week_games
    # already includes one scheduled (unplayed) game -- this just asserts
    # the pipeline completes and leaves clean NULLs, not NaNs, behind.
    _run_full_pipeline(two_week_games)

    engine = db_module.get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT points_for, points_against FROM gold_team_game_stats "
                 "WHERE game_id = '2025_02_KC_DEN' AND team_id = 'KC'")
        ).fetchone()

    assert row[0] is None
    assert row[1] is None


def test_schema_migration_backfills_columns_on_a_pre_existing_database(isolated_db):
    # Regression test: home_rest_days/away_rest_days/season/week were
    # added to silver.games/silver.injuries by editing the CREATE TABLE IF
    # NOT EXISTS statements in place. On a database that already ran the
    # OLD schema, that's a silent no-op -- the columns never get added.
    # Simulate that by creating the tables with the old (pre-migration)
    # shape directly, then confirm init_schema() backfills the columns.
    engine = db_module.get_engine()
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE silver_games (game_id TEXT PRIMARY KEY, season INTEGER NOT NULL, week INTEGER, "
            "game_date TEXT, home_team_id TEXT NOT NULL, away_team_id TEXT NOT NULL, home_score INTEGER, "
            "away_score INTEGER, venue_id TEXT, status TEXT NOT NULL)"
        ))
        conn.execute(text(
            "CREATE TABLE silver_injuries (injury_id INTEGER PRIMARY KEY, bronze_id INTEGER NOT NULL, "
            "player_id TEXT, team_id TEXT, status TEXT, injury_type TEXT, reported_at TEXT, "
            "expected_return TEXT, source TEXT NOT NULL)"
        ))

    db_module.init_schema()  # must not raise, and must add the missing columns

    with engine.connect() as conn:
        games_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(silver_games)"))}
        injuries_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(silver_injuries)"))}

    assert {"home_rest_days", "away_rest_days"} <= games_cols
    assert {"season", "week"} <= injuries_cols
