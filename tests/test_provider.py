"""
Tests for the data provider layer.

Split into:
- Unit tests (no network) that check the abstract interface is honored
  and that column contracts are stable.
- Integration tests (real network call to nflverse) marked so they can
  be skipped in CI environments without internet access via
  `pytest -m "not integration"`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.providers.base import NFLDataProvider
from src.providers.nflverse_provider import NFLverseProvider

EXPECTED_GAME_COLUMNS = {
    "game_id", "season", "week", "game_type", "gameday", "gametime",
    "home_team", "away_team", "home_score", "away_score", "location",
    "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "home_rest", "away_rest", "overtime", "result",
}

EXPECTED_INJURY_COLUMNS = {
    "season", "week", "team", "player_id", "full_name", "position",
    "report_status", "report_primary_injury", "practice_status", "date_modified",
}


def test_nflverse_provider_implements_interface():
    provider = NFLverseProvider()
    assert isinstance(provider, NFLDataProvider)
    assert provider.name == "NFLverseProvider"


def test_abstract_provider_cannot_be_instantiated():
    with pytest.raises(TypeError):
        NFLDataProvider()  # type: ignore[abstract]


@pytest.mark.integration
def test_get_games_returns_expected_schema():
    provider = NFLverseProvider()
    df = provider.get_games(season=2024, week=1)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert EXPECTED_GAME_COLUMNS.issubset(set(df.columns))
    # every game in week 1 should have a completed score (2024 is historical)
    assert df["home_score"].notna().all()
    assert (df["season"] == 2024).all()
    assert (df["week"] == 1).all()


@pytest.mark.integration
def test_get_games_filters_by_week():
    provider = NFLverseProvider()
    week1 = provider.get_games(season=2024, week=1)
    full_season = provider.get_games(season=2024)

    assert len(week1) < len(full_season)
    assert set(week1["game_id"]).issubset(set(full_season["game_id"]))


@pytest.mark.integration
def test_get_injuries_returns_expected_schema_even_with_source_drift():
    """Regression test: nflverse dropped `date_modified` for 2025.
    The provider should degrade gracefully rather than raising KeyError.
    """
    provider = NFLverseProvider()
    df = provider.get_injuries(season=2025, week=1)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert EXPECTED_INJURY_COLUMNS.issubset(set(df.columns))


@pytest.mark.integration
def test_get_players_returns_nonempty_with_ids():
    provider = NFLverseProvider()
    df = provider.get_players(season=2024)

    assert not df.empty
    assert df["player_id"].notna().all()
    assert "position" in df.columns


@pytest.mark.integration
def test_get_odds_derived_from_games():
    provider = NFLverseProvider()
    df = provider.get_odds(season=2024, week=1)

    assert not df.empty
    assert {"game_id", "spread", "total"}.issubset(set(df.columns))


@pytest.mark.integration
def test_get_plays_returns_real_scrimmage_data():
    """Regression test: play_by_play_{season}.csv.gz is gzip-compressed,
    and _fetch_csv's pd.read_csv call had no `compression` argument --
    pandas can't infer compression from a BytesIO payload (no filename
    suffix to infer from), so this silently tried to parse raw gzip
    bytes as text and raised UnicodeDecodeError. Found by running this
    against real data during Phase 3 Slice B, not by a unit test."""
    provider = NFLverseProvider()
    df = provider.get_plays(season=2024, week=1)

    assert not df.empty
    assert {"game_id", "epa", "pass_attempt", "rush_attempt", "posteam", "defteam"}.issubset(set(df.columns))
    # a real scrimmage play must have a non-null EPA
    scrimmage = df[(df["pass_attempt"] == 1) | (df["rush_attempt"] == 1)]
    assert not scrimmage.empty
    assert scrimmage["epa"].notna().any()
