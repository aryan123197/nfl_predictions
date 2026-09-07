"""
Abstract base class for NFL data providers.

Every concrete provider (nflverse, a paid live-odds API, a future
play-by-play vendor, etc.) implements this interface. The rest of the
pipeline (bronze ingestion, silver transforms, features, models) only
ever talks to this interface -- never to a specific vendor's SDK or
response schema. That's what lets us swap NFLverseProvider for a paid
live provider later without touching downstream code.

Design notes:
- Every method returns a pandas.DataFrame with a stable, documented
  column contract (see the docstring of each method). Providers are
  responsible for mapping their vendor-specific fields onto that
  contract.
- Every method accepts a `season` (and sometimes `week`) filter so
  callers can do incremental, point-in-time-safe pulls instead of
  always downloading everything.
- Providers should raise `ProviderError` (not bare exceptions) so the
  ingestion layer can catch failures uniformly and log them into
  metadata.pipeline_runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class ProviderError(RuntimeError):
    """Raised when a provider fails to fetch or parse source data."""


class NFLDataProvider(ABC):
    """Interface every NFL data provider must implement."""

    @abstractmethod
    def get_games(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Return game/schedule rows for a season (optionally one week).

        Expected columns (bronze contract):
            game_id, season, week, game_type, gameday, gametime,
            home_team, away_team, home_score, away_score,
            location, spread_line, total_line, home_moneyline,
            away_moneyline, home_rest, away_rest, overtime, result
        """
        raise NotImplementedError

    @abstractmethod
    def get_plays(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Return play-by-play rows for a season (optionally one week).

        Expected columns (bronze contract):
            play_id, game_id, qtr, game_seconds_remaining, down,
            ydstogo, yardline_100, posteam, defteam, play_type,
            yards_gained, epa, success, pass_attempt, rush_attempt,
            interception, fumble_lost, touchdown, passer_player_id,
            rusher_player_id, receiver_player_id
        """
        raise NotImplementedError

    @abstractmethod
    def get_players(self, season: int) -> pd.DataFrame:
        """Return the player/roster table for a season.

        Expected columns (bronze contract):
            player_id, full_name, position, team, status
        """
        raise NotImplementedError

    @abstractmethod
    def get_injuries(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Return injury report rows for a season (optionally one week).

        Expected columns (bronze contract):
            season, week, team, player_id, full_name, position,
            report_status, report_primary_injury, practice_status,
            date_modified
        """
        raise NotImplementedError

    @abstractmethod
    def get_odds(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Return betting line rows for a season (optionally one week).

        nflverse only carries opening/closing consensus lines embedded
        in the schedule; a paid live-odds provider would return
        multiple sportsbooks with timestamps. Expected columns:
            game_id, sportsbook, timestamp, spread, home_moneyline,
            away_moneyline, total
        """
        raise NotImplementedError

    # -- optional, not every provider will have this --
    def get_weather(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Return weather rows. Default: not supported by this provider."""
        raise NotImplementedError(f"{self.__class__.__name__} does not supply weather data")

    @property
    def name(self) -> str:
        return self.__class__.__name__
