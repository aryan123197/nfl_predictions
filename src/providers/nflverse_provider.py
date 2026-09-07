"""
NFLverseProvider — free, historical + near-real-time NFL data source.

Pulls directly from the nflverse-data GitHub releases (no API key, no
rate limit beyond GitHub's, updated automatically by the nflverse
project during the season). This is the provider we use for V1 so the
whole platform costs $0 in data fees.

Source releases used:
    schedules/games.csv        -> games, spread_line/total_line (odds)
    injuries/injuries_{season}.csv -> weekly injury reports
    pbp/play_by_play_{season}.csv.gz -> play-by-play (large; pulled lazily)
    players/players.csv        -> player roster/id table

Swap this out later for a paid live provider (e.g. SportsDataIO) by
writing a new class that implements the same NFLDataProvider interface
-- nothing downstream has to change.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import requests

from src.providers.base import NFLDataProvider, ProviderError

logger = logging.getLogger(__name__)

BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"
REQUEST_TIMEOUT_SECONDS = 60


class NFLverseProvider(NFLDataProvider):
    def __init__(self, session: Optional[requests.Session] = None):
        self._session = session or requests.Session()

    # -- internal helper --------------------------------------------------
    def _fetch_csv(self, release: str, filename: str) -> pd.DataFrame:
        url = f"{BASE_URL}/{release}/{filename}"
        try:
            resp = self._session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"Failed to fetch {url}: {exc}") from exc

        try:
            return pd.read_csv(pd.io.common.BytesIO(resp.content), low_memory=False)
        except Exception as exc:  # pandas raises many different error types
            raise ProviderError(f"Failed to parse CSV from {url}: {exc}") from exc

    # -- interface implementation -----------------------------------------
    def get_games(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        df = self._fetch_csv("schedules", "games.csv")
        df = df[df["season"] == season].copy()
        if week is not None:
            df = df[df["week"] == week]

        out = pd.DataFrame({
            "game_id": df["game_id"],
            "season": df["season"],
            "week": df["week"],
            "game_type": df["game_type"],
            "gameday": df["gameday"],
            "gametime": df["gametime"],
            "home_team": df["home_team"],
            "away_team": df["away_team"],
            "home_score": df["home_score"],
            "away_score": df["away_score"],
            "location": df["location"],
            "spread_line": df["spread_line"],
            "total_line": df["total_line"],
            "home_moneyline": df.get("home_moneyline"),
            "away_moneyline": df.get("away_moneyline"),
            "home_rest": df.get("home_rest"),
            "away_rest": df.get("away_rest"),
            "overtime": df.get("overtime"),
            "result": df.get("result"),
        })
        return out.reset_index(drop=True)

    def get_plays(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        # Play-by-play files are ~40-80MB per season compressed; fetch
        # lazily and only when actually needed (Phase 3+), not on every
        # ingestion run.
        df = self._fetch_csv("pbp", f"play_by_play_{season}.csv.gz")
        if week is not None:
            df = df[df["week"] == week]

        keep_cols = [
            "play_id", "game_id", "qtr", "game_seconds_remaining", "down",
            "ydstogo", "yardline_100", "posteam", "defteam", "play_type",
            "yards_gained", "epa", "success", "pass_attempt", "rush_attempt",
            "interception", "fumble_lost", "touchdown",
            "passer_player_id", "rusher_player_id", "receiver_player_id",
        ]
        available = [c for c in keep_cols if c in df.columns]
        return df[available].reset_index(drop=True)

    def get_players(self, season: int) -> pd.DataFrame:
        df = self._fetch_csv("players", "players.csv")
        out = pd.DataFrame({
            "player_id": df.get("gsis_id"),
            "full_name": df.get("display_name", df.get("full_name")),
            "position": df.get("position"),
            "team": df.get("latest_team", df.get("team")),
            "status": df.get("status"),
        })
        return out.dropna(subset=["player_id"]).reset_index(drop=True)

    def get_injuries(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        df = self._fetch_csv("injuries", f"injuries_{season}.csv")
        if week is not None:
            df = df[df["week"] == week]

        # nflverse has changed this file's columns between seasons
        # (e.g. `date_modified` was dropped for 2025). Use .get() with
        # a default so a source schema change degrades gracefully
        # instead of breaking ingestion -- this is exactly the kind of
        # drift the provider abstraction exists to absorb.
        out = pd.DataFrame({
            "season": df["season"],
            "week": df["week"],
            "team": df["team"],
            "player_id": df["gsis_id"],
            "full_name": df["full_name"],
            "position": df["position"],
            "report_status": df["report_status"],
            "report_primary_injury": df["report_primary_injury"],
            "practice_status": df["practice_status"],
            "date_modified": df["date_modified"] if "date_modified" in df.columns else None,
        })
        return out.reset_index(drop=True)

    def get_odds(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        # nflverse only has one consensus closing line per game (no
        # per-sportsbook, no line movement). Good enough for V1;
        # `sportsbook` and `timestamp` are placeholders a paid live
        # provider would populate properly.
        games = self.get_games(season, week)
        out = pd.DataFrame({
            "game_id": games["game_id"],
            "sportsbook": "nflverse_consensus",
            "timestamp": games["gameday"],
            "spread": games["spread_line"],
            "home_moneyline": games["home_moneyline"],
            "away_moneyline": games["away_moneyline"],
            "total": games["total_line"],
        })
        return out.reset_index(drop=True)
