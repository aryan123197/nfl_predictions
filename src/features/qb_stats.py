"""
Quarterback and player-level feature computation (design doc section 15).

Aggregates per-game QB performance from silver.plays, identifies starting QBs
per game (point-in-time), and computes rolling efficiency metrics across
'season_to_date' and 'last_4' windows.

Point-in-time correctness (design doc section 19):
Every QB rolling stat for game G is calculated exclusively from games strictly
prior to G.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np

WINDOWS: dict[str, Optional[int]] = {"season_to_date": None, "last_4": 4}

# Baseline prior for cold-start QBs (Week 1 or first career starts)
DEFAULT_QB_EPA = 0.0
DEFAULT_SUCCESS_RATE = 0.45
DEFAULT_SACK_RATE = 0.065
DEFAULT_TURNOVER_RATE = 0.03


def compute_per_game_qb_stats(plays: pd.DataFrame) -> pd.DataFrame:
    """Aggregates dropbacks, passes, EPA, sacks, hits, and turnovers per QB per game."""
    if plays.empty:
        return pd.DataFrame(columns=[
            "game_id", "season", "week", "team_id", "player_id",
            "dropbacks", "pass_attempts", "epa_sum", "pass_epa_sum",
            "success_sum", "turnover_sum", "sack_sum", "qb_hit_sum"
        ])

    plays = plays.copy()
    # Filter to plays with a valid passer
    valid_passers = plays[plays["passer_player_id"].notna() & (plays["passer_player_id"] != "")].copy()
    if valid_passers.empty:
        return pd.DataFrame(columns=[
            "game_id", "season", "week", "team_id", "player_id",
            "dropbacks", "pass_attempts", "epa_sum", "pass_epa_sum",
            "success_sum", "turnover_sum", "sack_sum", "qb_hit_sum"
        ])

    for col in ("pass_attempt", "sack", "success", "interception", "fumble_lost", "qb_hit"):
        if col in valid_passers.columns:
            valid_passers[col] = valid_passers[col].fillna(False).astype(bool)
        else:
            valid_passers[col] = False

    valid_passers["is_dropback"] = valid_passers["pass_attempt"] | valid_passers["sack"]
    valid_passers["turnover"] = valid_passers["interception"] | valid_passers["fumble_lost"]

    rows = []
    for (game_id, posteam, passer_id), group in valid_passers.groupby(
        ["game_id", "posteam", "passer_player_id"]
    ):
        season = group["season"].iloc[0] if "season" in group.columns else None
        week = group["week"].iloc[0] if "week" in group.columns else None
        dropbacks = int(group["is_dropback"].sum())
        pass_attempts = int(group["pass_attempt"].sum())
        pass_plays = group[group["pass_attempt"]]

        rows.append({
            "game_id": game_id,
            "season": season,
            "week": week,
            "team_id": posteam,
            "player_id": passer_id,
            "dropbacks": dropbacks,
            "pass_attempts": pass_attempts,
            "epa_sum": float(group["epa"].fillna(0.0).sum()),
            "pass_epa_sum": float(pass_plays["epa"].fillna(0.0).sum()) if not pass_plays.empty else 0.0,
            "success_sum": int(group["success"].sum()),
            "turnover_sum": int(group["turnover"].sum()),
            "sack_sum": int(group["sack"].sum()),
            "qb_hit_sum": int(group["qb_hit"].sum()),
        })

    return pd.DataFrame(rows)


def identify_game_starters(
    per_game_qb: pd.DataFrame,
    team_games: pd.DataFrame
) -> pd.DataFrame:
    """Identifies the starting QB for every team-game.
    
    For completed games with play data, the starter is the QB with the most dropbacks.
    For scheduled or unplayed games, defaults to the team's most recent starter.
    """
    if team_games.empty:
        return pd.DataFrame(columns=["team_id", "game_id", "season", "week", "player_id", "is_starter_change"])

    # Find highest dropback QB per team-game
    actual_starters = {}
    if not per_game_qb.empty:
        sorted_qb = per_game_qb.sort_values(
            ["game_id", "team_id", "dropbacks", "pass_attempts"],
            ascending=[True, True, False, False]
        )
        top_qb = sorted_qb.drop_duplicates(subset=["game_id", "team_id"], keep="first")
        for _, row in top_qb.iterrows():
            actual_starters[(row["team_id"], row["game_id"])] = row["player_id"]

    ordered = team_games.sort_values(["team_id", "season", "week", "game_id"], kind="stable")
    starter_rows = []

    for team_id, tg_group in ordered.groupby("team_id", sort=False):
        last_starter = None
        current_season = None

        for _, tg in tg_group.iterrows():
            season = tg["season"]
            game_id = tg["game_id"]
            week = tg["week"]

            if season != current_season:
                last_starter = None
                current_season = season

            key = (team_id, game_id)
            if key in actual_starters:
                starter_id = actual_starters[key]
            else:
                starter_id = last_starter

            # Check starter continuity
            starter_change = 1.0 if (last_starter is not None and starter_id != last_starter) else 0.0

            starter_rows.append({
                "team_id": team_id,
                "game_id": game_id,
                "season": season,
                "week": week,
                "player_id": starter_id,
                "is_starter_change": starter_change,
            })

            if starter_id is not None:
                last_starter = starter_id

    return pd.DataFrame(starter_rows)


def compute_rolling_qb_stats(
    per_game_qb: pd.DataFrame,
    starters: pd.DataFrame
) -> pd.DataFrame:
    """Computes point-in-time rolling stats for every starter across windows."""
    if starters.empty:
        return pd.DataFrame()

    per_game_by_player: dict[str, list[dict]] = {}
    if not per_game_qb.empty:
        ordered_qb = per_game_qb.sort_values(["player_id", "season", "week", "game_id"], kind="stable")
        for player_id, group in ordered_qb.groupby("player_id"):
            per_game_by_player[player_id] = group.to_dict("records")

    rows = []
    for _, starter in starters.iterrows():
        player_id = starter["player_id"]
        team_id = starter["team_id"]
        game_id = starter["game_id"]
        season = starter["season"]
        week = starter["week"]

        # If player has no history or is None, emit empty rolling rows
        if not player_id or player_id not in per_game_by_player:
            for window_name in WINDOWS:
                rows.append({
                    "player_id": player_id,
                    "team_id": team_id,
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "window": window_name,
                    "games_included": 0,
                    "dropbacks": 0,
                    "pass_attempts": 0,
                    "qb_epa_per_dropback": None,
                    "qb_pass_epa": None,
                    "qb_success_rate": None,
                    "qb_sack_rate": None,
                    "qb_turnover_rate": None,
                    "is_starter": True,
                })
            continue

        history = per_game_by_player[player_id]
        # Filter point-in-time: strictly prior to this game in the same season
        prior_games = [
            g for g in history
            if g["season"] == season and (
                g["week"] < week if (week is not None and g["week"] is not None)
                else g["game_id"] != game_id
            )
        ]

        for window_name, window_size in WINDOWS.items():
            window_history = prior_games if window_size is None else prior_games[-window_size:]
            n = len(window_history)

            if n == 0:
                rows.append({
                    "player_id": player_id,
                    "team_id": team_id,
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "window": window_name,
                    "games_included": 0,
                    "dropbacks": 0,
                    "pass_attempts": 0,
                    "qb_epa_per_dropback": None,
                    "qb_pass_epa": None,
                    "qb_success_rate": None,
                    "qb_sack_rate": None,
                    "qb_turnover_rate": None,
                    "is_starter": True,
                })
            else:
                total_dropbacks = sum(g["dropbacks"] for g in window_history)
                total_passes = sum(g["pass_attempts"] for g in window_history)
                total_epa = sum(g["epa_sum"] for g in window_history)
                total_pass_epa = sum(g["pass_epa_sum"] for g in window_history)
                total_successes = sum(g["success_sum"] for g in window_history)
                total_sacks = sum(g["sack_sum"] for g in window_history)
                total_turnovers = sum(g["turnover_sum"] for g in window_history)

                rows.append({
                    "player_id": player_id,
                    "team_id": team_id,
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "window": window_name,
                    "games_included": n,
                    "dropbacks": total_dropbacks,
                    "pass_attempts": total_passes,
                    "qb_epa_per_dropback": (total_epa / total_dropbacks) if total_dropbacks > 0 else None,
                    "qb_pass_epa": (total_pass_epa / total_passes) if total_passes > 0 else None,
                    "qb_success_rate": (total_successes / total_dropbacks) if total_dropbacks > 0 else None,
                    "qb_sack_rate": (total_sacks / total_dropbacks) if total_dropbacks > 0 else None,
                    "qb_turnover_rate": (total_turnovers / total_dropbacks) if total_dropbacks > 0 else None,
                    "is_starter": True,
                })

    return pd.DataFrame(rows)


def extract_qb_game_features(
    rolling_qb_stats: pd.DataFrame,
    starters: pd.DataFrame,
    games: pd.DataFrame
) -> pd.DataFrame:
    """Extracts home and away QB features for gold.game_features."""
    if games.empty:
        return pd.DataFrame()

    # Sourced from 'season_to_date' window
    s2d = rolling_qb_stats[rolling_qb_stats["window"] == "season_to_date"].copy() if not rolling_qb_stats.empty else pd.DataFrame()

    starters_dict = {}
    if not starters.empty:
        for _, r in starters.iterrows():
            starters_dict[(r["game_id"], r["team_id"])] = {
                "player_id": r["player_id"],
                "is_starter_change": r["is_starter_change"],
            }

    stats_dict = {}
    if not s2d.empty:
        for _, r in s2d.iterrows():
            stats_dict[(r["game_id"], r["team_id"])] = r.to_dict()

    feature_rows = []
    for _, g in games.iterrows():
        game_id = g["game_id"]
        home_team = g["home_team_id"]
        away_team = g["away_team_id"]

        home_starter_info = starters_dict.get((game_id, home_team), {})
        away_starter_info = starters_dict.get((game_id, away_team), {})

        home_qb_id = home_starter_info.get("player_id")
        away_qb_id = away_starter_info.get("player_id")

        home_change = home_starter_info.get("is_starter_change", 0.0)
        away_change = away_starter_info.get("is_starter_change", 0.0)

        home_stats = stats_dict.get((game_id, home_team), {})
        away_stats = stats_dict.get((game_id, away_team), {})

        home_epa = home_stats.get("qb_epa_per_dropback")
        away_epa = away_stats.get("qb_epa_per_dropback")

        home_succ = home_stats.get("qb_success_rate")
        away_succ = away_stats.get("qb_success_rate")

        epa_diff = None
        if home_epa is not None and away_epa is not None:
            epa_diff = home_epa - away_epa

        feature_rows.append({
            "game_id": game_id,
            "home_qb_id": home_qb_id,
            "away_qb_id": away_qb_id,
            "home_qb_epa": home_epa,
            "away_qb_epa": away_epa,
            "qb_epa_diff": epa_diff,
            "home_qb_success_rate": home_succ,
            "away_qb_success_rate": away_succ,
            "home_qb_starter_change": home_change,
            "away_qb_starter_change": away_change,
        })

    return pd.DataFrame(feature_rows)
