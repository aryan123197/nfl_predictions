"""
Tests for Quarterback and player-level feature computation (Phase 3 Slice C).
"""

import pandas as pd
import numpy as np
import pytest

from src.features.qb_stats import (
    compute_per_game_qb_stats,
    identify_game_starters,
    compute_rolling_qb_stats,
    extract_qb_game_features,
)


def test_compute_per_game_qb_stats_empty():
    empty_df = pd.DataFrame()
    res = compute_per_game_qb_stats(empty_df)
    assert res.empty
    assert "player_id" in res.columns
    assert "epa_sum" in res.columns


def test_compute_per_game_qb_stats_aggregation():
    plays = pd.DataFrame([
        {
            "game_id": "2025_01_KC_BAL",
            "season": 2025,
            "week": 1,
            "posteam": "KC",
            "passer_player_id": "00-0033873",  # Mahomes
            "pass_attempt": True,
            "sack": False,
            "success": True,
            "epa": 0.5,
            "interception": False,
            "fumble_lost": False,
            "qb_hit": False,
        },
        {
            "game_id": "2025_01_KC_BAL",
            "season": 2025,
            "week": 1,
            "posteam": "KC",
            "passer_player_id": "00-0033873",
            "pass_attempt": False,
            "sack": True,
            "success": False,
            "epa": -1.2,
            "interception": False,
            "fumble_lost": False,
            "qb_hit": True,
        },
        {
            "game_id": "2025_01_KC_BAL",
            "season": 2025,
            "week": 1,
            "posteam": "BAL",
            "passer_player_id": "00-0034796",  # Lamar
            "pass_attempt": True,
            "sack": False,
            "success": True,
            "epa": 1.1,
            "interception": False,
            "fumble_lost": False,
            "qb_hit": False,
        },
    ])

    res = compute_per_game_qb_stats(plays)
    assert len(res) == 2

    mahomes = res[res["player_id"] == "00-0033873"].iloc[0]
    assert mahomes["dropbacks"] == 2
    assert mahomes["pass_attempts"] == 1
    assert mahomes["sack_sum"] == 1
    assert mahomes["qb_hit_sum"] == 1
    assert mahomes["success_sum"] == 1
    assert pytest.approx(mahomes["epa_sum"], 0.001) == -0.7

    lamar = res[res["player_id"] == "00-0034796"].iloc[0]
    assert lamar["dropbacks"] == 1
    assert lamar["pass_attempts"] == 1
    assert pytest.approx(lamar["epa_sum"], 0.001) == 1.1


def test_identify_game_starters_and_changes():
    per_game_qb = pd.DataFrame([
        {"game_id": "G1", "season": 2025, "week": 1, "team_id": "KC", "player_id": "QB1", "dropbacks": 35, "pass_attempts": 30},
        {"game_id": "G1", "season": 2025, "week": 1, "team_id": "KC", "player_id": "QB_BACKUP", "dropbacks": 2, "pass_attempts": 2},
        {"game_id": "G2", "season": 2025, "week": 2, "team_id": "KC", "player_id": "QB1", "dropbacks": 30, "pass_attempts": 28},
        {"game_id": "G3", "season": 2025, "week": 3, "team_id": "KC", "player_id": "QB_BACKUP", "dropbacks": 25, "pass_attempts": 20},
    ])

    team_games = pd.DataFrame([
        {"team_id": "KC", "game_id": "G1", "season": 2025, "week": 1},
        {"team_id": "KC", "game_id": "G2", "season": 2025, "week": 2},
        {"team_id": "KC", "game_id": "G3", "season": 2025, "week": 3},
        {"team_id": "KC", "game_id": "G4", "season": 2025, "week": 4},  # Scheduled future game
    ])

    starters = identify_game_starters(per_game_qb, team_games)
    assert len(starters) == 4

    # G1: QB1 starts, first game of season -> starter_change = 0
    g1 = starters[starters["game_id"] == "G1"].iloc[0]
    assert g1["player_id"] == "QB1"
    assert g1["is_starter_change"] == 0.0

    # G2: QB1 starts again -> starter_change = 0
    g2 = starters[starters["game_id"] == "G2"].iloc[0]
    assert g2["player_id"] == "QB1"
    assert g2["is_starter_change"] == 0.0

    # G3: QB_BACKUP starts -> starter_change = 1
    g3 = starters[starters["game_id"] == "G3"].iloc[0]
    assert g3["player_id"] == "QB_BACKUP"
    assert g3["is_starter_change"] == 1.0

    # G4: Future game carried forward -> QB_BACKUP, starter_change = 0
    g4 = starters[starters["game_id"] == "G4"].iloc[0]
    assert g4["player_id"] == "QB_BACKUP"
    assert g4["is_starter_change"] == 0.0


def test_compute_rolling_qb_stats_point_in_time():
    per_game_qb = pd.DataFrame([
        {
            "game_id": "G1", "season": 2025, "week": 1, "team_id": "KC", "player_id": "QB1",
            "dropbacks": 20, "pass_attempts": 18, "epa_sum": 10.0, "pass_epa_sum": 10.0,
            "success_sum": 12, "turnover_sum": 0, "sack_sum": 1, "qb_hit_sum": 2
        },
        {
            "game_id": "G2", "season": 2025, "week": 2, "team_id": "KC", "player_id": "QB1",
            "dropbacks": 30, "pass_attempts": 28, "epa_sum": -5.0, "pass_epa_sum": -3.0,
            "success_sum": 13, "turnover_sum": 2, "sack_sum": 2, "qb_hit_sum": 3
        },
    ])

    starters = pd.DataFrame([
        {"team_id": "KC", "game_id": "G1", "season": 2025, "week": 1, "player_id": "QB1", "is_starter_change": 0.0},
        {"team_id": "KC", "game_id": "G2", "season": 2025, "week": 2, "player_id": "QB1", "is_starter_change": 0.0},
        {"team_id": "KC", "game_id": "G3", "season": 2025, "week": 3, "player_id": "QB1", "is_starter_change": 0.0},
    ])

    rolling = compute_rolling_qb_stats(per_game_qb, starters)
    s2d = rolling[rolling["window"] == "season_to_date"]

    # Week 1: 0 prior games -> stats are None / NaN
    g1_s2d = s2d[s2d["game_id"] == "G1"].iloc[0]
    assert g1_s2d["games_included"] == 0
    assert pd.isna(g1_s2d["qb_epa_per_dropback"])

    # Week 2: includes Week 1 (10.0 / 20 = 0.5 EPA/dropback)
    g2_s2d = s2d[s2d["game_id"] == "G2"].iloc[0]
    assert g2_s2d["games_included"] == 1
    assert pytest.approx(g2_s2d["qb_epa_per_dropback"], 0.001) == 0.5
    assert pytest.approx(g2_s2d["qb_success_rate"], 0.001) == 0.6  # 12 / 20

    # Week 3: includes Week 1 and Week 2 ( (10 - 5) / (20 + 30) = 5 / 50 = 0.1 EPA/dropback )
    g3_s2d = s2d[s2d["game_id"] == "G3"].iloc[0]
    assert g3_s2d["games_included"] == 2
    assert pytest.approx(g3_s2d["qb_epa_per_dropback"], 0.001) == 0.1
    assert pytest.approx(g3_s2d["qb_success_rate"], 0.001) == 0.5  # (12 + 13) / 50


def test_extract_qb_game_features():
    rolling_qb = pd.DataFrame([
        {
            "game_id": "G3", "team_id": "KC", "player_id": "QB_KC", "window": "season_to_date",
            "qb_epa_per_dropback": 0.25, "qb_success_rate": 0.55
        },
        {
            "game_id": "G3", "team_id": "LV", "player_id": "QB_LV", "window": "season_to_date",
            "qb_epa_per_dropback": -0.10, "qb_success_rate": 0.40
        }
    ])

    starters = pd.DataFrame([
        {"game_id": "G3", "team_id": "KC", "player_id": "QB_KC", "is_starter_change": 0.0},
        {"game_id": "G3", "team_id": "LV", "player_id": "QB_LV", "is_starter_change": 1.0},
    ])

    games = pd.DataFrame([
        {"game_id": "G3", "home_team_id": "KC", "away_team_id": "LV"}
    ])

    feat = extract_qb_game_features(rolling_qb, starters, games)
    assert len(feat) == 1
    row = feat.iloc[0]
    assert row["home_qb_id"] == "QB_KC"
    assert row["away_qb_id"] == "QB_LV"
    assert pytest.approx(row["home_qb_epa"], 0.001) == 0.25
    assert pytest.approx(row["away_qb_epa"], 0.001) == -0.10
    assert pytest.approx(row["qb_epa_diff"], 0.001) == 0.35
    assert row["home_qb_starter_change"] == 0.0
    assert row["away_qb_starter_change"] == 1.0
