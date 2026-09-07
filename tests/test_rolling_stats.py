"""Unit tests for src/features/rolling_stats.py -- no database needed."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.rolling_stats import compute_per_game_team_stats, compute_rolling_stats


def _play(game_id, posteam, defteam, epa, pass_attempt=False, rush_attempt=False, yards_gained=0.0,
          success=False, sack=False, qb_hit=False, interception=False, fumble_lost=False,
          touchdown=False, down=None, yardline_100=50, third_down_converted=False):
    return {
        "game_id": game_id, "posteam": posteam, "defteam": defteam, "epa": epa,
        "pass_attempt": pass_attempt, "rush_attempt": rush_attempt, "yards_gained": yards_gained,
        "success": success, "sack": sack, "qb_hit": qb_hit, "interception": interception,
        "fumble_lost": fumble_lost, "touchdown": touchdown, "down": down,
        "yardline_100": yardline_100, "third_down_converted": third_down_converted,
    }


def test_non_scrimmage_plays_are_excluded():
    # a kickoff (neither pass nor rush) must not count toward offensive EPA
    plays = pd.DataFrame([
        _play("g1", "A", "B", epa=-5.0, pass_attempt=False, rush_attempt=False),  # kickoff-like
        _play("g1", "A", "B", epa=1.0, pass_attempt=True, yards_gained=5),
    ])
    per_game = compute_per_game_team_stats(plays)
    row = per_game[per_game["team_id"] == "A"].iloc[0]
    assert row["plays"] == 1
    assert row["epa_sum"] == 1.0


def test_epa_per_play_is_a_true_average_not_average_of_per_play_values():
    plays = pd.DataFrame([
        _play("g1", "A", "B", epa=2.0, pass_attempt=True, yards_gained=3),
        _play("g1", "A", "B", epa=-1.0, rush_attempt=True, yards_gained=1),
        _play("g1", "A", "B", epa=0.5, pass_attempt=True, yards_gained=2),
    ])
    per_game = compute_per_game_team_stats(plays)
    row = per_game[per_game["team_id"] == "A"].iloc[0]
    assert row["plays"] == 3
    assert row["epa_sum"] == pytest.approx(1.5)


def test_defensive_epa_uses_opponent_offensive_plays():
    plays = pd.DataFrame([
        _play("g1", "A", "B", epa=3.0, pass_attempt=True, yards_gained=10),
    ])
    per_game = compute_per_game_team_stats(plays)
    b_row = per_game[per_game["team_id"] == "B"].iloc[0]
    assert b_row["def_plays"] == 1
    assert b_row["def_epa_sum"] == 3.0


def test_explosive_play_thresholds_differ_by_play_type():
    plays = pd.DataFrame([
        _play("g1", "A", "B", epa=0, pass_attempt=True, yards_gained=15),   # explosive pass (>=15)
        _play("g1", "A", "B", epa=0, pass_attempt=True, yards_gained=14),   # not explosive
        _play("g1", "A", "B", epa=0, rush_attempt=True, yards_gained=10),   # explosive rush (>=10)
        _play("g1", "A", "B", epa=0, rush_attempt=True, yards_gained=9),    # not explosive
    ])
    per_game = compute_per_game_team_stats(plays)
    row = per_game[per_game["team_id"] == "A"].iloc[0]
    assert row["explosive_sum"] == 2


def test_sack_counted_as_dropback():
    # Verified against real 2025 data: nflverse sets pass_attempt=1 on
    # every sacked play, so a sack is already a scrimmage play via
    # pass_attempt -- this just confirms it's counted as a dropback.
    plays = pd.DataFrame([
        _play("g1", "A", "B", epa=-1.5, pass_attempt=True, rush_attempt=False, sack=True, qb_hit=True),
    ])
    per_game = compute_per_game_team_stats(plays)
    row = per_game[per_game["team_id"] == "A"].iloc[0]
    assert row["dropbacks"] == 1
    assert row["sack_sum"] == 1
    assert row["qb_hit_sum"] == 1


def test_third_down_conversion_rate_inputs():
    plays = pd.DataFrame([
        _play("g1", "A", "B", epa=1, pass_attempt=True, down=3, third_down_converted=True),
        _play("g1", "A", "B", epa=-1, rush_attempt=True, down=3, third_down_converted=False),
        _play("g1", "A", "B", epa=1, rush_attempt=True, down=1, third_down_converted=False),
    ])
    per_game = compute_per_game_team_stats(plays)
    row = per_game[per_game["team_id"] == "A"].iloc[0]
    assert row["third_down_plays"] == 2
    assert row["third_down_conversions"] == 1


# -- compute_rolling_stats: point-in-time correctness -----------------------

def _team_game(team_id, game_id, season, week):
    return {"team_id": team_id, "game_id": game_id, "season": season, "week": week}


def test_first_game_of_season_has_no_history():
    team_games = pd.DataFrame([_team_game("A", "g1", 2025, 1)])
    per_game = pd.DataFrame(columns=["team_id", "game_id"])
    rolling = compute_rolling_stats(per_game, team_games)

    std_row = rolling[(rolling["team_id"] == "A") & (rolling["window"] == "season_to_date")].iloc[0]
    assert std_row["games_included"] == 0
    # pandas coerces a column mixing None with floats to float64/NaN at
    # DataFrame construction -- the DB-insert boundary (gold_transform.py's
    # _replace_table) is what turns this back into a real SQL NULL.
    assert pd.isna(std_row["off_epa"])


def test_rolling_stats_only_use_strictly_prior_games():
    plays = pd.DataFrame([
        _play("g1", "A", "B", epa=2.0, pass_attempt=True, yards_gained=5),  # week 1: A's good game
        _play("g2", "A", "C", epa=-2.0, pass_attempt=True, yards_gained=1),  # week 2: A's bad game
    ])
    per_game = compute_per_game_team_stats(plays)
    team_games = pd.DataFrame([
        _team_game("A", "g1", 2025, 1),
        _team_game("A", "g2", 2025, 2),
        _team_game("A", "g3", 2025, 3),  # week 3: no plays yet (scheduled/future)
    ])
    rolling = compute_rolling_stats(per_game, team_games)
    std = rolling[rolling["window"] == "season_to_date"].set_index("game_id")

    assert std.loc["g1", "games_included"] == 0  # nothing before week 1
    assert pd.isna(std.loc["g1", "off_epa"])
    assert std.loc["g2", "games_included"] == 1  # only week 1's game counts
    assert std.loc["g2", "off_epa"] == pytest.approx(2.0)  # week 1's EPA only, NOT week 2's own game
    assert std.loc["g3", "games_included"] == 2  # weeks 1 and 2
    assert std.loc["g3", "off_epa"] == pytest.approx(0.0)  # (2.0 + -2.0) / 2 games... but weighted by plays


def test_season_to_date_sums_across_games_weighted_by_play_count():
    plays = pd.DataFrame([
        # week 1: 3 plays, epa sum = 3.0
        _play("g1", "A", "B", epa=1.0, pass_attempt=True),
        _play("g1", "A", "B", epa=1.0, pass_attempt=True),
        _play("g1", "A", "B", epa=1.0, pass_attempt=True),
        # week 2: 1 play, epa sum = -3.0
        _play("g2", "A", "C", epa=-3.0, pass_attempt=True),
    ])
    per_game = compute_per_game_team_stats(plays)
    team_games = pd.DataFrame([
        _team_game("A", "g1", 2025, 1),
        _team_game("A", "g2", 2025, 2),
        _team_game("A", "g3", 2025, 3),
    ])
    rolling = compute_rolling_stats(per_game, team_games)
    g3_row = rolling[(rolling["window"] == "season_to_date") & (rolling["game_id"] == "g3")].iloc[0]
    # true weighted average: (3.0 + -3.0) / (3 + 1) plays = 0.0, NOT the
    # naive average of the two per-game rates (1.0 + -3.0) / 2 = -1.0
    assert g3_row["off_epa"] == pytest.approx(0.0)


def test_last_4_window_drops_games_older_than_four():
    plays = pd.DataFrame([_play(f"g{i}", "A", "B", epa=10.0 if i == 1 else 0.0, pass_attempt=True) for i in range(1, 6)])
    per_game = compute_per_game_team_stats(plays)
    team_games = pd.DataFrame([_team_game("A", f"g{i}", 2025, i) for i in range(1, 7)])
    rolling = compute_rolling_stats(per_game, team_games)

    g6_std = rolling[(rolling["window"] == "season_to_date") & (rolling["game_id"] == "g6")].iloc[0]
    g6_l4 = rolling[(rolling["window"] == "last_4") & (rolling["game_id"] == "g6")].iloc[0]
    assert g6_std["games_included"] == 5  # all 5 prior games
    assert g6_l4["games_included"] == 4  # only the most recent 4
    assert g6_l4["off_epa"] == pytest.approx(0.0)  # game 1's +10.0 EPA has rolled out of the window
    assert g6_std["off_epa"] > 0  # but it's still in season-to-date


def test_history_resets_at_season_boundary():
    plays = pd.DataFrame([_play("g1", "A", "B", epa=10.0, pass_attempt=True)])
    per_game = compute_per_game_team_stats(plays)
    team_games = pd.DataFrame([
        _team_game("A", "g1", 2025, 1),
        _team_game("A", "g2", 2026, 1),  # new season
    ])
    rolling = compute_rolling_stats(per_game, team_games)
    g2_row = rolling[(rolling["window"] == "season_to_date") & (rolling["game_id"] == "g2")].iloc[0]
    assert g2_row["games_included"] == 0  # 2025's game must not carry into 2026
