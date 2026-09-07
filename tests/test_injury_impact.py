"""Unit tests for src/features/injury_impact.py -- no database needed."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.injury_impact import _latest_report_per_player, compute_team_injury_impact, load_weights


@pytest.fixture()
def weights():
    return load_weights()


@pytest.fixture()
def players():
    return pd.DataFrame({
        "player_id": ["qb1", "wr1", "bench1"],
        "position": ["QB", "WR", "P"],
    })


def test_load_weights_has_expected_shape(weights):
    assert weights["version"] == "v0"
    assert weights["position_weights"]["QB"] == -10.0
    assert "Out" in weights["status_multipliers"]


def test_impact_scaled_by_status_multiplier(weights, players):
    injuries = pd.DataFrame({
        "player_id": ["qb1"],
        "team_id": ["KC"],
        "status": ["Out"],
        "reported_at": ["2025-09-04"],
    })
    impact = compute_team_injury_impact(injuries, players, weights=weights)
    assert impact["KC"] == pytest.approx(-10.0 * 1.0)


def test_questionable_status_reduces_impact_vs_out(weights, players):
    out_injury = pd.DataFrame({"player_id": ["qb1"], "team_id": ["KC"], "status": ["Out"],
                                "reported_at": ["2025-09-04"]})
    questionable_injury = pd.DataFrame({"player_id": ["qb1"], "team_id": ["KC"], "status": ["Questionable"],
                                         "reported_at": ["2025-09-04"]})

    out_impact = compute_team_injury_impact(out_injury, players, weights=weights)["KC"]
    questionable_impact = compute_team_injury_impact(questionable_injury, players, weights=weights)["KC"]

    assert abs(questionable_impact) < abs(out_impact)


def test_multiple_players_sum_per_team(weights, players):
    injuries = pd.DataFrame({
        "player_id": ["qb1", "wr1"],
        "team_id": ["KC", "KC"],
        "status": ["Out", "Out"],
        "reported_at": ["2025-09-04", "2025-09-04"],
    })
    impact = compute_team_injury_impact(injuries, players, weights=weights)
    assert impact["KC"] == pytest.approx(-10.0 - 3.0)


def test_only_latest_report_per_player_counts(weights, players):
    injuries = pd.DataFrame({
        "player_id": ["qb1", "qb1"],
        "team_id": ["KC", "KC"],
        "status": ["Questionable", "Out"],  # downgraded from Questionable to Out
        "reported_at": ["2025-09-03", "2025-09-05"],
    })
    impact = compute_team_injury_impact(injuries, players, weights=weights)
    # only the later "Out" report should count, not both summed
    assert impact["KC"] == pytest.approx(-10.0 * 1.0)


def test_unknown_position_uses_default_weight(weights):
    injuries = pd.DataFrame({"player_id": ["mystery"], "team_id": ["KC"], "status": ["Out"],
                              "reported_at": ["2025-09-04"]})
    players_missing = pd.DataFrame({"player_id": [], "position": []})
    impact = compute_team_injury_impact(injuries, players_missing, weights=weights)
    assert impact["KC"] == pytest.approx(weights["default_position_weight"] * 1.0)


def test_empty_injuries_returns_empty_dict(weights, players):
    impact = compute_team_injury_impact(pd.DataFrame(columns=["player_id", "team_id", "status", "reported_at"]),
                                         players, weights=weights)
    assert impact == {}


def test_latest_report_does_not_mix_fields_across_reports_with_missing_timestamp():
    # Regression test: groupby(...).last() picks the last *non-null value
    # per column independently*, not the last row as a whole. With one
    # timestamped report and one un-timestamped (NaT) report for the same
    # player, the old implementation could return status="Out" (the row
    # that's actually NaT) paired with reported_at="2025-09-03" (borrowed
    # from the OTHER, Doubtful row) -- a combination that never existed in
    # the source data. Real for 2025 nflverse data (see DECISIONS.md #5),
    # which frequently has no reported_at at all.
    injuries = pd.DataFrame({
        "player_id": ["qb1", "qb1"],
        "team_id": ["KC", "KC"],
        "status": ["Doubtful", "Out"],
        "reported_at": ["2025-09-03", None],
    })
    latest = _latest_report_per_player(injuries)
    row = latest[latest["player_id"] == "qb1"].iloc[0]

    # the chosen row must be exactly one of the two real reports -- never
    # a status/timestamp pairing that never actually existed
    assert (row["status"], pd.isna(row["reported_at"])) in [("Doubtful", False), ("Out", True)]
