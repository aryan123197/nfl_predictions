"""
Tests for Phase 10: Closing Line Value (CLV) & Market Movement Tracker.
"""

from src.ml.clv import calculate_clv


def test_clv_movement_and_edge():
    # Model favored KC by 6.0 at opening (-3.0). Line closed at -4.5.
    # Market moved in model's direction -> +1.5 points of CLV!
    clv = calculate_clv(
        game_id="2026_01_BAL_KC",
        home_team_id="KC",
        away_team_id="BAL",
        opening_spread=-3.0,
        closing_spread=-4.5,
        predicted_margin=6.0,
    )
    assert clv.market_line_movement == -1.5
    assert clv.clv_edge_points == 1.5
    assert clv.beat_closing_line is True


def test_clv_against_model():
    # Model favored KC (-3.0 opening), but line moved to -1.5 (market steamed towards BAL)
    clv = calculate_clv(
        game_id="2026_01_BAL_KC",
        home_team_id="KC",
        away_team_id="BAL",
        opening_spread=-3.0,
        closing_spread=-1.5,
        predicted_margin=6.0,
    )
    assert clv.market_line_movement == 1.5
    assert clv.clv_edge_points == -1.5
    assert clv.beat_closing_line is False


def test_clv_missing_spreads():
    clv = calculate_clv(
        game_id="2026_01_BAL_KC",
        home_team_id="KC",
        away_team_id="BAL",
        opening_spread=None,
        closing_spread=-3.5,
        predicted_margin=4.0,
    )
    assert clv.clv_edge_points is None
    assert clv.beat_closing_line is None
