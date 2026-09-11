"""
Tests for Phase 10: Kelly Criterion & +EV Betting Strategy Engine.
"""

import pytest
from src.ml.betting_strategy import (
    ValueTier,
    american_to_decimal,
    calculate_expected_value,
    calculate_kelly_fraction,
    decimal_to_implied_prob,
    evaluate_spread_bet,
)


def test_odds_conversions():
    # Standard -110 American odds
    dec_neg = american_to_decimal(-110)
    assert round(dec_neg, 4) == 1.9091
    assert round(decimal_to_implied_prob(dec_neg), 4) == 0.5238

    # Underdog +150 American odds
    dec_pos = american_to_decimal(150)
    assert dec_pos == 2.5
    assert decimal_to_implied_prob(dec_pos) == 0.40


def test_expected_value_calculations():
    # 55% win rate at -110 odds has positive EV
    ev_positive = calculate_expected_value(0.55, decimal_odds=1.9091)
    assert ev_positive > 0.0
    assert round(ev_positive, 2) == 5.0  # ~+5% EV

    # 50% win rate at -110 odds has negative EV (vig loss)
    ev_negative = calculate_expected_value(0.50, decimal_odds=1.9091)
    assert ev_negative < 0.0
    assert ev_negative == pytest.approx(-4.545, abs=0.01)



def test_kelly_sizing_fractions():
    # 55% win prob at -110 odds: f* = (0.9091 * 0.55 - 0.45) / 0.9091 = 0.055
    sizing = calculate_kelly_fraction(0.55, decimal_odds=1.9091)
    assert sizing.full_kelly_pct > 0
    assert sizing.half_kelly_pct == round(sizing.full_kelly_pct * 0.5, 3)
    assert sizing.quarter_kelly_pct == round(sizing.full_kelly_pct * 0.25, 3)
    assert 0.5 <= sizing.recommended_units <= 4.0

    # Underdog or negative EV returns zero bet sizing
    zero_sizing = calculate_kelly_fraction(0.50, decimal_odds=1.9091)
    assert zero_sizing.full_kelly_pct == 0.0
    assert zero_sizing.recommended_units == 0.0


def test_evaluate_spread_bet():
    # Test Home Favorite with strong edge
    rec = evaluate_spread_bet(
        game_id="2026_01_BAL_KC",
        home_team_id="KC",
        away_team_id="BAL",
        market_spread=-3.5,
        home_cover_prob=0.58,
    )
    assert rec is not None
    assert rec.target_team_id == "KC"
    assert rec.value_tier in (ValueTier.STRONG_VALUE, ValueTier.MODERATE_VALUE)
    assert rec.expected_value_pct > 0
    assert rec.sizing.recommended_units > 0

    # Test Away Underdog with strong edge
    rec_away = evaluate_spread_bet(
        game_id="2026_01_SF_DET",
        home_team_id="DET",
        away_team_id="SF",
        market_spread=-6.0,
        home_cover_prob=0.40,  # Away cover prob is 60%
    )
    assert rec_away is not None
    assert rec_away.target_team_id == "SF"
    assert rec_away.market_line == 6.0
    assert rec_away.value_tier == ValueTier.STRONG_VALUE
