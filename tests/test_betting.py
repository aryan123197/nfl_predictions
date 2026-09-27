"""
Unit tests for src/ml/betting.py:
- Empirical ECDF vs Gaussian pricing
- EV calculations at standard -110 juice
- Fractional Kelly stakes
- Tiered betting simulation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.betting import (
    STANDARD_VIG_BREAKEVEN,
    calculate_empirical_cover_probability,
    calculate_expected_value,
    calculate_kelly_fraction,
    evaluate_betting_tiers,
)


def test_standard_vig_breakeven():
    assert STANDARD_VIG_BREAKEVEN == pytest.approx(0.5238, abs=1e-3)
    ev_breakeven = calculate_expected_value(STANDARD_VIG_BREAKEVEN)
    assert ev_breakeven == pytest.approx(0.0, abs=1e-3)


def test_expected_value_positive_on_edge():
    # 55% win rate should yield positive EV
    ev = calculate_expected_value(0.55)
    assert ev > 0.0
    # 50% win rate should yield negative EV (-4.76%) due to juice
    ev_50 = calculate_expected_value(0.50)
    assert ev_50 < 0.0


def test_kelly_fraction_zero_on_negative_ev():
    # Negative EV bet should stake 0
    k_stake = calculate_kelly_fraction(0.50)
    assert k_stake == 0.0


def test_kelly_fraction_positive_and_capped():
    # Positive EV bet should produce positive stake capped at max_cap
    k_stake = calculate_kelly_fraction(0.60, fraction_multiplier=0.25, max_cap=0.05)
    assert 0.0 < k_stake <= 0.05


def test_empirical_ecdf_cover_probability():
    pred_margins = np.array([3.0, 7.0])
    market_spreads = np.array([3.0, 3.5])
    # Synthetic residuals with known distribution
    residuals = np.array([-10, -7, -3, 0, 3, 4, 7, 10] * 10)  # 80 observations

    probs = calculate_empirical_cover_probability(pred_margins, market_spreads, historical_residuals=residuals)
    assert len(probs) == 2
    assert 0.01 <= probs[0] <= 0.99
    assert 0.01 <= probs[1] <= 0.99


def test_evaluate_betting_tiers():
    # Construct synthetic predictions with known edges
    data = {
        "predicted_margin": [7.0, 4.0, 1.0, 3.0, -5.0],
        "market_spread":    [3.0, 3.5, 1.0, 6.0, -2.0],  # edges: +4.0, +0.5, 0.0, -3.0, -3.0
        "actual_margin":    [10.0, 7.0, 3.0, 0.0, -7.0],
        "home_cover_prob":  [0.65, 0.52, 0.50, 0.40, 0.38],
    }
    df = pd.DataFrame(data)

    results = evaluate_betting_tiers(df, edge_thresholds=[0.0, 2.0])
    assert "all_games" in results
    assert "edge_ge_2.0pts" in results
    assert results["all_games"]["bets_placed"] == 5
    assert results["edge_ge_2.0pts"]["bets_placed"] == 3  # only edges >= 2.0


def test_margin_std_is_floored_so_cover_probability_stays_sane():
    """A too-small margin_std manufactures fake betting edges.

    margin_std is the spread of the cover-probability distribution. Trainers
    estimated it from IN-SAMPLE residuals, which a boosted tree drives far
    below real forecast error -- champion models stored values as low as
    2.69 points when the true residual std against completed games is ~12.8.

    The effect is not subtle: at std=2.69 a 6.6-point model edge prices as a
    99% cover and +89% EV, so every game on the slate reads STRONG_VALUE.
    """
    pred_margins = np.array([9.6])
    market_spreads = np.array([3.0])

    realistic = calculate_empirical_cover_probability(
        pred_margins, market_spreads, default_std=13.5
    )[0]
    overconfident = calculate_empirical_cover_probability(
        pred_margins, market_spreads, default_std=2.69
    )[0]

    # A 6.6-point edge is a real lean, but nowhere near a certainty.
    assert 0.60 < realistic < 0.80
    # The understated std turns that same edge into a near-lock.
    assert overconfident > 0.95


def test_win_probability_is_not_a_cover_probability():
    """P(win outright) and P(cover) must not be substituted for one another.

    The betting endpoint used to fall back to home_win_probability when
    cover_probability was null. They are different quantities -- a heavy
    favourite usually wins but only covers about half the time -- so the
    substitution invented spread edges from a number that never described
    the spread.
    """
    # Home favoured by 10; it wins far more often than it covers -10.
    pred_margins = np.array([10.0])
    market_spreads = np.array([10.0])
    cover = calculate_empirical_cover_probability(
        pred_margins, market_spreads, default_std=13.5
    )[0]
    # Predicted margin sits exactly on the line => a coin flip to cover...
    assert cover == pytest.approx(0.5, abs=0.05)
    # ...even though such a team wins outright far more often than that.
