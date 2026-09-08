"""
Betting strategy, empirical key-number calibration, Expected Value (EV),
and Kelly Criterion position sizing -- informed by empirical sports modeling
and Levine (2019) 'Beating Vegas'.

Key capabilities:
1. Empirical Residual Distribution / ECDF: Accurately prices non-linear NFL
   margin clusters around key numbers (3, 7, 6, 10, 14, 4) instead of naive smooth Gaussian.
2. Expected Value (EV) calculation against standard sportsbook vig (-110 juice = 52.38% breakeven).
3. Selective betting filter (Edge >= threshold, e.g. 1.5 pts / EV > 0).
4. Fractional Kelly Criterion bankroll growth simulation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


STANDARD_VIG_BREAKEVEN = 1.10 / 2.10  # ~0.5238 (52.38%)


def calculate_empirical_cover_probability(
    predicted_margins: np.ndarray | pd.Series,
    market_spreads: np.ndarray | pd.Series,
    historical_residuals: np.ndarray | pd.Series | None = None,
    default_std: float = 13.5,
) -> np.ndarray:
    """Calculate home cover probability: P(actual_margin > market_spread).

    If historical_residuals (actual_margin - predicted_margin on training data)
    are provided with sufficient sample (>= 30), evaluates the Empirical Cumulative
    Distribution Function (ECDF) to capture discrete NFL key-number clustering (3, 7, 6, 10).
    Otherwise, falls back to calibrated normal CDF.
    """
    p_m = np.asarray(predicted_margins, dtype=float)
    m_s = np.asarray(market_spreads, dtype=float)

    # Required residual difference for home to cover:
    # actual_margin > market_spread  <=>  predicted_margin + residual > market_spread
    # <=> residual > market_spread - predicted_margin
    cutoff = m_s - p_m

    if historical_residuals is not None:
        res = np.asarray(historical_residuals, dtype=float)
        res = res[~np.isnan(res)]
        if len(res) >= 30:
            # Vectorized ECDF: proportion of historical residuals strictly greater than cutoff
            # P(residual > cutoff) = mean(res > cutoff)
            probs = np.array([float(np.mean(res > c)) for c in cutoff])
            # Bound slightly away from 0/1 to avoid extreme zero probabilities
            return np.clip(probs, 0.01, 0.99)

    # Fallback to calibrated Gaussian CDF via math.erf
    sigma = max(float(default_std), 1.0)
    z = (p_m - m_s) / sigma
    erf_vec = np.vectorize(math.erf)
    probs = 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))
    return np.clip(probs, 0.01, 0.99)


def calculate_expected_value(cover_prob: float, american_odds: int = -110) -> float:
    """Calculate expected value per unit staked.

    For standard -110 juice:
    - Bet 1.10 units to win 1.00 profit.
    - If win: profit = +1.0
    - If loss: loss = -1.10
    - EV = p * 1.0 - (1 - p) * 1.10 = 2.10 * p - 1.10
    """
    p = float(cover_prob)
    if american_odds == -110:
        return 2.10 * p - 1.10
    elif american_odds < 0:
        stake = abs(american_odds) / 100.0
        return p * 1.0 - (1.0 - p) * stake
    else:
        win_payout = american_odds / 100.0
        return p * win_payout - (1.0 - p) * 1.0


def calculate_kelly_fraction(
    cover_prob: float,
    american_odds: int = -110,
    fraction_multiplier: float = 0.25,
    max_cap: float = 0.05,
) -> float:
    """Calculate fractional Kelly stake proportion.

    Full Kelly: f* = (b * p - q) / b
    where b = decimal profit ratio (for -110, b = 100/110 = 0.9091), q = 1 - p.
    Quarter Kelly (fraction_multiplier=0.25) protects against drawdown.
    """
    p = float(cover_prob)
    q = 1.0 - p
    b = 100.0 / abs(american_odds) if american_odds < 0 else american_odds / 100.0

    full_kelly = (b * p - q) / b
    if full_kelly <= 0:
        return 0.0

    stake = full_kelly * fraction_multiplier
    return min(float(stake), float(max_cap))


def evaluate_betting_tiers(
    out_of_sample_df: pd.DataFrame,
    edge_thresholds: list[float] | None = None,
    initial_bankroll: float = 100.0,
    flat_unit_size: float = 1.0,
    kelly_multiplier: float = 0.25,
) -> dict[str, Any]:
    """Evaluate betting performance across multiple edge threshold tiers.

    Columns expected in out_of_sample_df:
    - 'predicted_margin': model's predicted home margin
    - 'market_spread': closing home spread (positive = home favored)
    - 'actual_margin': actual home margin (home_score - away_score)
    - 'home_cover_prob': estimated home cover probability
    """
    if edge_thresholds is None:
        edge_thresholds = [0.0, 1.5, 2.5, 3.5, 5.0]

    df = out_of_sample_df.dropna(subset=["predicted_margin", "market_spread", "actual_margin"]).copy()
    if df.empty:
        return {}

    # Calculate model edge from home perspective:
    # Model likes home if predicted_margin > market_spread
    # Edge = |predicted_margin - market_spread|
    df["raw_edge"] = df["predicted_margin"] - df["market_spread"]
    df["abs_edge"] = df["raw_edge"].abs()

    # Determine model bet side:
    # If raw_edge > 0 -> bet Home Cover
    # If raw_edge < 0 -> bet Away Cover
    df["bet_home"] = df["raw_edge"] > 0
    df["bet_side_prob"] = np.where(df["bet_home"], df["home_cover_prob"], 1.0 - df["home_cover_prob"])

    # Actual cover outcome:
    # Home covered if actual_margin > market_spread
    # Push if actual_margin == market_spread
    df["home_covered"] = df["actual_margin"] > df["market_spread"]
    df["is_push"] = df["actual_margin"] == df["market_spread"]

    # Bet result: 1 (win), 0 (loss), None (push)
    df["bet_won"] = np.where(
        df["is_push"],
        np.nan,
        np.where(df["bet_home"] == df["home_covered"], 1.0, 0.0),
    )

    results = {}
    for threshold in edge_thresholds:
        tier_df = df[df["abs_edge"] >= threshold].copy()
        n_total_eligible = len(tier_df)
        if n_total_eligible == 0:
            continue

        non_push = tier_df.dropna(subset=["bet_won"])
        wins = int((non_push["bet_won"] == 1.0).sum())
        losses = int((non_push["bet_won"] == 0.0).sum())
        pushes = int(tier_df["is_push"].sum())
        n_graded = wins + losses

        win_rate = (wins / n_graded) if n_graded > 0 else 0.0

        # Flat unit betting ($100 stake / 1.10 risk per bet at -110):
        # Profit on win: +1.0 units, Loss on loss: -1.10 units, Push: 0 units
        flat_units_net = (wins * 1.0) - (losses * 1.10)
        total_staked_flat = n_graded * 1.10
        flat_roi = (flat_units_net / total_staked_flat) if total_staked_flat > 0 else 0.0

        # Kelly bankroll simulation
        bankroll = float(initial_bankroll)
        bankroll_history = [bankroll]
        for _, row in tier_df.iterrows():
            if pd.isna(row["bet_won"]):
                continue  # push: no change
            side_prob = row["bet_side_prob"]
            k_frac = calculate_kelly_fraction(side_prob, fraction_multiplier=kelly_multiplier)
            bet_amount = bankroll * k_frac
            if row["bet_won"] == 1.0:
                bankroll += bet_amount * (100.0 / 110.0)
            else:
                bankroll -= bet_amount
            bankroll_history.append(float(round(bankroll, 2)))

        tier_key = f"edge_ge_{threshold:.1f}pts" if threshold > 0 else "all_games"
        results[tier_key] = {
            "threshold_pts": threshold,
            "bets_placed": n_total_eligible,
            "record": f"{wins}-{losses}-{pushes}",
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "ats_win_rate": float(round(win_rate, 4)),
            "flat_units_net": float(round(flat_units_net, 2)),
            "flat_roi_pct": float(round(flat_roi * 100, 2)),
            "simulated_final_bankroll": float(round(bankroll, 2)),
            "simulated_bankroll_growth_pct": float(round(((bankroll - initial_bankroll) / initial_bankroll) * 100, 2)),
        }

    return results
