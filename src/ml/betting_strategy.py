"""
Phase 10: Kelly Criterion & Expected Value (+EV) Betting Strategy Engine.

Calculates mathematical edge, Expected Value (EV), and optimal fractional Kelly
bankroll allocations for point spread and moneyline wagers against market odds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ValueTier(str, Enum):
    STRONG_VALUE = "STRONG_VALUE"      # EV >= 5.0%
    MODERATE_VALUE = "MODERATE_VALUE"  # 2.0% <= EV < 5.0%
    MARGINAL_VALUE = "MARGINAL_VALUE"  # 0.5% <= EV < 2.0%
    NO_BET = "NO_BET"                  # EV < 0.5%


@dataclass(frozen=True)
class KellySizing:
    full_kelly_pct: float
    half_kelly_pct: float
    quarter_kelly_pct: float
    recommended_units: float  # In standard units (where 1 unit = 1% bankroll, capped at 5 units)


@dataclass(frozen=True)
class BetRecommendation:
    game_id: str
    target_team_id: str
    bet_type: str  # "SPREAD" | "MONEYLINE"
    market_line: float  # Spread line (e.g. -3.5) or decimal odds
    model_probability: float  # Projected win or cover probability
    implied_probability: float  # Breakeven probability from market odds
    expected_value_pct: float  # EV% = (P * b - Q) * 100
    edge_pct: float  # Probability difference: Model Prob - Implied Prob
    value_tier: ValueTier
    sizing: KellySizing
    analysis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "target_team_id": self.target_team_id,
            "bet_type": self.bet_type,
            "market_line": self.market_line,
            "model_probability": round(self.model_probability, 4),
            "implied_probability": round(self.implied_probability, 4),
            "expected_value_pct": round(self.expected_value_pct, 2),
            "edge_pct": round(self.edge_pct, 2),
            "value_tier": self.value_tier.value,
            "full_kelly_pct": round(self.sizing.full_kelly_pct, 2),
            "half_kelly_pct": round(self.sizing.half_kelly_pct, 2),
            "quarter_kelly_pct": round(self.sizing.quarter_kelly_pct, 2),
            "recommended_units": round(self.sizing.recommended_units, 1),
            "analysis": self.analysis,
        }


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds (e.g. -110, +150) to decimal odds multiplier."""
    if american_odds > 0:
        return 1.0 + (american_odds / 100.0)
    else:
        return 1.0 + (100.0 / abs(american_odds))


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Calculate breakeven implied win probability from decimal odds."""
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def calculate_kelly_fraction(
    win_prob: float,
    decimal_odds: float = 1.9091,  # Standard -110 spread odds (10/11 payout)
) -> KellySizing:
    """Compute optimal Kelly Criterion bankroll fractions.
    
    Formula: f* = (b * p - q) / b
    where b = decimal_odds - 1 (net profit per unit staked), p = win_prob, q = 1 - win_prob.
    """
    if win_prob <= 0.0 or decimal_odds <= 1.0:
        return KellySizing(0.0, 0.0, 0.0, 0.0)

    b = decimal_odds - 1.0
    p = win_prob
    q = 1.0 - p

    raw_kelly = (b * p - q) / b

    if raw_kelly <= 0.0:
        return KellySizing(0.0, 0.0, 0.0, 0.0)

    full_pct = raw_kelly * 100.0
    half_pct = full_pct * 0.5
    quarter_pct = full_pct * 0.25

    # Standard recommended unit: Half Kelly capped safely between 0.5 and 4.0 units
    rec_units = max(0.5, min(4.0, half_pct))

    return KellySizing(
        full_kelly_pct=round(full_pct, 3),
        half_kelly_pct=round(half_pct, 3),
        quarter_kelly_pct=round(quarter_pct, 3),
        recommended_units=round(rec_units, 1),
    )


def calculate_expected_value(
    win_prob: float,
    decimal_odds: float = 1.9091,
) -> float:
    """Calculate Expected Value percentage: EV% = (P * (decimal_odds - 1) - (1 - P)) * 100%."""
    b = decimal_odds - 1.0
    ev = (win_prob * b) - (1.0 - win_prob)
    return ev * 100.0


def evaluate_spread_bet(
    game_id: str,
    home_team_id: str,
    away_team_id: str,
    market_spread: float | None,  # Current market spread (e.g. -3.5 => home favored by 3.5)
    home_cover_prob: float | None,
    american_odds: float = -110.0,
) -> BetRecommendation | None:
    """Evaluate against-the-spread (ATS) betting opportunities."""
    if market_spread is None or home_cover_prob is None:
        return None

    decimal_odds = american_to_decimal(american_odds)
    implied_prob = decimal_to_implied_prob(decimal_odds)

    # Determine which side has positive edge
    away_cover_prob = 1.0 - home_cover_prob

    if home_cover_prob >= implied_prob:
        target_team = home_team_id
        target_prob = home_cover_prob
        target_line = market_spread
    else:
        target_team = away_team_id
        target_prob = away_cover_prob
        target_line = -market_spread

    edge_pct = (target_prob - implied_prob) * 100.0
    ev_pct = calculate_expected_value(target_prob, decimal_odds)
    sizing = calculate_kelly_fraction(target_prob, decimal_odds)

    if ev_pct >= 5.0:
        tier = ValueTier.STRONG_VALUE
        analysis = (
            f"Strong +EV Edge (+{ev_pct:.1f}%): Model projects {target_team} {target_line:+.1f} "
            f"with {target_prob * 100:.1f}% cover rate vs {implied_prob * 100:.1f}% market breakeven."
        )
    elif ev_pct >= 2.0:
        tier = ValueTier.MODERATE_VALUE
        analysis = (
            f"Moderate +EV Edge (+{ev_pct:.1f}%): Model favors {target_team} {target_line:+.1f} "
            f"({target_prob * 100:.1f}% vs {implied_prob * 100:.1f}% breakeven)."
        )
    elif ev_pct >= 0.5:
        tier = ValueTier.MARGINAL_VALUE
        analysis = f"Marginal value (+{ev_pct:.1f}% EV). Consider passing or small quarter-unit wager."
    else:
        tier = ValueTier.NO_BET
        analysis = "Market is efficiently priced. No +EV edge detected."

    return BetRecommendation(
        game_id=game_id,
        target_team_id=target_team,
        bet_type="SPREAD",
        market_line=target_line,
        model_probability=target_prob,
        implied_probability=implied_prob,
        expected_value_pct=ev_pct,
        edge_pct=edge_pct,
        value_tier=tier,
        sizing=sizing,
        analysis=analysis,
    )
