"""
Phase 10: Closing Line Value (CLV) & Market Movement Tracker.

Tracks how betting spreads shift between initial opening and final closing numbers,
measuring model closing line alpha (the gold standard of predictive sharpness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CLVMetric:
    game_id: str
    home_team_id: str
    away_team_id: str
    opening_spread: float | None
    closing_spread: float | None
    predicted_margin: float | None
    market_line_movement: float  # closing_spread - opening_spread
    clv_edge_points: float | None  # Positive if model aligned with the market direction
    beat_closing_line: bool | None  # True if model took a better line than where the market closed

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "opening_spread": self.opening_spread,
            "closing_spread": self.closing_spread,
            "predicted_margin": self.predicted_margin,
            "market_line_movement": round(self.market_line_movement, 2),
            "clv_edge_points": round(self.clv_edge_points, 2) if self.clv_edge_points is not None else None,
            "beat_closing_line": self.beat_closing_line,
        }


def calculate_clv(
    game_id: str,
    home_team_id: str,
    away_team_id: str,
    opening_spread: float | None,
    closing_spread: float | None,
    predicted_margin: float | None,
) -> CLVMetric:
    """Compute Closing Line Value (CLV) for a matchup.
    
    Convention: Spread is negative when home team is favored (e.g. -3.5).
    Margin is positive when home team wins by more points.
    
    Market line movement:
    If opening was -3.0 and closing moved to -4.5, movement is -1.5 (market steamed towards home).
    If model predicted margin was +6.0 (favoring home), picking home at -3.0 gained +1.5 pts of CLV!
    """
    if opening_spread is None or closing_spread is None:
        return CLVMetric(
            game_id=game_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            opening_spread=opening_spread,
            closing_spread=closing_spread,
            predicted_margin=predicted_margin,
            market_line_movement=0.0,
            clv_edge_points=None,
            beat_closing_line=None,
        )

    # Spread movement in points
    # (e.g., -4.5 closing - (-3.0 opening) = -1.5 points shift towards home)
    line_movement = closing_spread - opening_spread

    if predicted_margin is None:
        return CLVMetric(
            game_id=game_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            opening_spread=opening_spread,
            closing_spread=closing_spread,
            predicted_margin=predicted_margin,
            market_line_movement=line_movement,
            clv_edge_points=None,
            beat_closing_line=None,
        )

    # Model side preference at opening line:
    # Home is favored by -opening_spread. Model expects predicted_margin.
    # Edge on home at opening = predicted_margin - (-opening_spread) = predicted_margin + opening_spread
    opening_home_edge = predicted_margin + opening_spread

    # If opening_home_edge > 0, model liked home team.
    # CLV is positive if market moved towards home (line moved from -3.0 to -4.5, so closing_spread < opening_spread)
    if opening_home_edge > 0:
        # Model backed home
        clv_points = opening_spread - closing_spread
        beat_clv = clv_points > 0.0
    else:
        # Model backed away
        clv_points = closing_spread - opening_spread
        beat_clv = clv_points > 0.0

    return CLVMetric(
        game_id=game_id,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        opening_spread=opening_spread,
        closing_spread=closing_spread,
        predicted_margin=predicted_margin,
        market_line_movement=line_movement,
        clv_edge_points=clv_points,
        beat_closing_line=beat_clv,
    )
