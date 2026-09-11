"""
Pydantic response models -- the API's public contract.

Almost every field is Optional, and that is deliberate rather than lazy.
The warehouse is genuinely full of legitimate nulls: a scheduled game has no
score, a team's first game of a season has no rolling EPA or recent form, no
weather provider exists yet, and no model has predicted anything until Phase
4 lands. Typing those as required would force the API to invent values --
exactly the failure this project already fixed once, where a NULL read back
as NaN and poisoned everything downstream.

So: null means "genuinely not known", the frontend renders it as an explicit
absence, and no layer substitutes a plausible-looking zero.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class GameFeatures(BaseModel):
    """The model-facing view of a game, from gold.game_features."""

    home_elo: Optional[float] = None
    away_elo: Optional[float] = None
    elo_difference: Optional[float] = None
    home_injury_impact: Optional[float] = None
    away_injury_impact: Optional[float] = None
    home_recent_form: Optional[float] = Field(
        None, description="Win rate over the last 4 completed games strictly before this one"
    )
    away_recent_form: Optional[float] = None
    home_off_epa: Optional[float] = Field(None, description="Season-to-date offensive EPA/play, prior games only")
    away_off_epa: Optional[float] = None
    home_def_epa: Optional[float] = Field(None, description="Season-to-date EPA/play allowed, prior games only")
    away_def_epa: Optional[float] = None
    home_pass_epa: Optional[float] = None
    away_pass_epa: Optional[float] = None
    home_rush_epa: Optional[float] = None
    away_rush_epa: Optional[float] = None
    home_turnover_rate: Optional[float] = None
    away_turnover_rate: Optional[float] = None
    home_pressure_rate: Optional[float] = None
    away_pressure_rate: Optional[float] = None
    home_explosive_play_rate: Optional[float] = None
    away_explosive_play_rate: Optional[float] = None
    home_third_down_rate: Optional[float] = None
    away_third_down_rate: Optional[float] = None
    home_success_rate: Optional[float] = None
    away_success_rate: Optional[float] = None
    home_qb_id: Optional[str] = None
    away_qb_id: Optional[str] = None
    home_qb_epa: Optional[float] = None
    away_qb_epa: Optional[float] = None
    qb_epa_diff: Optional[float] = None
    home_qb_success_rate: Optional[float] = None
    away_qb_success_rate: Optional[float] = None
    home_qb_starter_change: Optional[float] = None
    away_qb_starter_change: Optional[float] = None
    opening_spread: Optional[float] = None
    current_spread: Optional[float] = None
    spread_movement: Optional[float] = Field(
        None, description="Always 0 in V1: one odds snapshot per game, see DECISIONS.md #2"
    )


class Prediction(BaseModel):
    """A model prediction, per design doc section 21.

    Served only when Phase 4 has produced one. The absence of this object on
    a game means no model has predicted it -- not a 50/50 forecast.
    """

    game_id: str
    model_version: Optional[str] = None
    prediction_timestamp: Optional[str] = None
    home_win_probability: Optional[float] = None
    away_win_probability: Optional[float] = None
    predicted_home_score: Optional[float] = None
    predicted_away_score: Optional[float] = None
    predicted_margin: Optional[float] = None
    market_spread: Optional[float] = None
    cover_probability: Optional[float] = None


class Game(BaseModel):
    game_id: str
    season: int
    week: Optional[int] = None
    game_date: Optional[str] = None
    status: str = Field(description="scheduled | final")
    home_team_id: str
    away_team_id: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    home_rest_days: Optional[int] = None
    away_rest_days: Optional[int] = None
    features: GameFeatures
    prediction: Optional[Prediction] = Field(
        None, description="null when no model has predicted this game yet"
    )


class Team(BaseModel):
    team_id: str
    first_season: Optional[int] = None
    last_season: Optional[int] = None
    elo: Optional[float] = Field(None, description="Most recent Elo rating, null before any rated game")


class ModelPerformance(BaseModel):
    """Design doc section 34's GET /model/performance.

    Reports availability explicitly instead of returning zeroed metrics,
    so the UI can say "no model yet" rather than showing a 0% accuracy that
    reads like a catastrophically bad model.
    """

    available: bool
    reason: Optional[str] = None
    model_version: Optional[str] = None
    games_evaluated: Optional[int] = None
    accuracy: Optional[float] = None
    brier_score: Optional[float] = None
    log_loss: Optional[float] = None
    mae_margin: Optional[float] = None
    ats_accuracy: Optional[float] = None


class ExplanationFactor(BaseModel):
    feature: str
    display_name: str
    value: Optional[float] = None
    contribution: float
    description: str


class PredictionExplanation(BaseModel):
    """TreeSHAP feature attributions for a game prediction (Design Doc §36)."""

    game_id: str
    home_team_id: str
    away_team_id: str
    favored_team: str
    win_probability: float
    home_win_probability: float
    away_win_probability: float
    base_probability: float
    top_positive_factors: list[ExplanationFactor] = []
    top_negative_factors: list[ExplanationFactor] = []


class Health(BaseModel):
    status: str
    warehouse_ready: bool = Field(description="False when the pipelines have never run against this database")
    predictions_available: bool = Field(description="False until Phase 4 lands ml.predictions")
    seasons: list[int] = []


class SystemHealthAudit(BaseModel):
    status: str
    passed_count: int
    warnings_count: int
    errors_count: int
    passed: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    metrics: dict[str, Any] = {}


class BettingRecommendation(BaseModel):
    game_id: str
    target_team_id: str
    bet_type: str = "SPREAD"
    market_line: float
    model_probability: float
    implied_probability: float
    expected_value_pct: float
    edge_pct: float
    value_tier: str
    full_kelly_pct: float
    half_kelly_pct: float
    quarter_kelly_pct: float
    recommended_units: float
    analysis: str


class BettingSlate(BaseModel):
    season: Optional[int] = None
    week: Optional[int] = None
    total_recommendations: int
    strong_value_count: int
    recommendations: list[BettingRecommendation] = []

