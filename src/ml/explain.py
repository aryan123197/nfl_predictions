"""
Phase 8: Model Explainability Engine (Design Doc §36).

Computes TreeSHAP feature attributions for any game prediction to answer:
"Why does the model favor this team?"
Identifies top positive and negative factors directly from model decision paths.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db import get_engine, qualified_table
from src.ml.features import FEATURE_COLUMNS
from src.ml.registry import load_champion_model
from src.ml.train import NFLPredictionModel

logger = logging.getLogger("ml_explain")

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "elo_difference": "Elo Rating Advantage",
    "home_elo": "Home Team Elo Rating",
    "away_elo": "Away Team Elo Rating",
    "home_off_epa": "Home Offensive Efficiency (EPA/play)",
    "away_off_epa": "Away Offensive Efficiency (EPA/play)",
    "home_def_epa": "Home Defensive Strength (EPA allowed)",
    "away_def_epa": "Away Defensive Strength (EPA allowed)",
    "home_injury_impact": "Home Injury Health Impact",
    "away_injury_impact": "Away Injury Health Impact",
    "home_rest_days": "Home Rest Advantage",
    "away_rest_days": "Away Rest Advantage",
    "home_recent_form": "Home Recent Form (Last 4 Games)",
    "away_recent_form": "Away Recent Form (Last 4 Games)",
    "home_qb_epa": "Home Starting QB Passing EPA",
    "away_qb_epa": "Away Starting QB Passing EPA",
    "qb_epa_diff": "Starting QB Matchup Advantage",
    "home_qb_success_rate": "Home QB Passing Success Rate",
    "away_qb_success_rate": "Away QB Passing Success Rate",
    "home_turnover_rate": "Home Turnover Rate",
    "away_turnover_rate": "Away Turnover Rate",
    "home_pressure_rate": "Home Pass Rush Pressure Rate",
    "away_pressure_rate": "Away Pass Rush Pressure Rate",
    "home_explosive_play_rate": "Home Explosive Play Rate",
    "away_explosive_play_rate": "Away Explosive Play Rate",
    "home_third_down_rate": "Home 3rd Down Conversion Rate",
    "away_third_down_rate": "Away 3rd Down Conversion Rate",
    "home_success_rate": "Home Play Success Rate",
    "away_success_rate": "Away Play Success Rate",
    "opening_spread": "Opening Betting Spread",
    "current_spread": "Current Market Spread",
    "spread_movement": "Betting Line Movement",
}


def _human_readable_impact(feature_name: str, value: float, contrib: float, home_team: str, away_team: str) -> str:
    """Format a factor into natural language text."""
    display = FEATURE_DISPLAY_NAMES.get(feature_name, feature_name.replace("_", " ").title())
    if "elo" in feature_name:
        if contrib > 0:
            return f"+ Higher Elo strength for {home_team} ({value:.0f})"
        else:
            return f"- Stronger Elo rating for {away_team} ({value:.0f})"
    elif "rest" in feature_name:
        return f"{'+ Extra' if contrib > 0 else '- Less'} rest days ({int(value)} days)"
    elif "injury" in feature_name:
        if abs(value) > 0:
            return f"Injury impact penalty ({value:+.1f} pts)"
        return f"Clean injury report"
    elif "epa" in feature_name or "rate" in feature_name:
        return f"{display}: {value:+.2f}"
    elif "spread" in feature_name:
        return f"Market spread factor ({value:+.1f})"
    return f"{display} ({value:.2f})"


def explain_game_prediction(
    game_id: str,
    engine: Engine | None = None,
    model: NFLPredictionModel | None = None,
    top_n: int = 4,
) -> dict[str, Any] | None:
    """Compute TreeSHAP feature attributions for a given game prediction.
    
    Returns structured explanation showing base win probability, top positive factors
    (favoring home team), and top negative factors (favoring away team).
    """
    if engine is None:
        engine = get_engine()

    if model is None:
        champ_model, meta = load_champion_model()
        if champ_model is None:
            logger.warning("No champion model found for explainability.")
            return None
        model = champ_model

    gf_table = qualified_table("gold", "game_features")
    sg_table = qualified_table("silver", "games")

    query = text(f"""
        SELECT gf.*, sg.home_team_id, sg.away_team_id
        FROM {gf_table} gf
        JOIN {sg_table} sg ON gf.game_id = sg.game_id
        WHERE gf.game_id = :game_id
    """)

    with engine.connect() as conn:
        row = conn.execute(query, {"game_id": game_id}).fetchone()

    if not row:
        logger.warning("Game %s not found in gold.game_features/silver.games", game_id)
        return None

    row_dict = dict(row._mapping)
    home_team = row_dict["home_team_id"]
    away_team = row_dict["away_team_id"]

    # Assemble feature vector matching FEATURE_COLUMNS
    X = pd.DataFrame([{col: row_dict.get(col, np.nan) for col in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS).astype(float)

    # Compute TreeSHAP attributions using XGBoost's native booster
    booster = model.win_model.get_booster()
    dmatrix = xgb.DMatrix(X)
    shap_matrix = booster.predict(dmatrix, pred_contribs=True)
    
    # shap_matrix shape is (1, num_features + 1), where last column is bias / base margin
    contribs = shap_matrix[0, :-1]
    bias = float(shap_matrix[0, -1])

    # Base probability from intercept (sigmoid(bias))
    base_probability = float(1.0 / (1.0 + np.exp(-bias)))
    final_prob = float(model.win_model.predict_proba(X)[0, 1])

    positive_factors = []
    negative_factors = []

    for idx, col in enumerate(FEATURE_COLUMNS):
        raw_val = X.iloc[0][col]
        shap_val = float(contribs[idx])
        if abs(shap_val) < 0.001 or pd.isna(raw_val):
            continue

        desc = _human_readable_impact(col, raw_val, shap_val, home_team, away_team)
        factor = {
            "feature": col,
            "display_name": FEATURE_DISPLAY_NAMES.get(col, col.replace("_", " ").title()),
            "value": float(raw_val) if pd.notna(raw_val) else None,
            "contribution": round(shap_val, 4),
            "description": desc,
        }

        if shap_val > 0:
            positive_factors.append(factor)
        else:
            negative_factors.append(factor)

    # Sort factors by absolute magnitude of contribution
    positive_factors.sort(key=lambda f: f["contribution"], reverse=True)
    negative_factors.sort(key=lambda f: f["contribution"])

    favored_team = home_team if final_prob >= 0.5 else away_team
    win_prob = final_prob if final_prob >= 0.5 else (1.0 - final_prob)

    return {
        "game_id": game_id,
        "home_team_id": home_team,
        "away_team_id": away_team,
        "favored_team": favored_team,
        "win_probability": round(win_prob, 4),
        "home_win_probability": round(final_prob, 4),
        "away_win_probability": round(1.0 - final_prob, 4),
        "base_probability": round(base_probability, 4),
        "top_positive_factors": positive_factors[:top_n],
        "top_negative_factors": negative_factors[:top_n],
        "all_contributions": {
            f["feature"]: f["contribution"] for f in (positive_factors + negative_factors)
        },
    }
