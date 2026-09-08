"""
Phase 4: Features -> XGBoost -> Predictions (design doc section 24).

Trains the XGBoost win-probability baseline on gold.game_features,
evaluates it on a held-out season (or, if only one season of data
exists, a held-out tail of weeks within that season), and writes its
predictions into ml.predictions.

Scope (see DECISIONS.md): win probability only. Score and spread
prediction (design doc section 20) are deferred to a follow-up slice --
sharing this same training/evaluation scaffolding once it exists.

This is NOT the walk-forward retraining loop (design doc section 25,
Phase 5) -- that simulates week-by-week retraining across a season.
This is Phase 4's own scope per the design doc's phase breakdown
("Phase 4: Features -> XGBoost -> Predictions"): one split, one model,
proving the training/evaluation/prediction mechanics work end to end
before Phase 5 wraps a retraining loop around them.

Model versioning here is a plain models/<version>/ directory with a
metadata.json (training data, features, hyperparameters, metrics, git
commit -- the fields design doc section 27 calls for) rather than
MLflow -- MLflow is explicitly Phase 8 (MLOps) in this project's own
phase table, not Phase 4.

Usage:
    python -m src.ml.train
    python -m src.ml.train --holdout-season 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
from sqlalchemy import text

from src.db import get_engine, init_schema, qualified_table
from src.ml.features import FEATURE_COLUMNS, load_training_frame, split_features_target

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ml_train")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Hyperparameters for win probability classifier and score/margin regressors
XGB_CLASSIFIER_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
}

XGB_REGRESSOR_PARAMS = {
    "n_estimators": 150,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mae",
}

# Below this many holdout games, evaluation metrics are too noisy to mean
# much (a single upset swings accuracy by several points) -- warn rather
# than silently reporting a number that looks precise but isn't.
MIN_HOLDOUT_GAMES_FOR_RELIABLE_METRICS = 50


class NFLPredictionModel:
    """Multi-target prediction bundle for NFL games:
    - Win probability (binary classification)
    - Standalone Margin (continuous regression: Home Score - Away Score)
    - Spread Residual Model (predicts residual adjustment: Actual Margin - Market Spread)
    - Total points (continuous regression: Home Score + Away Score)
    - ATS cover probability (derived from empirical residual ECDF / key numbers).
    """

    def __init__(
        self,
        win_model: xgb.XGBClassifier,
        margin_model: xgb.XGBRegressor,
        total_model: xgb.XGBRegressor,
        spread_residual_model: xgb.XGBRegressor | None = None,
        margin_std: float = 13.5,
        historical_residuals: np.ndarray | None = None,
    ):
        self.win_model = win_model
        self.margin_model = margin_model
        self.total_model = total_model
        self.spread_residual_model = spread_residual_model
        self.margin_std = float(margin_std) if margin_std > 1.0 else 13.5
        self.historical_residuals = historical_residuals

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.win_model.predict_proba(X)

    def predict_margin(self, X: pd.DataFrame, market_spreads: pd.Series | np.ndarray | None = None) -> np.ndarray:
        pred = self.margin_model.predict(X)
        if self.spread_residual_model is not None and market_spreads is not None:
            spreads = pd.Series(market_spreads).values
            has_spread = pd.notna(spreads)
            if np.any(has_spread):
                delta = self.spread_residual_model.predict(X)
                pred = np.where(has_spread, spreads + delta, pred)
        return pred

    def predict_total(self, X: pd.DataFrame) -> np.ndarray:
        return self.total_model.predict(X)

    def predict_scores(self, X: pd.DataFrame, market_spreads: pd.Series | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        margin = self.predict_margin(X, market_spreads)
        total = self.predict_total(X)
        home_score = (total + margin) / 2.0
        away_score = (total - margin) / 2.0
        return home_score, away_score

    def predict_cover_proba(self, X: pd.DataFrame, market_spreads: pd.Series | np.ndarray) -> np.ndarray:
        from src.ml.betting import calculate_empirical_cover_probability
        margin = self.predict_margin(X, market_spreads)
        return calculate_empirical_cover_probability(
            predicted_margins=margin,
            market_spreads=market_spreads,
            historical_residuals=self.historical_residuals,
            default_std=self.margin_std,
        )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def split_train_holdout(df: pd.DataFrame, holdout_season: int | None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Point-in-time-safe train/holdout split.

    Prefers a season-based split (train on every season strictly before
    `holdout_season`, evaluate on `holdout_season`) -- clean, and matches
    how a real production retrain would work (train on everything known,
    evaluate on the next season). Falls back to a week-based split
    within a single season when only one season of data exists (holding
    out its last quarter of weeks), since a season-based split would
    otherwise leave nothing to train on. Individual games within the
    holdout set are still point-in-time correct regardless of which
    split is used -- that's a property of gold.game_features itself
    (Phase 3), not of this split logic.
    """
    seasons = sorted(df["season"].unique())
    if holdout_season is None:
        holdout_season = seasons[-1]

    train = df[df["season"] < holdout_season]
    holdout = df[df["season"] == holdout_season]

    if train.empty and len(seasons) == 1:
        season = seasons[0]
        weeks = sorted(df["week"].unique())
        cutoff_week = weeks[int(len(weeks) * 0.75)]
        train = df[df["week"] < cutoff_week]
        holdout = df[df["week"] >= cutoff_week]
        logger.warning("Only one season (%s) of data available -- falling back to a within-season "
                        "split at week %s instead of a season holdout.", season, cutoff_week)
        return train, holdout, f"weeks<{cutoff_week}_of_{season}"

    return train, holdout, f"season<{holdout_season}"


def train_and_evaluate(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> tuple[NFLPredictionModel, dict]:
    X_train, y_win_train = split_features_target(train_df)
    X_holdout, y_win_holdout = split_features_target(holdout_df)

    y_margin_train = train_df["actual_margin"].astype(float)
    y_total_train = (train_df["actual_home_score"] + train_df["actual_away_score"]).astype(float)

    y_margin_holdout = holdout_df["actual_margin"].astype(float)
    y_total_holdout = (holdout_df["actual_home_score"] + holdout_df["actual_away_score"]).astype(float)

    # 1. Fit Win Probability Classifier
    win_model = xgb.XGBClassifier(**XGB_CLASSIFIER_PARAMS)
    win_model.fit(X_train, y_win_train)

    # 2. Fit Standalone Margin Regressor
    margin_model = xgb.XGBRegressor(**XGB_REGRESSOR_PARAMS)
    margin_model.fit(X_train, y_margin_train)

    # 3. Fit Total Score Regressor
    total_model = xgb.XGBRegressor(**XGB_REGRESSOR_PARAMS)
    total_model.fit(X_train, y_total_train)

    # 4. Fit Spread Residual Regressor
    has_spread_train = train_df["current_spread"].notna()
    spread_residual_model = None
    if has_spread_train.sum() >= 20:
        spread_residual_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mae",
        )
        y_ats_train = (train_df["actual_margin"] - train_df["current_spread"]).astype(float)
        spread_residual_model.fit(X_train[has_spread_train], y_ats_train[has_spread_train])

    # Estimate training residual std for margin
    train_pred_margins = (
        np.where(
            has_spread_train.values,
            train_df["current_spread"].values + spread_residual_model.predict(X_train),
            margin_model.predict(X_train),
        )
        if spread_residual_model is not None
        else margin_model.predict(X_train)
    )
    residuals = y_margin_train.values - train_pred_margins
    margin_std = float(np.std(residuals)) if len(residuals) > 1 else 13.5
    if margin_std < 1.0:
        margin_std = 13.5

    model = NFLPredictionModel(
        win_model=win_model,
        margin_model=margin_model,
        total_model=total_model,
        spread_residual_model=spread_residual_model,
        margin_std=margin_std,
        historical_residuals=residuals,
    )

    # Evaluate Win Probability
    probs = model.predict_proba(X_holdout)[:, 1]
    win_preds = (probs >= 0.5).astype(int)

    # Evaluate Margin & Total (with market spreads for residual adjustment)
    market_spreads_holdout = holdout_df["current_spread"].astype(float)
    pred_margins = model.predict_margin(X_holdout, market_spreads=market_spreads_holdout)
    pred_totals = model.predict_total(X_holdout)

    mae_margin = float(mean_absolute_error(y_margin_holdout, pred_margins))
    rmse_margin = float(np.sqrt(mean_squared_error(y_margin_holdout, pred_margins)))
    mae_total = float(mean_absolute_error(y_total_holdout, pred_totals))

    # Evaluate ATS Accuracy on games with odds (excluding pushes)
    market_spreads = holdout_df["current_spread"].astype(float)
    has_odds = market_spreads.notna()
    ats_accuracy = None
    if has_odds.sum() > 0:
        ats_preds = pred_margins[has_odds]
        ats_lines = market_spreads[has_odds].values
        ats_actuals = y_margin_holdout[has_odds].values

        non_push = (ats_actuals != ats_lines) & (ats_preds != ats_lines)
        if non_push.sum() > 0:
            model_picked_home = ats_preds[non_push] > ats_lines[non_push]
            actual_home_covered = ats_actuals[non_push] > ats_lines[non_push]
            ats_accuracy = float(np.mean(model_picked_home == actual_home_covered))

    metrics = {
        "accuracy": float(accuracy_score(y_win_holdout, win_preds)),
        "log_loss": float(log_loss(y_win_holdout, probs, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_win_holdout, probs)),
        "mae_margin": mae_margin,
        "rmse_margin": rmse_margin,
        "mae_total": mae_total,
        "ats_accuracy": ats_accuracy,
        "n_train": len(train_df),
        "n_holdout": len(holdout_df),
    }
    if len(holdout_df) < MIN_HOLDOUT_GAMES_FOR_RELIABLE_METRICS:
        logger.warning("Holdout set has only %d games -- metrics are noisy at this size (see "
                        "MIN_HOLDOUT_GAMES_FOR_RELIABLE_METRICS), treat them as directional, not precise.",
                        len(holdout_df))
    return model, metrics


def save_model(model: NFLPredictionModel, metrics: dict, split_description: str, model_version: str) -> Path:
    out_dir = MODELS_DIR / model_version
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")

    metadata = {
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "feature_columns": FEATURE_COLUMNS,
        "target": ["home_team_won", "actual_margin", "actual_total"],
        "split": split_description,
        "hyperparameters": {
            "classifier": XGB_CLASSIFIER_PARAMS,
            "regressor": XGB_REGRESSOR_PARAMS,
        },
        "margin_std": model.margin_std,
        "metrics": metrics,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return out_dir


def write_predictions(engine, holdout_df: pd.DataFrame, model: NFLPredictionModel, model_version: str) -> int:
    X_holdout, _ = split_features_target(holdout_df)
    probs = model.predict_proba(X_holdout)[:, 1]
    home_scores, away_scores = model.predict_scores(X_holdout)
    margins = model.predict_margin(X_holdout)
    market_spreads = holdout_df["current_spread"].astype(float)
    cover_probs = model.predict_cover_proba(X_holdout, market_spreads)

    table = qualified_table("ml", "predictions")
    rows = []
    for i, (_, g) in enumerate(holdout_df.iterrows()):
        m_spread = float(g["current_spread"]) if pd.notna(g["current_spread"]) else None
        c_prob = float(cover_probs[i]) if m_spread is not None else None
        rows.append({
            "game_id": g["game_id"],
            "model_version": model_version,
            "home_win_probability": float(probs[i]),
            "away_win_probability": float(1.0 - probs[i]),
            "predicted_home_score": float(round(home_scores[i], 1)),
            "predicted_away_score": float(round(away_scores[i], 1)),
            "predicted_margin": float(round(margins[i], 1)),
            "market_spread": m_spread,
            "cover_probability": float(round(c_prob, 4)) if c_prob is not None else None,
            "feature_snapshot_id": model_version,
        })
    if not rows:
        return 0
    with engine.begin() as conn:
        cols = list(rows[0].keys())
        placeholders = ", ".join(f":{c}" for c in cols)
        col_list = ", ".join(cols)
        # INSERT only -- ml.predictions is never updated or replaced (design
        # doc section 21: "Predictions must NEVER be overwritten"). Re-running
        # training adds new rows under a new model_version, it never touches
        # a prior run's rows.
        conn.execute(text(f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"), rows)
    return len(rows)


def run(holdout_season: int | None = None) -> dict:
    init_schema()
    engine = get_engine()

    df = load_training_frame(engine)
    if df.empty:
        raise RuntimeError("No labeled training data in gold.game_features/ml.game_results -- "
                            "run the full ingestion -> silver -> gold -> game_results_transform "
                            "pipeline for at least one season with completed games first.")

    train_df, holdout_df, split_description = split_train_holdout(df, holdout_season)
    if train_df.empty or holdout_df.empty:
        raise RuntimeError(f"Split '{split_description}' produced an empty train or holdout set "
                            f"(train={len(train_df)}, holdout={len(holdout_df)}) -- need more data.")

    model, metrics = train_and_evaluate(train_df, holdout_df)
    # Timestamp alone isn't unique enough -- two runs within the same second
    # (e.g. back-to-back retrains in an automated pipeline, or two runs in a
    # test) would collide on model_version, silently merging what should be
    # two distinct, independently-queryable prediction sets in ml.predictions.
    # The timestamp prefix stays for human readability; the hex suffix is
    # what actually guarantees uniqueness.
    model_version = f"xgboost_v1_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
    out_dir = save_model(model, metrics, split_description, model_version)
    prediction_count = write_predictions(engine, holdout_df, model, model_version)

    logger.info("Trained %s on %d games, evaluated on %d (%s)", model_version, len(train_df),
                len(holdout_df), split_description)
    logger.info("Metrics: %s", metrics)
    logger.info("Model saved to %s", out_dir)
    logger.info("Wrote %d rows to ml.predictions", prediction_count)

    return {"model_version": model_version, "metrics": metrics, "predictions_written": prediction_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Phase 4/7 XGBoost win-probability, score, and spread models.")
    parser.add_argument("--holdout-season", type=int, default=None,
                         help="Season to hold out for evaluation (default: the most recent season present).")
    args = parser.parse_args()
    run(holdout_season=args.holdout_season)


if __name__ == "__main__":
    main()
