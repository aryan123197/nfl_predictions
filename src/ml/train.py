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
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sqlalchemy import text

from src.db import get_engine, init_schema, qualified_table
from src.ml.features import FEATURE_COLUMNS, load_training_frame, split_features_target

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ml_train")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# A shallow, lightly-regularized baseline -- deliberately conservative given
# how little data exists per season (~270 games). Revisit once there's a
# real hyperparameter-tuning need, not before.
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
}

# Below this many holdout games, evaluation metrics are too noisy to mean
# much (a single upset swings accuracy by several points) -- warn rather
# than silently reporting a number that looks precise but isn't.
MIN_HOLDOUT_GAMES_FOR_RELIABLE_METRICS = 50


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


def train_and_evaluate(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> tuple[xgb.XGBClassifier, dict]:
    X_train, y_train = split_features_target(train_df)
    X_holdout, y_holdout = split_features_target(holdout_df)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_holdout)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_holdout, preds)),
        "log_loss": float(log_loss(y_holdout, probs, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_holdout, probs)),
        "n_train": len(train_df),
        "n_holdout": len(holdout_df),
    }
    if len(holdout_df) < MIN_HOLDOUT_GAMES_FOR_RELIABLE_METRICS:
        logger.warning("Holdout set has only %d games -- metrics are noisy at this size (see "
                        "MIN_HOLDOUT_GAMES_FOR_RELIABLE_METRICS), treat them as directional, not precise.",
                        len(holdout_df))
    return model, metrics


def save_model(model: xgb.XGBClassifier, metrics: dict, split_description: str, model_version: str) -> Path:
    out_dir = MODELS_DIR / model_version
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")

    metadata = {
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "feature_columns": FEATURE_COLUMNS,
        "target": "home_team_won",
        "split": split_description,
        "hyperparameters": XGB_PARAMS,
        "metrics": metrics,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return out_dir


def write_predictions(engine, holdout_df: pd.DataFrame, model: xgb.XGBClassifier, model_version: str) -> int:
    X_holdout, _ = split_features_target(holdout_df)
    probs = model.predict_proba(X_holdout)[:, 1]

    table = qualified_table("ml", "predictions")
    rows = []
    for (_, g), home_prob in zip(holdout_df.iterrows(), probs):
        rows.append({
            "game_id": g["game_id"],
            "model_version": model_version,
            "home_win_probability": float(home_prob),
            "away_win_probability": float(1.0 - home_prob),
            "predicted_home_score": None,   # score prediction: deferred, see module docstring
            "predicted_away_score": None,
            "predicted_margin": None,
            "market_spread": g["current_spread"] if pd.notna(g["current_spread"]) else None,
            "cover_probability": None,      # deferred alongside score/margin prediction
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
    parser = argparse.ArgumentParser(description="Train the Phase 4 XGBoost win-probability baseline.")
    parser.add_argument("--holdout-season", type=int, default=None,
                         help="Season to hold out for evaluation (default: the most recent season present).")
    args = parser.parse_args()
    run(holdout_season=args.holdout_season)


if __name__ == "__main__":
    main()
