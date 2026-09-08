"""
Phase 6: Automated Weekly ML Retraining & Model Promotion Pipeline (design doc sections 26-29).

Workflow:
1. Updates actual game results and point-in-time gold features.
2. Checks the retraining cadence (DECISIONS.md #4: skip retraining in Weeks 1-3 unless forced).
3. Trains a candidate NFLPredictionModel on all available cumulative historical data.
4. Evaluates Candidate vs. Champion on holdout/validation games using the Decision #1 promotion gate.
5. Promotes and registers candidate if it improves performance without exceeding regression tolerances.
6. Records pipeline execution details and metrics to metadata.pipeline_runs.

Usage:
    python -m src.pipeline.run_retrain
    python -m src.pipeline.run_retrain --season 2025 --week 6
    python -m src.pipeline.run_retrain --season 2025 --week 2 --force-retrain
"""

from __future__ import annotations

import argparse
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error

from src.db import get_engine, init_schema
from src.ingest.pipeline_metadata import finish_run, start_run
from src.ml.features import load_training_frame, split_features_target
from src.ml.promotion import evaluate_candidate_promotion
from src.ml.registry import (
    load_champion_model,
    register_champion,
)
from src.ml.train import (
    NFLPredictionModel,
    XGB_CLASSIFIER_PARAMS,
    XGB_REGRESSOR_PARAMS,
    split_train_holdout,
)
from src.transform import game_results_transform, gold_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("retrain_runner")


def _evaluate_model_on_df(model: NFLPredictionModel, holdout_df: pd.DataFrame) -> dict[str, Any]:
    """Compute standard metrics (accuracy, log_loss, brier_score, MAE/RMSE) on evaluation dataset."""
    X_holdout, y_win_holdout = split_features_target(holdout_df)
    y_margin_holdout = holdout_df["actual_margin"].astype(float)
    y_total_holdout = (holdout_df["actual_home_score"] + holdout_df["actual_away_score"]).astype(float)

    probs = model.predict_proba(X_holdout)[:, 1]
    win_preds = (probs >= 0.5).astype(int)

    market_spreads = holdout_df["current_spread"].astype(float)
    pred_margins = model.predict_margin(X_holdout, market_spreads=market_spreads)
    pred_totals = model.predict_total(X_holdout)

    mae_margin = float(mean_absolute_error(y_margin_holdout, pred_margins))
    rmse_margin = float(np.sqrt(mean_squared_error(y_margin_holdout, pred_margins)))
    mae_total = float(mean_absolute_error(y_total_holdout, pred_totals))

    ats_accuracy = None
    has_spread = holdout_df["current_spread"].notna()
    if has_spread.sum() > 0:
        ats_preds = pred_margins[has_spread]
        ats_lines = -market_spreads[has_spread]
        ats_actuals = y_margin_holdout[has_spread]
        non_push = ats_actuals != ats_lines
        if non_push.sum() > 0:
            model_picked_home = ats_preds[non_push] > ats_lines[non_push]
            actual_home_covered = ats_actuals[non_push] > ats_lines[non_push]
            ats_accuracy = float(np.mean(model_picked_home == actual_home_covered))

    return {
        "accuracy": float(accuracy_score(y_win_holdout, win_preds)),
        "log_loss": float(log_loss(y_win_holdout, probs, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_win_holdout, probs)),
        "mae_margin": mae_margin,
        "rmse_margin": rmse_margin,
        "mae_total": mae_total,
        "ats_accuracy": ats_accuracy,
        "n_eval": len(holdout_df),
    }


def run(
    season: int | None = None,
    week: int | None = None,
    force_retrain: bool = False,
) -> dict:
    """Execute weekly ML retraining, candidate evaluation, and promotion gate."""
    init_schema()
    engine = get_engine()
    run_id = start_run("model_retraining_pipeline")

    logger.info("=== Starting Weekly ML Retraining Pipeline [run_id=%s, season=%s, week=%s] ===", run_id, season, week)

    try:
        # 1. Refresh Gold tables and Game Results
        logger.info("[Step 1/4] Refreshing gold features and game results...")
        gold_transform.run()
        game_results_transform.run()

        # 2. Check Retraining Cadence Rule (DECISIONS.md #4)
        if week is not None and week < 4 and not force_retrain:
            msg = (
                f"Skipping model weight retraining for Week {week} (< Week 4) per DECISIONS.md #4 "
                "to prevent sample overfitting on small early-season data. Pass --force-retrain to override."
            )
            logger.info(msg)
            finish_run(run_id, "success", records_processed=0)
            return {
                "run_id": run_id,
                "status": "skipped_cadence_gate",
                "message": msg,
                "promoted": False,
            }

        # 3. Load Cumulative Labeled Dataset
        logger.info("[Step 2/4] Loading cumulative training data...")
        df = load_training_frame(engine)
        if df.empty:
            raise RuntimeError("No labeled training data available in gold.game_features/ml.game_results")

        train_df, holdout_df, split_desc = split_train_holdout(df, holdout_season=season)
        if train_df.empty or holdout_df.empty:
            raise RuntimeError(f"Insufficient data split for training/evaluation ({len(train_df)} train, {len(holdout_df)} holdout)")

        # 4. Train Candidate Model
        logger.info("[Step 3/4] Training candidate model on %d games (evaluating on %d)...", len(train_df), len(holdout_df))
        X_train, y_win_train = split_features_target(train_df)
        y_margin_train = train_df["actual_margin"].astype(float)
        y_total_train = (train_df["actual_home_score"] + train_df["actual_away_score"]).astype(float)

        import xgboost as xgb
        win_model = xgb.XGBClassifier(**XGB_CLASSIFIER_PARAMS)
        win_model.fit(X_train, y_win_train)

        margin_model = xgb.XGBRegressor(**XGB_REGRESSOR_PARAMS)
        margin_model.fit(X_train, y_margin_train)

        total_model = xgb.XGBRegressor(**XGB_REGRESSOR_PARAMS)
        total_model.fit(X_train, y_total_train)

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

        candidate_model = NFLPredictionModel(
            win_model=win_model,
            margin_model=margin_model,
            total_model=total_model,
            spread_residual_model=spread_residual_model,
            margin_std=margin_std,
            historical_residuals=residuals,
        )

        candidate_metrics = _evaluate_model_on_df(candidate_model, holdout_df)
        candidate_version = f"candidate_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"

        # 5. Promotion Gate (Decision #1)
        logger.info("[Step 4/4] Evaluating candidate against champion via promotion gate...")
        champ_model, champ_meta = load_champion_model()

        promoted = False
        promotion_reason = ""

        if champ_model is None:
            # Initial bootstrap: no champion exists yet, promote candidate
            promoted = True
            promotion_reason = "Initial champion bootstrap -- no prior champion existed"
            logger.info("No prior champion found; promoting candidate as initial champion.")
            register_champion(
                model=candidate_model,
                metrics=candidate_metrics,
                model_version=f"champion_bootstrap_{datetime.now(timezone.utc):%Y%m%d}",
                training_metadata={"split": split_desc, "n_train": len(train_df)},
            )
        else:
            champ_metrics = _evaluate_model_on_df(champ_model, holdout_df)
            decision = evaluate_candidate_promotion(
                champion_metrics=champ_metrics,
                candidate_metrics=candidate_metrics,
            )
            promoted = decision.promoted
            promotion_reason = decision.reason
            logger.info("Promotion Gate Result: %s (Reason: %s)", "PROMOTED" if promoted else "REJECTED", decision.reason)

            if promoted:
                champion_version = f"champ_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
                register_champion(
                    model=candidate_model,
                    metrics=candidate_metrics,
                    model_version=champion_version,
                    training_metadata={
                        "split": split_desc,
                        "n_train": len(train_df),
                        "decision_reason": decision.reason,
                        "metric_deltas": decision.delta,
                    },
                )

        finish_run(run_id, "success", records_processed=len(train_df) + len(holdout_df))
        logger.info("=== Weekly Retraining Pipeline Completed Successfully [promoted=%s] ===", promoted)
        return {
            "run_id": run_id,
            "status": "success",
            "candidate_version": candidate_version,
            "candidate_metrics": candidate_metrics,
            "promoted": promoted,
            "promotion_reason": promotion_reason,
        }

    except Exception as e:
        logger.exception("Retraining pipeline failed with exception: %s", e)
        finish_run(run_id, "failed", records_processed=0, error_message=str(e))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the automated weekly ML retraining and promotion gate pipeline.")
    parser.add_argument("--season", type=int, default=None, help="Target holdout/evaluation season")
    parser.add_argument("--week", type=int, default=None, help="Current season week (to evaluate cadence rule)")
    parser.add_argument("--force-retrain", action="store_true", help="Override the Week 1-3 cadence gate")
    args = parser.parse_args()

    run(season=args.season, week=args.week, force_retrain=args.force_retrain)


if __name__ == "__main__":
    main()
