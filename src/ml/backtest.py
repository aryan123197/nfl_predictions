"""
Phase 5: Walk-Forward Backtesting (design doc sections 25-28).

Simulates week-by-week real-world operation and continuous retraining across
an NFL season (default: 2025):
1. Trains an initial baseline champion model on all historical seasons prior
   to the target season.
2. For each week W in the target season:
   a. Generates out-of-sample pre-game predictions using the active champion model.
   b. Writes point-in-time predictions to ml.predictions (insert-only).
   c. Evaluates prediction accuracy/loss against actual game results.
   d. Applies the retraining schedule (DECISIONS.md #4):
      - Weeks 1-3: skips current-season retraining to avoid overfitting on small samples.
      - Weeks 4+: trains candidate model on all available history up to week W,
        evaluates candidate vs champion using the multi-metric promotion gate
        (DECISIONS.md #1), and promotes candidate if criteria are met.
3. Computes cumulative season metrics and outputs a comprehensive backtest summary report.

Usage:
    python -m src.ml.backtest --season 2025
    python -m src.ml.backtest --season 2025 --retrain-start-week 4
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db import get_engine, init_schema, qualified_table
from src.ml.features import FEATURE_COLUMNS, TARGET_COLUMN, load_training_frame, split_features_target
from src.ml.promotion import evaluate_candidate_promotion, PromotionDecision
from src.ml.train import (
    NFLPredictionModel,
    XGB_CLASSIFIER_PARAMS,
    XGB_REGRESSOR_PARAMS,
    _git_commit,
    save_model,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ml_backtest")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
BACKTESTS_DIR = MODELS_DIR / "backtests"


def train_model(train_df: pd.DataFrame) -> NFLPredictionModel:
    """Train the full NFLPredictionModel bundle (win classifier + standalone margin + spread residual + total regressor)."""
    X_train, y_win_train = split_features_target(train_df)
    y_margin_train = train_df["actual_margin"].astype(float)
    y_total_train = (train_df["actual_home_score"] + train_df["actual_away_score"]).astype(float)

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

    return NFLPredictionModel(
        win_model=win_model,
        margin_model=margin_model,
        total_model=total_model,
        spread_residual_model=spread_residual_model,
        margin_std=margin_std,
        historical_residuals=residuals,
    )


def compute_metrics(
    y_true_win: pd.Series | np.ndarray,
    probs: np.ndarray,
    y_true_margin: pd.Series | np.ndarray | None = None,
    pred_margins: np.ndarray | None = None,
    market_spreads: pd.Series | np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute calibration, margin error, and ATS accuracy metrics."""
    y_true_arr = np.asarray(y_true_win).astype(int)
    probs_arr = np.asarray(probs, dtype=float)
    preds = (probs_arr >= 0.5).astype(int)

    acc = float(accuracy_score(y_true_arr, preds)) if len(y_true_arr) > 0 else 0.0
    ll = float(log_loss(y_true_arr, probs_arr, labels=[0, 1])) if len(y_true_arr) > 0 else 0.0
    brier = float(brier_score_loss(y_true_arr, probs_arr)) if len(y_true_arr) > 0 else 0.0

    mae_margin = None
    rmse_margin = None
    ats_acc = None

    if y_true_margin is not None and pred_margins is not None:
        y_m = np.asarray(y_true_margin, dtype=float)
        p_m = np.asarray(pred_margins, dtype=float)
        if len(y_m) > 0:
            mae_margin = float(mean_absolute_error(y_m, p_m))
            rmse_margin = float(np.sqrt(mean_squared_error(y_m, p_m)))

    if y_true_margin is not None and pred_margins is not None and market_spreads is not None:
        y_m = np.asarray(y_true_margin, dtype=float)
        p_m = np.asarray(pred_margins, dtype=float)
        spreads = np.asarray(market_spreads, dtype=float)
        has_odds = ~np.isnan(spreads)
        if has_odds.sum() > 0:
            valid_y = y_m[has_odds]
            valid_p = p_m[has_odds]
            valid_s = spreads[has_odds]
            non_push = (valid_y != valid_s) & (valid_p != valid_s)
            if non_push.sum() > 0:
                picked_home = valid_p[non_push] > valid_s[non_push]
                actual_covered = valid_y[non_push] > valid_s[non_push]
                ats_acc = float(np.mean(picked_home == actual_covered))

    return {
        "accuracy": acc,
        "log_loss": ll,
        "brier_score": brier,
        "mae_margin": mae_margin,
        "rmse_margin": rmse_margin,
        "ats_accuracy": ats_acc,
        "n_games": len(y_true_arr),
    }


def write_week_predictions(
    engine: Engine,
    week_df: pd.DataFrame,
    model: NFLPredictionModel,
    model_version: str,
) -> int:
    """Insert point-in-time predictions (win prob, scores, margin, cover prob) into ml.predictions."""
    X_week, _ = split_features_target(week_df)
    probs = model.predict_proba(X_week)[:, 1]
    market_spreads = week_df["current_spread"].astype(float)
    margins = model.predict_margin(X_week, market_spreads=market_spreads)
    home_scores, away_scores = model.predict_scores(X_week, market_spreads=market_spreads)
    cover_probs = model.predict_cover_proba(X_week, market_spreads)

    table = qualified_table("ml", "predictions")
    rows = []
    for i, (_, g) in enumerate(week_df.iterrows()):
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
        conn.execute(text(f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"), rows)
    return len(rows)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def run_walk_forward_backtest(
    season: int = 2025,
    start_week: int = 1,
    end_week: int | None = None,
    retrain_start_week: int = 4,
    write_db: bool = True,
    save_models: bool = False,
    engine: Engine | None = None,
) -> dict[str, Any]:
    """Execute the sequential walk-forward simulation across target season weeks.

    Parameters
    ----------
    season: target season to backtest (e.g. 2025)
    start_week: starting week for predictions (default: 1)
    end_week: optional ending week (default: max week in season)
    retrain_start_week: week to begin retraining per Decision #4 (default: 4)
    write_db: whether to insert predictions into ml.predictions
    save_models: whether to persist model artifacts to models/
    engine: optional SQLAlchemy engine (defaults to get_engine())

    Returns
    -------
    Dictionary summarizing overall backtest performance, weekly breakdown,
    and model lineage.
    """
    init_schema()
    if engine is None:
        engine = get_engine()

    all_df = load_training_frame(engine)
    if all_df.empty:
        raise RuntimeError("No labeled training data found in gold.game_features/ml.game_results.")

    seasons_available = sorted(all_df["season"].unique())
    if season not in seasons_available:
        raise ValueError(f"Season {season} not found in available seasons: {seasons_available}")

    # Prior seasons for initial preseason training
    preseason_train_df = all_df[all_df["season"] < season]
    if preseason_train_df.empty:
        # Fallback for single-season test fixtures
        logger.warning("No prior seasons before %d. Initializing base model on early slice of season %d.", season, season)
        season_df = all_df[all_df["season"] == season]
        min_week = season_df["week"].min()
        preseason_train_df = season_df[season_df["week"] <= min_week]

    champion_version = f"champ_{season}_preseason_{datetime.now(timezone.utc):%Y%m%d}_{uuid.uuid4().hex[:6]}"
    champion_model = train_model(preseason_train_df)
    if save_models:
        save_model(champion_model, {"n_train": len(preseason_train_df)}, f"season<{season}", champion_version)
    logger.info("Initialized preseason champion model %s on %d historical games.", champion_version, len(preseason_train_df))

    season_df = all_df[all_df["season"] == season]
    available_weeks = sorted(w for w in season_df["week"].unique() if w >= start_week)
    if end_week is not None:
        available_weeks = [w for w in available_weeks if w <= end_week]

    if not available_weeks:
        raise ValueError(f"No games available in season {season} for weeks >= {start_week}")

    weekly_results: list[dict[str, Any]] = []
    lineage_log: list[dict[str, Any]] = []
    all_out_of_sample_y: list[int] = []
    all_out_of_sample_probs: list[float] = []
    all_out_of_sample_y_margin: list[float] = []
    all_out_of_sample_pred_margin: list[float] = []
    all_out_of_sample_spreads: list[float] = []
    all_out_of_sample_cover_probs: list[float] = []

    # Sequential walk-forward loop
    for week in available_weeks:
        week_games = season_df[season_df["week"] == week]
        if week_games.empty:
            continue

        # Step 1: Pre-game prediction with active champion
        X_week, y_week = split_features_target(week_games)
        probs_week = champion_model.predict_proba(X_week)[:, 1]
        spreads_week = week_games["current_spread"].astype(float).values
        margins_week = champion_model.predict_margin(X_week, market_spreads=spreads_week)
        y_margin_week = week_games["actual_margin"].astype(float).values
        cover_probs_week = champion_model.predict_cover_proba(X_week, spreads_week)

        # Record out-of-sample predictions
        all_out_of_sample_y.extend(y_week.tolist())
        all_out_of_sample_probs.extend(probs_week.tolist())
        all_out_of_sample_y_margin.extend(y_margin_week.tolist())
        all_out_of_sample_pred_margin.extend(margins_week.tolist())
        all_out_of_sample_spreads.extend(spreads_week.tolist())
        all_out_of_sample_cover_probs.extend(cover_probs_week.tolist())

        pred_version_tag = f"backtest_{season}_w{week:02d}_{champion_version}"
        if write_db:
            write_week_predictions(engine, week_games, champion_model, pred_version_tag)

        # Step 2: Post-game evaluation for this week
        week_metrics = compute_metrics(
            y_true_win=y_week,
            probs=probs_week,
            y_true_margin=y_margin_week,
            pred_margins=margins_week,
            market_spreads=spreads_week,
        )
        weekly_results.append({
            "week": int(week),
            "n_games": len(week_games),
            "model_version": champion_version,
            "accuracy": week_metrics["accuracy"],
            "log_loss": week_metrics["log_loss"],
            "brier_score": week_metrics["brier_score"],
            "mae_margin": week_metrics["mae_margin"],
            "ats_accuracy": week_metrics["ats_accuracy"],
        })
        ats_str = f" | ATS: {week_metrics['ats_accuracy']*100:.1f}%" if week_metrics["ats_accuracy"] is not None else ""
        mae_str = f" | MAE: {week_metrics['mae_margin']:.1f}" if week_metrics["mae_margin"] is not None else ""
        logger.info("Week %02d [%s]: %d games | Acc: %.1f%%%s%s | LogLoss: %.4f | Brier: %.4f",
                    week, champion_version, len(week_games),
                    week_metrics["accuracy"] * 100, ats_str, mae_str, week_metrics["log_loss"], week_metrics["brier_score"])

        # Step 3: Retraining & Candidate Promotion (Decision #4 & Decision #1)
        if week < retrain_start_week:
            lineage_log.append({
                "week": int(week),
                "action": "retained",
                "champion_version": champion_version,
                "reason": f"Week {week} < {retrain_start_week}: skipping current-season retrain per Decision #4",
            })
            continue
        # Week >= retrain_start_week: Train and evaluate candidate strictly out-of-sample
        # Validation window: the completed week W games (out-of-sample for candidate pre-eval training)
        validation_df = week_games.copy()

        # Training set for evaluating candidate before promotion: strictly prior to validation window
        candidate_eval_train_df = all_df[
            (all_df["season"] < season) | ((all_df["season"] == season) & (all_df["week"] < week))
        ]

        if len(candidate_eval_train_df) >= 8 and len(candidate_eval_train_df[TARGET_COLUMN].unique()) > 1:
            X_val, y_val = split_features_target(validation_df)
            val_spreads = validation_df["current_spread"].astype(float).values
            champ_val_probs = champion_model.predict_proba(X_val)[:, 1]
            champ_val_metrics = compute_metrics(
                y_true_win=y_val,
                probs=champ_val_probs,
                y_true_margin=validation_df["actual_margin"].astype(float).values,
                pred_margins=champion_model.predict_margin(X_val, market_spreads=val_spreads),
                market_spreads=val_spreads,
            )

            cand_eval_model = train_model(candidate_eval_train_df)
            cand_val_probs = cand_eval_model.predict_proba(X_val)[:, 1]
            cand_val_metrics = compute_metrics(
                y_true_win=y_val,
                probs=cand_val_probs,
                y_true_margin=validation_df["actual_margin"].astype(float).values,
                pred_margins=cand_eval_model.predict_margin(X_val, market_spreads=val_spreads),
                market_spreads=val_spreads,
            )

            decision: PromotionDecision = evaluate_candidate_promotion(
                champion_metrics=champ_val_metrics,
                candidate_metrics=cand_val_metrics,
            )
        else:
            # Not enough pre-eval data to construct separate validation split
            decision = PromotionDecision(
                promoted=False,
                reason="Insufficient historical data prior to validation week to evaluate candidate out-of-sample",
                delta={},
                champion_metrics={},
                candidate_metrics={},
            )

        if decision.promoted:
            new_version = f"champ_{season}_w{week:02d}_{datetime.now(timezone.utc):%Y%m%d}_{uuid.uuid4().hex[:6]}"
            logger.info("Week %02d: PROMOTING new candidate model %s over %s. Reason: %s",
                        week, new_version, champion_version, decision.reason)
            # Refit on all completed data up to week W to form the new champion
            full_train_df = all_df[
                (all_df["season"] < season) | ((all_df["season"] == season) & (all_df["week"] <= week))
            ]
            champion_model = train_model(full_train_df)
            champion_version = new_version
            if save_models:
                save_model(champion_model, cand_val_metrics, f"seasons<{season}_or_week<={week}", champion_version)

            lineage_log.append({
                "week": int(week),
                "action": "promoted",
                "champion_version": champion_version,
                "reason": decision.reason,
                "delta": decision.delta,
            })
        else:
            logger.info("Week %02d: REJECTING candidate. Champion %s retained. Reason: %s",
                        week, champion_version, decision.reason)
            lineage_log.append({
                "week": int(week),
                "action": "retained",
                "champion_version": champion_version,
                "reason": decision.reason,
                "delta": decision.delta,
            })

    # Step 4: Compute cumulative backtest performance
    cumulative_metrics = compute_metrics(
        y_true_win=np.array(all_out_of_sample_y),
        probs=np.array(all_out_of_sample_probs),
        y_true_margin=np.array(all_out_of_sample_y_margin),
        pred_margins=np.array(all_out_of_sample_pred_margin),
        market_spreads=np.array(all_out_of_sample_spreads),
    )

    # Step 5: Evaluate selective betting tiers and Kelly bankroll simulation
    from src.ml.betting import evaluate_betting_tiers
    out_of_sample_df = pd.DataFrame({
        "predicted_margin": all_out_of_sample_pred_margin,
        "market_spread": all_out_of_sample_spreads,
        "actual_margin": all_out_of_sample_y_margin,
        "home_cover_prob": all_out_of_sample_cover_probs,
    })
    betting_tier_results = evaluate_betting_tiers(out_of_sample_df)

    backtest_id = f"backtest_{season}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    summary = {
        "backtest_id": backtest_id,
        "season": int(season),
        "start_week": int(available_weeks[0]),
        "end_week": int(available_weeks[-1]),
        "retrain_start_week": int(retrain_start_week),
        "git_commit": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cumulative_metrics": cumulative_metrics,
        "betting_tiers": betting_tier_results,
        "weekly_results": weekly_results,
        "lineage_log": lineage_log,
    }

    # Save summary report artifact
    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = BACKTESTS_DIR / f"{backtest_id}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    logger.info("==================================================")
    logger.info("WALK-FORWARD BACKTEST COMPLETE: Season %d (Weeks %d..%d)", season, available_weeks[0], available_weeks[-1])
    logger.info("Total Out-of-Sample Games: %d", cumulative_metrics["n_games"])
    logger.info("Cumulative Win Accuracy: %.2f%%", cumulative_metrics["accuracy"] * 100)
    if cumulative_metrics.get("ats_accuracy") is not None:
        logger.info("Cumulative Overall ATS Accuracy: %.2f%%", cumulative_metrics["ats_accuracy"] * 100)
    if cumulative_metrics.get("mae_margin") is not None:
        logger.info("Cumulative Margin MAE: %.2f pts", cumulative_metrics["mae_margin"])
    logger.info("Cumulative Log Loss: %.4f", cumulative_metrics["log_loss"])
    logger.info("Cumulative Brier Score: %.4f", cumulative_metrics["brier_score"])

    logger.info("--------------------------------------------------")
    logger.info("SELECTIVE BETTING TIERS & KELLY BANKROLL (Start: 100u):")
    for tier_name, t in betting_tier_results.items():
        logger.info("  * %-16s: %3d bets | Record: %-8s | ATS: %5.1f%% | Net: %+6.2fu | ROI: %+5.1f%% | Bankroll: %6.2fu (%+5.1f%%)",
                    tier_name, t["bets_placed"], t["record"], t["ats_win_rate"] * 100,
                    t["flat_units_net"], t["flat_roi_pct"],
                    t["simulated_final_bankroll"], t["simulated_bankroll_growth_pct"])
    logger.info("--------------------------------------------------")
    logger.info("Summary saved to: %s", summary_path)
    logger.info("==================================================")

    return summary

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 walk-forward backtest simulation.")
    parser.add_argument("--season", type=int, default=2025, help="Target season to backtest (default: 2025).")
    parser.add_argument("--start-week", type=int, default=1, help="Starting week (default: 1).")
    parser.add_argument("--end-week", type=int, default=None, help="Ending week (default: full season).")
    parser.add_argument("--retrain-start-week", type=int, default=4, help="Week to begin retraining per Decision #4 (default: 4).")
    parser.add_argument("--no-write-db", action="store_true", help="Do not write predictions to ml.predictions table.")
    parser.add_argument("--save-models", action="store_true", help="Persist trained champion model artifacts to models/ directory.")

    args = parser.parse_args()
    run_walk_forward_backtest(
        season=args.season,
        start_week=args.start_week,
        end_week=args.end_week,
        retrain_start_week=args.retrain_start_week,
        write_db=not args.no_write_db,
        save_models=args.save_models,
    )


if __name__ == "__main__":
    main()
