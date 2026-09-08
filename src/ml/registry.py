"""
Model registry and champion prediction serving interface.

Manages loading, registering, and serving predictions from the active production
champion model (design doc sections 21, 27, 28).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db import get_engine, init_schema, qualified_table
from src.ml.features import FEATURE_COLUMNS, split_features_target
from src.ml.train import (
    NFLPredictionModel,
    XGB_CLASSIFIER_PARAMS,
    XGB_REGRESSOR_PARAMS,
    _git_commit,
    save_model,
)

logger = logging.getLogger("ml_registry")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
CHAMPION_DIR = MODELS_DIR / "champion"


def get_champion_dir() -> Path:
    """Return the base directory for the active champion model."""
    return CHAMPION_DIR


def get_champion_model_path() -> Path | None:
    """Return path to model.joblib in CHAMPION_DIR if it exists, otherwise None."""
    joblib_path = CHAMPION_DIR / "model.joblib"
    if joblib_path.exists():
        return joblib_path
    return None


def load_champion_model() -> tuple[NFLPredictionModel | None, dict[str, Any] | None]:
    """Load the active champion NFLPredictionModel and its metadata.

    Returns (None, None) if no champion model has been registered yet.
    """
    model_path = get_champion_model_path()
    if not model_path:
        # Fallback: check if any valid model exists under MODELS_DIR
        if not MODELS_DIR.exists():
            return None, None
        candidate_dirs = [
            d for d in MODELS_DIR.iterdir()
            if d.is_dir() and d.name not in ("backtests", "champion", "__pycache__") and (d / "model.joblib").exists()
        ]
        if not candidate_dirs:
            return None, None
        # Use latest directory by timestamp/alphabetical order
        candidate_dirs.sort(key=lambda d: d.name, reverse=True)
        model_path = candidate_dirs[0] / "model.joblib"
        metadata_path = candidate_dirs[0] / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        try:
            model = joblib.load(model_path)
            logger.info("Loaded fallback model from %s", candidate_dirs[0].name)
            return model, metadata
        except Exception as e:
            logger.error("Failed to load fallback model: %s", e)
            return None, None

    metadata_path = CHAMPION_DIR / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    try:
        model = joblib.load(model_path)
        logger.info("Loaded champion model %s", metadata.get("model_version", "unknown"))
        return model, metadata
    except Exception as e:
        logger.error("Failed to load champion model from %s: %s", model_path, e)
        return None, None


def register_champion(
    model: NFLPredictionModel,
    metrics: dict[str, Any],
    model_version: str,
    training_metadata: dict[str, Any] | None = None,
) -> Path:
    """Register and promote a model as the active production champion.

    Saves the model to its versioned directory (models/<model_version>)
    and syncs the artifact and metadata to models/champion/.
    """
    # 1. Save versioned copy
    version_dir = MODELS_DIR / model_version
    version_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, version_dir / "model.joblib")

    meta_version = {
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "feature_columns": FEATURE_COLUMNS,
        "target": ["home_team_won", "actual_margin", "actual_total"],
        "split": training_metadata.get("split", "production_retrain") if training_metadata else "production_retrain",
        "hyperparameters": {
            "classifier": XGB_CLASSIFIER_PARAMS,
            "regressor": XGB_REGRESSOR_PARAMS,
        },
        "margin_std": model.margin_std,
        "metrics": metrics,
    }
    with open(version_dir / "metadata.json", "w") as f:
        json.dump(meta_version, f, indent=2)

    # 2. Update champion directory
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CHAMPION_DIR / "model.joblib")

    meta = {
        "model_version": model_version,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "training_metadata": training_metadata or {},
    }
    with open(CHAMPION_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Successfully registered and promoted champion model: %s", model_version)
    return version_dir


def predict_upcoming_games(
    engine: Engine | None = None,
    season: int | None = None,
    week: int | None = None,
    model: NFLPredictionModel | None = None,
    model_version: str | None = None,
) -> int:
    """Generate and write predictions for games in gold.game_features.

    If season and/or week are specified, restricts predictions to those games.
    Otherwise predicts all scheduled/unplayed games without current predictions.
    """
    if engine is None:
        engine = get_engine()
    init_schema()

    if model is None:
        champ_model, metadata = load_champion_model()
        if champ_model is None:
            logger.warning("No champion model found to generate predictions.")
            return 0
        model = champ_model
        if model_version is None and metadata:
            model_version = metadata.get("model_version")

    if model_version is None:
        model_version = f"prod_champion_{datetime.now(timezone.utc):%Y%m%d}"

    gf_table = qualified_table("gold", "game_features")
    sg_table = qualified_table("silver", "games")

    filters = []
    params: dict[str, Any] = {}
    if season is not None:
        filters.append("gf.season = :season")
        params["season"] = season
    if week is not None:
        filters.append("gf.week = :week")
        params["week"] = week

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    query = f"""
    SELECT
        gf.*,
        sg.home_score AS actual_home_score,
        sg.away_score AS actual_away_score,
        sg.status AS game_status
    FROM {gf_table} gf
    JOIN {sg_table} sg ON gf.game_id = sg.game_id
    {where_clause}
    ORDER BY gf.season, gf.week, gf.game_id
    """

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    if df.empty:
        logger.info("No matchups found in gold.game_features matching filters (season=%s, week=%s)", season, week)
        return 0

    # Extract features
    X = pd.DataFrame(index=df.index)
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = np.nan
    X = X.astype(float)

    probs = model.predict_proba(X)[:, 1]
    home_scores, away_scores = model.predict_scores(X)
    margins = model.predict_margin(X)
    market_spreads = df["current_spread"].astype(float)
    cover_probs = model.predict_cover_proba(X, market_spreads)

    pred_table = qualified_table("ml", "predictions")
    pred_timestamp = datetime.now(timezone.utc).isoformat()
    rows = []
    for i, (_, g) in enumerate(df.iterrows()):
        m_spread = float(g["current_spread"]) if pd.notna(g["current_spread"]) else None
        c_prob = float(cover_probs[i]) if m_spread is not None else None
        rows.append({
            "game_id": g["game_id"],
            "model_version": model_version,
            "prediction_timestamp": pred_timestamp,
            "home_win_probability": float(probs[i]),
            "away_win_probability": float(1.0 - probs[i]),
            "predicted_home_score": float(round(home_scores[i], 1)),
            "predicted_away_score": float(round(away_scores[i], 1)),
            "predicted_margin": float(round(margins[i], 1)),
            "market_spread": m_spread,
            "cover_probability": float(round(c_prob, 4)) if c_prob is not None else None,
            "feature_snapshot_id": model_version,
        })

    with engine.begin() as conn:
        cols = list(rows[0].keys())
        if engine.dialect.name != "sqlite":
            from psycopg2.extras import execute_values
            raw_conn = conn.connection.dbapi_connection
            cur = raw_conn.cursor()
            col_list = ", ".join(f'"{c}"' for c in cols)
            sql = f"INSERT INTO {pred_table} ({col_list}) VALUES %s"
            data = [tuple(r[c] for c in cols) for r in rows]
            execute_values(cur, sql, data, page_size=2000)
        else:
            placeholders = ", ".join(f":{c}" for c in cols)
            col_list = ", ".join(cols)
            conn.execute(text(f"INSERT INTO {pred_table} ({col_list}) VALUES ({placeholders})"), rows)

    logger.info("Wrote %d predictions to ml.predictions under version %s", len(rows), model_version)
    return len(rows)
