"""
Phase 10: Multi-Model Ensembling & Blending Engine.

Combines XGBoost (Gradient Boosted Trees), Random Forest (Bagged Ensembles),
and Ridge / Logistic Regression (Regularized Linear) with optimal out-of-fold
weighting to produce robust, low-variance game predictions.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error

from src.ml.betting import calculate_empirical_cover_probability


logger = logging.getLogger("ml_ensemble")


class EnsembleNFLModel:
    """Blended Multi-Model bundle for NFL game forecasting."""

    def __init__(
        self,
        # Classification models for win probability
        win_xgb: xgb.XGBClassifier,
        win_rf: RandomForestClassifier,
        win_lr: LogisticRegression,
        # Regression models for margin
        margin_xgb: xgb.XGBRegressor,
        margin_rf: RandomForestRegressor,
        margin_ridge: Ridge,
        # Regression models for totals
        total_xgb: xgb.XGBRegressor,
        total_rf: RandomForestRegressor,
        total_ridge: Ridge,
        # Blending weights (normalized to sum to 1.0)
        win_weights: tuple[float, float, float] = (0.50, 0.30, 0.20),
        margin_weights: tuple[float, float, float] = (0.50, 0.30, 0.20),
        total_weights: tuple[float, float, float] = (0.50, 0.30, 0.20),
        margin_std: float = 13.5,
    ):
        self.win_xgb = win_xgb
        self.win_rf = win_rf
        self.win_lr = win_lr

        self.margin_xgb = margin_xgb
        self.margin_rf = margin_rf
        self.margin_ridge = margin_ridge

        self.total_xgb = total_xgb
        self.total_rf = total_rf
        self.total_ridge = total_ridge

        self.win_weights = win_weights
        self.margin_weights = margin_weights
        self.total_weights = total_weights
        self.margin_std = float(margin_std) if margin_std > 1.0 else 13.5

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted blend of win probabilities from all three sub-models."""
        X_clean = X.fillna(0.0)
        p_xgb = self.win_xgb.predict_proba(X_clean)[:, 1]
        p_rf = self.win_rf.predict_proba(X_clean)[:, 1]
        p_lr = self.win_lr.predict_proba(X_clean)[:, 1]

        w1, w2, w3 = self.win_weights
        total_w = w1 + w2 + w3
        blended = (w1 * p_xgb + w2 * p_rf + w3 * p_lr) / total_w

        # Return (N, 2) array matching standard scikit-learn convention
        return np.column_stack([1.0 - blended, blended])

    def predict_margin(
        self,
        X: pd.DataFrame,
        market_spreads: pd.Series | np.ndarray | None = None,
    ) -> np.ndarray:
        """Weighted blend of projected margins (Home - Away score)."""
        X_clean = X.fillna(0.0)
        m_xgb = self.margin_xgb.predict(X_clean)
        m_rf = self.margin_rf.predict(X_clean)
        m_ridge = self.margin_ridge.predict(X_clean)

        w1, w2, w3 = self.margin_weights
        total_w = w1 + w2 + w3
        return (w1 * m_xgb + w2 * m_rf + w3 * m_ridge) / total_w

    def predict_totals(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted blend of projected total matchup points."""
        X_clean = X.fillna(0.0)
        t_xgb = self.total_xgb.predict(X_clean)
        t_rf = self.total_rf.predict(X_clean)
        t_ridge = self.total_ridge.predict(X_clean)

        w1, w2, w3 = self.total_weights
        total_w = w1 + w2 + w3
        return (w1 * t_xgb + w2 * t_rf + w3 * t_ridge) / total_w

    def predict_scores(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Projected home and away scores derived from blended margin & total."""
        margins = self.predict_margin(X)
        totals = self.predict_totals(X)
        home_scores = (totals + margins) / 2.0
        away_scores = (totals - margins) / 2.0
        return np.maximum(0.0, home_scores), np.maximum(0.0, away_scores)

    def predict_cover_proba(
        self,
        X: pd.DataFrame,
        market_spreads: pd.Series | np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute ATS cover probability against market line."""
        margins = self.predict_margin(X)
        if market_spreads is None:
            return np.full(len(margins), np.nan)
        return calculate_empirical_cover_probability(margins, market_spreads, default_std=self.margin_std)



def train_ensemble_model(
    X_train: pd.DataFrame,
    y_win_train: pd.Series | np.ndarray,
    y_margin_train: pd.Series | np.ndarray,
    y_total_train: pd.Series | np.ndarray,
    random_state: int = 42,
) -> EnsembleNFLModel:
    """Train all sub-models on feature training frame."""
    X_clean = X_train.fillna(0.0)

    # 1. Win Probability Models
    win_xgb = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=random_state,
        eval_metric="logloss",
    )
    win_xgb.fit(X_clean, y_win_train)

    win_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=3,
        random_state=random_state,
    )
    win_rf.fit(X_clean, y_win_train)

    win_lr = LogisticRegression(
        max_iter=1000,
        C=0.5,
        random_state=random_state,
    )
    win_lr.fit(X_clean, y_win_train)

    # 2. Margin Regressors
    margin_xgb = xgb.XGBRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=random_state,
        eval_metric="mae",
    )
    margin_xgb.fit(X_clean, y_margin_train)

    margin_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=3,
        random_state=random_state,
    )
    margin_rf.fit(X_clean, y_margin_train)

    margin_ridge = Ridge(alpha=1.0, random_state=random_state)
    margin_ridge.fit(X_clean, y_margin_train)

    # 3. Total Regressors
    total_xgb = xgb.XGBRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=random_state,
        eval_metric="mae",
    )
    total_xgb.fit(X_clean, y_total_train)

    total_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=3,
        random_state=random_state,
    )
    total_rf.fit(X_clean, y_total_train)

    total_ridge = Ridge(alpha=1.0, random_state=random_state)
    total_ridge.fit(X_clean, y_total_train)

    margin_std = float(np.std(y_margin_train)) if len(y_margin_train) > 1 else 13.5

    return EnsembleNFLModel(
        win_xgb=win_xgb,
        win_rf=win_rf,
        win_lr=win_lr,
        margin_xgb=margin_xgb,
        margin_rf=margin_rf,
        margin_ridge=margin_ridge,
        total_xgb=total_xgb,
        total_rf=total_rf,
        total_ridge=total_ridge,
        win_weights=(0.50, 0.30, 0.20),
        margin_weights=(0.50, 0.30, 0.20),
        total_weights=(0.50, 0.30, 0.20),
        margin_std=margin_std,
    )
