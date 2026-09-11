"""
Tests for Phase 10: Multi-Model Ensembling & Blending Engine.
"""

import numpy as np
import pandas as pd
from src.ml.ensemble import EnsembleNFLModel, train_ensemble_model
from src.ml.features import FEATURE_COLUMNS


def test_ensemble_model_training_and_inference():
    np.random.seed(42)
    n_samples = 60
    X = pd.DataFrame(np.random.randn(n_samples, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
    y_win = (X["elo_difference"] > 0).astype(int)
    y_margin = X["elo_difference"] * 0.1 + np.random.randn(n_samples) * 3
    y_total = 44.0 + np.random.randn(n_samples) * 4

    ensemble = train_ensemble_model(
        X_train=X,
        y_win_train=y_win,
        y_margin_train=y_margin,
        y_total_train=y_total,
    )

    assert isinstance(ensemble, EnsembleNFLModel)

    # Test win probabilities
    probs = ensemble.predict_proba(X)
    assert probs.shape == (n_samples, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(probs[:, 0] + probs[:, 1], 1.0, atol=1e-5)

    # Test margins
    margins = ensemble.predict_margin(X)
    assert len(margins) == n_samples

    # Test scores
    home_scores, away_scores = ensemble.predict_scores(X)
    assert len(home_scores) == n_samples
    assert len(away_scores) == n_samples
    assert np.all(home_scores >= 0.0)
    assert np.all(away_scores >= 0.0)

    # Test cover probability
    market_spreads = np.array([-3.5] * n_samples)
    cover_probs = ensemble.predict_cover_proba(X, market_spreads)
    assert len(cover_probs) == n_samples
    assert np.all(cover_probs >= 0.0) and np.all(cover_probs <= 1.0)
