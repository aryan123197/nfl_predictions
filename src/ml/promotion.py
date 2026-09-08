"""
Candidate model promotion evaluation -- design doc section 28 & DECISIONS.md #1.

Implements the multi-metric model promotion gate for weekly retraining
in the walk-forward backtesting loop (Phase 5) and automated production
retraining (Phase 6).

A candidate model is evaluated on a validation set and promoted over the
incumbent champion ONLY if:
1. It does not regress on ANY core metric (log loss, Brier score, accuracy)
   beyond the allowed noise tolerance (guards against catastrophic regression
   or promoting on a single noisy metric).
2. It improves on AT LEAST ONE core metric compared to the champion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_TOLERANCES = {
    "log_loss": 0.015,     # Candidate log_loss cannot exceed champion + 0.015
    "brier_score": 0.010,  # Candidate brier_score cannot exceed champion + 0.010
    "accuracy": 0.020,     # Candidate accuracy cannot drop below champion - 0.020 (2%)
}


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reason: str
    delta: dict[str, float]
    champion_metrics: dict[str, Any]
    candidate_metrics: dict[str, Any]


def evaluate_candidate_promotion(
    champion_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    tolerances: dict[str, float] | None = None,
) -> PromotionDecision:
    """Evaluate whether candidate model metrics justify promoting over the champion.

    Parameters
    ----------
    champion_metrics: dict containing 'accuracy', 'log_loss', 'brier_score'.
    candidate_metrics: dict containing 'accuracy', 'log_loss', 'brier_score'.
    tolerances: optional dict of metric tolerance thresholds.

    Returns
    -------
    PromotionDecision indicating whether to promote, why, and metric deltas.
    """
    tol = dict(DEFAULT_TOLERANCES)
    if tolerances:
        tol.update(tolerances)

    delta = {
        "accuracy": candidate_metrics["accuracy"] - champion_metrics["accuracy"],
        "log_loss": candidate_metrics["log_loss"] - champion_metrics["log_loss"],
        "brier_score": candidate_metrics["brier_score"] - champion_metrics["brier_score"],
    }

    # Check 1: Regression limits (lower is better for log_loss & brier_score; higher for accuracy)
    if delta["log_loss"] > tol["log_loss"]:
        return PromotionDecision(
            promoted=False,
            reason=f"Log loss regressed by {delta['log_loss']:+.4f} (exceeds tolerance of +{tol['log_loss']:.4f})",
            delta=delta,
            champion_metrics=champion_metrics,
            candidate_metrics=candidate_metrics,
        )

    if delta["brier_score"] > tol["brier_score"]:
        return PromotionDecision(
            promoted=False,
            reason=f"Brier score regressed by {delta['brier_score']:+.4f} (exceeds tolerance of +{tol['brier_score']:.4f})",
            delta=delta,
            champion_metrics=champion_metrics,
            candidate_metrics=candidate_metrics,
        )

    if delta["accuracy"] < -tol["accuracy"]:
        return PromotionDecision(
            promoted=False,
            reason=f"Accuracy dropped by {abs(delta['accuracy']):.2%} (exceeds tolerance of {tol['accuracy']:.2%})",
            delta=delta,
            champion_metrics=champion_metrics,
            candidate_metrics=candidate_metrics,
        )

    # Check 2: At least one metric must strictly improve
    improved_log_loss = delta["log_loss"] < -1e-5
    improved_brier = delta["brier_score"] < -1e-5
    improved_accuracy = delta["accuracy"] > 1e-5

    improvements = []
    if improved_log_loss:
        improvements.append(f"log_loss {delta['log_loss']:+.4f}")
    if improved_brier:
        improvements.append(f"brier_score {delta['brier_score']:+.4f}")
    if improved_accuracy:
        improvements.append(f"accuracy {delta['accuracy']:+.2%}")

    if not improvements:
        return PromotionDecision(
            promoted=False,
            reason="Candidate failed to improve upon champion across any metric within tolerance boundaries",
            delta=delta,
            champion_metrics=champion_metrics,
            candidate_metrics=candidate_metrics,
        )

    return PromotionDecision(
        promoted=True,
        reason=f"Candidate promoted: improved on {', '.join(improvements)} without exceeding regression tolerances",
        delta=delta,
        champion_metrics=champion_metrics,
        candidate_metrics=candidate_metrics,
    )
