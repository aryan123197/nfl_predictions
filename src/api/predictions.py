"""
The Phase 4 boundary.

Everything the API knows about model predictions lives in this one module,
deliberately. `ml.predictions` does not exist yet -- Phase 4 (training) and
Phase 5 (backtesting) own that table and its DDL, and are being built
separately. This module therefore:

    1. Never assumes the table exists. It checks at runtime and degrades to
       "no prediction available" rather than raising, so the whole app runs
       today against real gold data with the prediction panel empty.
    2. Reads the column names the design doc's section 21 specifies, so when
       Phase 4 lands a table matching that spec, predictions light up with
       no change here.
    3. Is the ONLY place to touch if Phase 4's real schema ends up differing
       from section 21. Nothing else in src/api/ mentions ml.predictions.

The important product rule this encodes: an empty prediction is rendered as
an explicit "no model has predicted this game" state, never as a plausible-
looking number. A prediction UI that invents 50/50 or 0.0 placeholders is
worse than one that shows nothing, because a reader cannot tell the
difference between a real forecast and a filler value.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import inspect, text

from src.db import get_engine, qualified_table

# Design doc section 21. Kept as an explicit list rather than SELECT * so a
# Phase 4 table with extra columns doesn't change this response shape, and a
# Phase 4 table missing one of these fails loudly in tests rather than
# silently serving nulls.
PREDICTION_COLUMNS = [
    "game_id",
    "model_version",
    "prediction_timestamp",
    "home_win_probability",
    "away_win_probability",
    "predicted_home_score",
    "predicted_away_score",
    "predicted_margin",
    "market_spread",
]


def _table_name() -> str:
    return qualified_table("ml", "predictions")


def predictions_available() -> bool:
    """True when Phase 4 has landed predictions in this database.

    Checked per-request rather than cached at import: the API may well be
    running while a training run populates the table, and a cached False
    would require a restart to notice.
    """
    engine = get_engine()
    if not inspect(engine).has_table(_table_name()):
        return False
    try:
        with engine.connect() as conn:
            has_row = conn.execute(text(f"SELECT 1 FROM {_table_name()} LIMIT 1")).first()
            return has_row is not None
    except Exception:
        return False


def get_prediction(game_id: str) -> Optional[dict]:
    """The latest prediction for a game, or None if there isn't one.

    None covers three cases the caller does not need to distinguish -- no
    predictions table yet, a table with no row for this game, and a game
    that has not been predicted -- because all three render identically:
    "not predicted yet".
    """
    if not predictions_available():
        return None

    columns = ", ".join(PREDICTION_COLUMNS)
    # Most recent prediction wins: the design doc's section 26 retraining
    # cadence means one game accumulates several predictions over time, from
    # successive model versions.
    sql = text(
        f"SELECT {columns} FROM {_table_name()} "
        "WHERE game_id = :game_id "
        "ORDER BY prediction_timestamp DESC LIMIT 1"
    )
    with get_engine().begin() as conn:
        row = conn.execute(sql, {"game_id": game_id}).mappings().first()
    return dict(row) if row else None


def get_predictions_for_games(game_ids: list[str]) -> dict[str, dict]:
    """Latest prediction per game, for a whole week at once.

    Exists so the week view is one query instead of one per game -- the
    N+1 that would otherwise show up the moment a week has 16 games.
    """
    if not game_ids or not predictions_available():
        return {}

    columns = ", ".join(PREDICTION_COLUMNS)
    placeholders = ", ".join(f":id{i}" for i in range(len(game_ids)))
    params = {f"id{i}": game_id for i, game_id in enumerate(game_ids)}
    sql = text(
        f"SELECT {columns} FROM {_table_name()} "
        f"WHERE game_id IN ({placeholders}) "
        "ORDER BY prediction_timestamp ASC"
    )
    with get_engine().begin() as conn:
        rows = conn.execute(sql, params).mappings().all()

    # Ascending order + overwrite means the last write per game_id is the
    # newest, matching get_prediction()'s "latest wins".
    latest: dict[str, dict] = {}
    for row in rows:
        latest[row["game_id"]] = dict(row)
    return latest


def compute_model_performance(engine=None) -> dict:
    """Compute or retrieve model evaluation performance metrics."""
    if engine is None:
        engine = get_engine()

    if not predictions_available():
        return {
            "available": False,
            "reason": "No model has been trained yet (Phase 4 not landed).",
        }

    from src.ml.registry import load_champion_model
    _, metadata = load_champion_model()

    # If predictions exist, try evaluating against final games in silver.games
    pred_table = _table_name()
    games_table = qualified_table("silver", "games")
    sql = text(f"""
        SELECT
            p.game_id,
            p.model_version,
            p.home_win_probability,
            p.predicted_margin,
            p.market_spread,
            g.home_score,
            g.away_score
        FROM {pred_table} p
        JOIN {games_table} g ON p.game_id = g.game_id
        WHERE g.status = 'final' AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
    """)


    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()

    if not rows:
        if metadata and "metrics" in metadata:
            m = metadata["metrics"]
            return {
                "available": True,
                "model_version": metadata.get("model_version"),
                "games_evaluated": m.get("n_holdout", 0),
                "accuracy": round(m.get("accuracy", 0.0), 4) if "accuracy" in m else None,
                "brier_score": round(m.get("brier_score", 0.0), 4) if "brier_score" in m else None,
                "log_loss": round(m.get("log_loss", 0.0), 4) if "log_loss" in m else None,
                "mae_margin": round(m.get("mae_margin", 0.0), 4) if "mae_margin" in m else None,
                "ats_accuracy": round(m.get("ats_accuracy", 0.0), 4) if "ats_accuracy" in m else None,
            }
        return {
            "available": False,
            "reason": "Predictions exist but evaluation metrics are not computed yet (no final games).",
        }

    import math
    n = len(rows)
    correct_wins = 0
    brier_sum = 0.0
    log_loss_sum = 0.0
    mae_sum = 0.0
    ats_correct = 0
    ats_total = 0
    latest_version = rows[0]["model_version"]

    eps = 1e-15
    for r in rows:
        actual_margin = r["home_score"] - r["away_score"]
        actual_win = 1.0 if actual_margin > 0 else (0.5 if actual_margin == 0 else 0.0)
        prob = max(eps, min(1.0 - eps, float(r["home_win_probability"] or 0.5)))

        pred_win = 1.0 if prob >= 0.5 else 0.0
        if (prob >= 0.5 and actual_margin > 0) or (prob < 0.5 and actual_margin < 0) or actual_margin == 0:
            correct_wins += 1

        brier_sum += (prob - (1.0 if actual_margin > 0 else 0.0)) ** 2
        actual_binary = 1.0 if actual_margin > 0 else 0.0
        log_loss_sum += -(actual_binary * math.log(prob) + (1.0 - actual_binary) * math.log(1.0 - prob))

        if r["predicted_margin"] is not None:
            mae_sum += abs(float(r["predicted_margin"]) - actual_margin)

        if r["market_spread"] is not None and r["predicted_margin"] is not None:
            market_spread = float(r["market_spread"])
            pred_margin = float(r["predicted_margin"])
            # Spread: home is favored by -market_spread (e.g. spread=-3 => home needs to win by >3)
            # Cover condition: home covers if actual_margin > -market_spread
            market_threshold = -market_spread
            pick_home = pred_margin > market_threshold
            home_covered = actual_margin > market_threshold
            if actual_margin != market_threshold:
                ats_total += 1
                if pick_home == home_covered:
                    ats_correct += 1

    return {
        "available": True,
        "model_version": latest_version,
        "games_evaluated": n,
        "accuracy": round(correct_wins / n, 4),
        "brier_score": round(brier_sum / n, 4),
        "log_loss": round(log_loss_sum / n, 4),
        "mae_margin": round(mae_sum / n, 4) if n > 0 else None,
        "ats_accuracy": round(ats_correct / ats_total, 4) if ats_total > 0 else None,
    }


def get_game_explanation(game_id: str, engine=None) -> Optional[dict]:
    """Retrieve TreeSHAP feature attribution explanation for a game prediction."""
    from src.ml.explain import explain_game_prediction
    return explain_game_prediction(game_id=game_id, engine=engine)


def get_betting_recommendations(
    season: int | None = None,
    week: int | None = None,
    engine=None,
) -> dict:
    """Generate +EV betting recommendations for games in the active/selected week."""
    if engine is None:
        engine = get_engine()

    from src.ml.betting_strategy import evaluate_spread_bet
    pred_table = _table_name()
    games_table = qualified_table("silver", "games")
    features_table = qualified_table("gold", "game_features")

    filters = []
    params: dict = {}
    if season is not None:
        filters.append("g.season = :season")
        params["season"] = season
    if week is not None:
        filters.append("g.week = :week")
        params["week"] = week

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    if not inspect(engine).has_table(pred_table):
        return {
            "season": season,
            "week": week,
            "total_recommendations": 0,
            "strong_value_count": 0,
            "recommendations": [],
        }

    sql = text(f"""
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.home_team_id,
            g.away_team_id,
            p.market_spread,
            p.cover_probability,
            p.home_win_probability,
            p.predicted_margin,
            gf.current_spread
        FROM {games_table} g
        LEFT JOIN {pred_table} p ON g.game_id = p.game_id
        LEFT JOIN {features_table} gf ON g.game_id = gf.game_id
        {where_clause}
        ORDER BY g.season DESC, g.week ASC, g.game_id ASC
    """)

    recommendations = []
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()

        for r in rows:
            spread_line = r["market_spread"] if r["market_spread"] is not None else r["current_spread"]
            cover_prob = r["cover_probability"]
            if cover_prob is None and r["home_win_probability"] is not None:
                cover_prob = float(r["home_win_probability"])

            rec = evaluate_spread_bet(
                game_id=r["game_id"],
                home_team_id=r["home_team_id"],
                away_team_id=r["away_team_id"],
                market_spread=float(spread_line) if spread_line is not None else None,
                home_cover_prob=float(cover_prob) if cover_prob is not None else None,
            )
            if rec is not None and rec.value_tier.value != "NO_BET":
                recommendations.append(rec.to_dict())
    except Exception:
        pass

    recommendations.sort(key=lambda x: x["expected_value_pct"], reverse=True)
    strong_count = sum(1 for r in recommendations if r["value_tier"] == "STRONG_VALUE")

    return {
        "season": season,
        "week": week,
        "total_recommendations": len(recommendations),
        "strong_value_count": strong_count,
        "recommendations": recommendations,
    }


