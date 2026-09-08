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
    """True when Phase 4 has landed a predictions table in this database.

    Checked per-request rather than cached at import: the API may well be
    running while the Phase 4 agent creates the table, and a cached False
    would require a restart to notice.
    """
    engine = get_engine()
    return inspect(engine).has_table(_table_name())


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
