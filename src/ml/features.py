"""
Phase 4: gold.game_features + ml.game_results -> a training frame.

FEATURE_COLUMNS is the explicit, versioned list of columns fed to the
model -- deliberately not "every numeric column in gold.game_features",
so the feature set a model was trained on is always traceable (design
doc section 27: "Training data version, Feature version" are tracked
fields). Identifier columns (game_id, season, week, team IDs) are kept
alongside the features for splitting/joining, but excluded from X.

Missing values (NaN) are left as-is, not imputed -- XGBoost handles
missing values natively (it learns a default split direction per node),
which is one of the reasons XGBoost was chosen as the baseline (see
DECISIONS.md): a large fraction of real rows have NaN off_epa/def_epa
(a team's first ~1-3 games of a season, before rolling stats exist) and
imputing those with e.g. 0 would assert "league-average," which is a
much stronger and more arbitrary claim than "unknown."
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db import qualified_table

FEATURE_COLUMNS = [
    "home_elo", "away_elo", "elo_difference",
    "home_injury_impact", "away_injury_impact",
    "home_rest_days", "away_rest_days",
    "home_recent_form", "away_recent_form",
    "home_off_epa", "away_off_epa", "home_def_epa", "away_def_epa",
    "home_pass_epa", "away_pass_epa",
    "home_rush_epa", "away_rush_epa",
    "home_turnover_rate", "away_turnover_rate",
    "home_pressure_rate", "away_pressure_rate",
    "home_explosive_play_rate", "away_explosive_play_rate",
    "home_third_down_rate", "away_third_down_rate",
    "home_success_rate", "away_success_rate",
    "home_qb_epa", "away_qb_epa", "qb_epa_diff",
    "home_qb_success_rate", "away_qb_success_rate",
    "home_qb_starter_change", "away_qb_starter_change",
    "opening_spread", "current_spread", "spread_movement",
]

ID_COLUMNS = ["game_id", "season", "week", "home_team_id", "away_team_id"]

TARGET_COLUMN = "home_team_won"
TARGET_COLUMNS = [
    "home_team_won",
    "actual_home_score",
    "actual_away_score",
    "actual_margin",
    "home_team_covered",
]


def load_training_frame(engine: Engine) -> pd.DataFrame:
    """One row per FINAL game with a known winner: ID_COLUMNS +
    FEATURE_COLUMNS + TARGET_COLUMNS. Ties (home_team_won IS NULL) are
    dropped -- see DECISIONS.md for why a binary win-probability
    classifier doesn't have a well-defined label for a tie."""
    features_table = qualified_table("gold", "game_features")
    results_table = qualified_table("ml", "game_results")

    query = f"""
        SELECT f.*, r.home_team_won, r.actual_home_score, r.actual_away_score,
               r.actual_margin, r.home_team_covered
        FROM {features_table} f
        JOIN {results_table} r ON f.game_id = r.game_id
        WHERE r.home_team_won IS NOT NULL
    """
    df = pd.read_sql(text(query), engine)
    return df[ID_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # .astype(float) matters even though every FEATURE_COLUMNS entry is
    # numeric-or-NULL by construction: a column that's entirely NULL across
    # the whole training set (no rows with real data yet -- e.g. EPA columns
    # before any play-by-play has been ingested) comes back from pd.read_sql
    # as pandas dtype `object` holding None, not float64/NaN, since there's
    # no numeric value in the column to force the usual float coercion.
    # XGBoost rejects `object` dtype outright ("DataFrame.dtypes for data
    # must be int, float, bool or category"). Explicit casting makes this
    # correct regardless of which columns happen to be fully null.
    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(bool).astype(int)
    return X, y
