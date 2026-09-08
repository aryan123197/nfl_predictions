-- Phase 4: ML prediction tracking (design doc sections 21-22).
--
-- Brand new tables, so plain CREATE TABLE IF NOT EXISTS is sufficient --
-- no ALTER TABLE migration needed like the mid-flight column additions
-- in schema/002_silver.sql and schema/005_gold_rolling_stats.sql.

CREATE SCHEMA IF NOT EXISTS ml;

-- ---------------------------------------------------------------------
-- ml.game_results -- actual outcomes, one row per completed game
-- ---------------------------------------------------------------------
-- Derived mechanically from silver.games (status='final') by
-- src/transform/game_results_transform.py. Kept as its own table,
-- separate from silver.games, because it carries evaluation-specific
-- derived fields (home_team_covered) that silver.games has no reason
-- to know about -- silver.games is "what happened", ml.game_results is
-- "what happened, shaped for scoring predictions against."
CREATE TABLE IF NOT EXISTS ml.game_results (
    game_id            TEXT PRIMARY KEY REFERENCES silver.games(game_id),
    actual_home_score  INTEGER NOT NULL,
    actual_away_score  INTEGER NOT NULL,
    actual_margin      INTEGER NOT NULL,  -- home_score - away_score
    home_team_won      BOOLEAN,           -- NULL for a tie -- neither team "won" (see DECISIONS.md)
    home_team_covered  BOOLEAN,           -- actual_margin vs. gold.game_odds.current_spread -- NULL if no odds
    result_timestamp   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------
-- ml.predictions -- one row per prediction, NEVER overwritten or updated
-- ---------------------------------------------------------------------
-- Design doc section 21 is explicit: "Predictions must NEVER be
-- overwritten. Each prediction is a historical record of what the model
-- believed at a particular point in time." So this table is
-- insert-only -- no upsert/replace logic anywhere in the codebase
-- should ever DELETE or UPDATE a row here. Re-predicting the same game
-- with a newer model version adds a new row (same game_id, different
-- model_version), it doesn't touch the old one.
CREATE TABLE IF NOT EXISTS ml.predictions (
    prediction_id           BIGSERIAL PRIMARY KEY,
    game_id                 TEXT NOT NULL REFERENCES silver.games(game_id),
    model_version           TEXT NOT NULL,
    prediction_timestamp    TIMESTAMPTZ NOT NULL DEFAULT now(),

    home_win_probability    DOUBLE PRECISION NOT NULL,
    away_win_probability    DOUBLE PRECISION NOT NULL,

    predicted_home_score    DOUBLE PRECISION,  -- NULL in Slice A -- win probability only, see DECISIONS.md
    predicted_away_score    DOUBLE PRECISION,
    predicted_margin        DOUBLE PRECISION,

    market_spread           DOUBLE PRECISION,
    cover_probability       DOUBLE PRECISION,

    feature_snapshot_id     TEXT  -- ties this prediction to the exact gold.game_features row/run that produced it
);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_game
    ON ml.predictions (game_id, model_version);
