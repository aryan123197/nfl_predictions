-- Gold layer, Slice A (design doc sections 12, 17, 18).
--
-- Scoped to what's derivable from silver.games / silver.odds /
-- silver.injuries alone -- no play-by-play dependency. EPA-based team
-- rolling stats and player features (design doc sections 14-15) need
-- play-by-play data that isn't ingested yet -- those are Slice B, along
-- with the remaining gold.team_rolling_stats / player_rolling_stats
-- tables the design doc lists.
--
-- All gold tables here are fully rebuilt on every gold_transform run
-- (not incrementally updated) -- see src/transform/gold_transform.py
-- for why that's an acceptable V1 tradeoff.

CREATE SCHEMA IF NOT EXISTS gold;

-- ---------------------------------------------------------------------
-- gold.team_ratings -- Elo, one row per team per game (design doc §17)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gold.team_ratings (
    id       BIGSERIAL PRIMARY KEY,
    team_id  TEXT NOT NULL,
    game_id  TEXT NOT NULL REFERENCES silver.games(game_id),
    season   INTEGER NOT NULL,
    week     INTEGER,
    elo_pre  DOUBLE PRECISION NOT NULL,
    elo_post DOUBLE PRECISION NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_gold_team_ratings_team_game
    ON gold.team_ratings (team_id, game_id);

-- ---------------------------------------------------------------------
-- gold.team_game_stats -- one row per team per game (design doc §12)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gold.team_game_stats (
    id             BIGSERIAL PRIMARY KEY,
    team_id        TEXT NOT NULL,
    opponent_id    TEXT NOT NULL,
    game_id        TEXT NOT NULL REFERENCES silver.games(game_id),
    season         INTEGER NOT NULL,
    week           INTEGER,
    is_home        BOOLEAN NOT NULL,
    points_for     INTEGER,
    points_against INTEGER,
    win            BOOLEAN,  -- NULL until the game is final
    margin         INTEGER
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_gold_team_game_stats_team_game
    ON gold.team_game_stats (team_id, game_id);

-- ---------------------------------------------------------------------
-- gold.injury_impact -- one row per team per game (design doc §16)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gold.injury_impact (
    id             BIGSERIAL PRIMARY KEY,
    team_id        TEXT NOT NULL,
    game_id        TEXT NOT NULL REFERENCES silver.games(game_id),
    season         INTEGER NOT NULL,
    week           INTEGER,
    injury_impact  DOUBLE PRECISION NOT NULL,
    config_version TEXT NOT NULL  -- ties each row to the injury_impact_v0.yaml version that produced it
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_gold_injury_impact_team_game
    ON gold.injury_impact (team_id, game_id);

-- ---------------------------------------------------------------------
-- gold.game_odds -- one row per game (design doc §12)
-- ---------------------------------------------------------------------
-- V1 has exactly one odds snapshot per game (nflverse's closing
-- consensus line), so opening_spread == current_spread and
-- spread_movement == 0 for every row -- these columns exist now so
-- gold.game_features doesn't need a schema change once a second,
-- time-varying odds source is added (see DECISIONS.md #2).
CREATE TABLE IF NOT EXISTS gold.game_odds (
    game_id          TEXT PRIMARY KEY REFERENCES silver.games(game_id),
    sportsbook       TEXT NOT NULL,
    source           TEXT NOT NULL,
    opening_spread   DOUBLE PRECISION,
    current_spread   DOUBLE PRECISION,
    spread_movement  DOUBLE PRECISION,
    total             DOUBLE PRECISION,
    moneyline_home   DOUBLE PRECISION,
    moneyline_away   DOUBLE PRECISION
);

-- ---------------------------------------------------------------------
-- gold.game_features -- one row per prediction opportunity (design doc §18)
-- ---------------------------------------------------------------------
-- Slice A columns only. temperature/wind are included per the design
-- doc's example feature set but stay NULL -- no weather provider exists
-- yet (see README/DECISIONS.md). EPA-based columns from the design
-- doc's example (home_off_epa, home_def_epa, home_qb_epa, ...) are
-- deliberately omitted here rather than left NULL, since they don't
-- exist as a concept until Slice B's play-by-play ingestion lands --
-- adding them then is a real schema change, not a backfill.
CREATE TABLE IF NOT EXISTS gold.game_features (
    game_id             TEXT PRIMARY KEY REFERENCES silver.games(game_id),
    season              INTEGER NOT NULL,
    week                INTEGER,
    home_team_id        TEXT NOT NULL,
    away_team_id        TEXT NOT NULL,
    home_elo            DOUBLE PRECISION,
    away_elo            DOUBLE PRECISION,
    elo_difference       DOUBLE PRECISION,
    home_injury_impact   DOUBLE PRECISION,
    away_injury_impact   DOUBLE PRECISION,
    home_rest_days       INTEGER,
    away_rest_days       INTEGER,
    home_recent_form     DOUBLE PRECISION,  -- win pct over the last N completed games strictly before this one
    away_recent_form     DOUBLE PRECISION,
    temperature          DOUBLE PRECISION,  -- not populated yet: no weather provider (see DECISIONS.md follow-ups)
    wind                 DOUBLE PRECISION,  -- not populated yet: no weather provider (see DECISIONS.md follow-ups)
    opening_spread       DOUBLE PRECISION,
    current_spread       DOUBLE PRECISION,
    spread_movement      DOUBLE PRECISION,
    generated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_gold_game_features_season_week
    ON gold.game_features (season, week);
