-- Gold layer, Slice C: Quarterback rolling stats (design doc section 15)
-- and QB-level columns on gold.game_features (section 18).

-- ---------------------------------------------------------------------
-- gold.qb_rolling_stats -- one row per QB per game per window
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gold.qb_rolling_stats (
    id                    BIGSERIAL PRIMARY KEY,
    player_id             TEXT,
    team_id               TEXT NOT NULL,
    game_id               TEXT NOT NULL REFERENCES silver.games(game_id),
    season                INTEGER NOT NULL,
    week                  INTEGER,
    "window"              TEXT NOT NULL,  -- 'season_to_date' | 'last_4'
    games_included        INTEGER NOT NULL,
    dropbacks             INTEGER,
    pass_attempts         INTEGER,
    qb_epa_per_dropback   DOUBLE PRECISION,
    qb_pass_epa           DOUBLE PRECISION,
    qb_success_rate       DOUBLE PRECISION,
    qb_sack_rate          DOUBLE PRECISION,
    qb_turnover_rate      DOUBLE PRECISION,
    is_starter            BOOLEAN NOT NULL DEFAULT true
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_gold_qb_rolling_stats_team_game_window
    ON gold.qb_rolling_stats (team_id, game_id, "window");

-- ---------------------------------------------------------------------
-- gold.game_features QB feature columns
-- ---------------------------------------------------------------------
ALTER TABLE gold.game_features ADD COLUMN IF NOT EXISTS home_qb_id TEXT;
ALTER TABLE gold.game_features ADD COLUMN IF NOT EXISTS away_qb_id TEXT;
ALTER TABLE gold.game_features ADD COLUMN IF NOT EXISTS home_qb_epa DOUBLE PRECISION;
ALTER TABLE gold.game_features ADD COLUMN IF NOT EXISTS away_qb_epa DOUBLE PRECISION;
ALTER TABLE gold.game_features ADD COLUMN IF NOT EXISTS qb_epa_diff DOUBLE PRECISION;
ALTER TABLE gold.game_features ADD COLUMN IF NOT EXISTS home_qb_success_rate DOUBLE PRECISION;
ALTER TABLE gold.game_features ADD COLUMN IF NOT EXISTS away_qb_success_rate DOUBLE PRECISION;
ALTER TABLE gold.game_features ADD COLUMN IF NOT EXISTS home_qb_starter_change DOUBLE PRECISION;
ALTER TABLE gold.game_features ADD COLUMN IF NOT EXISTS away_qb_starter_change DOUBLE PRECISION;
