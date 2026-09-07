-- Gold layer, Slice B: play-by-play-derived rolling team stats
-- (design doc section 14) and the EPA columns of gold.game_features
-- (section 18) that depend on them.

-- ---------------------------------------------------------------------
-- gold.team_rolling_stats -- one row per team per game per window
-- ---------------------------------------------------------------------
-- Design doc section 14 lists four candidate windows (season-to-date,
-- last 8/4/2 games). Scoped to two for V1: 'season_to_date' (the
-- primary signal, and what feeds gold.game_features below) and
-- 'last_4' (short-term form, kept here for future model experiments
-- but not yet wired into game_features -- see the note there).
-- Long format (one row per window) rather than one wide column per
-- window-metric combination, so adding a third window later is new
-- rows, not a schema change.
--
-- Point-in-time correctness (design doc section 19): every row is
-- computed from that team's plays in games strictly BEFORE this game
-- (never including it) -- see src/transform/plays_transform.py's
-- windowing logic and tests/test_rolling_stats.py.
CREATE TABLE IF NOT EXISTS gold.team_rolling_stats (
    id                    BIGSERIAL PRIMARY KEY,
    team_id               TEXT NOT NULL,
    game_id               TEXT NOT NULL REFERENCES silver.games(game_id),
    season                INTEGER NOT NULL,
    week                  INTEGER,
    window                TEXT NOT NULL,  -- 'season_to_date' | 'last_4'
    games_included        INTEGER NOT NULL,  -- prior games actually available (0 for a team's first game of the season)
    epa_per_play          DOUBLE PRECISION,
    off_epa               DOUBLE PRECISION,
    def_epa               DOUBLE PRECISION,
    pass_epa              DOUBLE PRECISION,
    rush_epa              DOUBLE PRECISION,
    success_rate          DOUBLE PRECISION,
    turnover_rate         DOUBLE PRECISION,
    sack_rate             DOUBLE PRECISION,
    pressure_rate         DOUBLE PRECISION,  -- qb_hit rate: nflverse has no dedicated "pressure" flag, see schema/001_bronze.sql
    explosive_play_rate   DOUBLE PRECISION,
    red_zone_td_rate      DOUBLE PRECISION,  -- per-play proxy (TD rate on snaps inside the 20), not true per-drive red-zone efficiency -- see README follow-ups
    third_down_rate       DOUBLE PRECISION
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_gold_team_rolling_stats_team_game_window
    ON gold.team_rolling_stats (team_id, game_id, window);

-- ---------------------------------------------------------------------
-- gold.game_features EPA columns (design doc section 18 example)
-- ---------------------------------------------------------------------
-- gold.game_features already exists (schema/003_gold.sql, Slice A) --
-- these are new columns on it, so ALTER TABLE ADD COLUMN IF NOT EXISTS
-- (same migration pattern as schema/002_silver.sql), not a fresh
-- CREATE TABLE. Sourced from the 'season_to_date' window only, per
-- src/transform/plays_transform.py -- 'last_4' stays in
-- gold.team_rolling_stats for now rather than doubling every column
-- here without a proven modeling need. home_qb_epa/away_qb_epa from the
-- design doc's example are deliberately not added yet -- that's a
-- player-level split (design doc section 15), out of scope until
-- Slice C's player features.
ALTER TABLE gold.game_features ADD COLUMN IF NOT EXISTS home_off_epa DOUBLE PRECISION;
ALTER TABLE gold.game_features ADD COLUMN IF NOT EXISTS away_off_epa DOUBLE PRECISION;
ALTER TABLE gold.game_features ADD COLUMN IF NOT EXISTS home_def_epa DOUBLE PRECISION;
ALTER TABLE gold.game_features ADD COLUMN IF NOT EXISTS away_def_epa DOUBLE PRECISION;
