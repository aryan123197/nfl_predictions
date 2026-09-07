-- Bronze layer: raw, append-oriented, minimally-modified source data.
-- Every table carries the common bronze metadata columns called for in
-- the design doc (source, ingested_at, source_timestamp, pipeline_run_id,
-- raw_payload_hash) so any row can be traced back to exactly which
-- pipeline run produced it and from which provider.

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS metadata;

-- ---------------------------------------------------------------------
-- metadata.pipeline_runs / pipeline_state (created first: bronze tables
-- reference pipeline_run_id)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS metadata.pipeline_runs (
    run_id             BIGSERIAL PRIMARY KEY,
    pipeline_name      TEXT NOT NULL,
    started_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at       TIMESTAMPTZ,
    status             TEXT NOT NULL DEFAULT 'running',  -- running|success|failed
    records_processed  INTEGER,
    error_message      TEXT
);

CREATE TABLE IF NOT EXISTS metadata.pipeline_state (
    pipeline_name             TEXT PRIMARY KEY,
    last_successful_run       BIGINT REFERENCES metadata.pipeline_runs(run_id),
    last_processed_timestamp  TIMESTAMPTZ,
    status                    TEXT
);

-- ---------------------------------------------------------------------
-- bronze.games_raw
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bronze.games_raw (
    id                 BIGSERIAL PRIMARY KEY,
    game_id            TEXT NOT NULL,
    season             INTEGER NOT NULL,
    week               INTEGER,
    game_type          TEXT,
    gameday            TEXT,
    gametime           TEXT,
    home_team          TEXT,
    away_team          TEXT,
    home_score         DOUBLE PRECISION,
    away_score         DOUBLE PRECISION,
    location           TEXT,
    spread_line        DOUBLE PRECISION,
    total_line         DOUBLE PRECISION,
    home_moneyline     DOUBLE PRECISION,
    away_moneyline     DOUBLE PRECISION,
    home_rest          DOUBLE PRECISION,
    away_rest          DOUBLE PRECISION,
    overtime           DOUBLE PRECISION,
    result             DOUBLE PRECISION,
    -- bronze metadata
    source             TEXT NOT NULL,
    ingested_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_timestamp   TIMESTAMPTZ,
    pipeline_run_id    BIGINT REFERENCES metadata.pipeline_runs(run_id),
    raw_payload_hash   TEXT
);

-- one row per (game_id, source, ingested batch) -- we upsert on
-- (game_id, source) so re-running ingestion updates scores/lines
-- instead of creating duplicate rows for the same game+source.
CREATE UNIQUE INDEX IF NOT EXISTS uq_games_raw_game_source
    ON bronze.games_raw (game_id, source);

CREATE INDEX IF NOT EXISTS idx_games_raw_season_week
    ON bronze.games_raw (season, week);

-- ---------------------------------------------------------------------
-- bronze.injuries_raw
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bronze.injuries_raw (
    id                       BIGSERIAL PRIMARY KEY,
    season                   INTEGER NOT NULL,
    week                     INTEGER,
    team                     TEXT,
    player_id                TEXT,
    full_name                TEXT,
    position                 TEXT,
    report_status            TEXT,
    report_primary_injury    TEXT,
    practice_status          TEXT,
    date_modified            TEXT,
    source                   TEXT NOT NULL,
    ingested_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    pipeline_run_id          BIGINT REFERENCES metadata.pipeline_runs(run_id),
    raw_payload_hash         TEXT
);

CREATE INDEX IF NOT EXISTS idx_injuries_raw_season_week
    ON bronze.injuries_raw (season, week);
CREATE INDEX IF NOT EXISTS idx_injuries_raw_player
    ON bronze.injuries_raw (player_id);

-- ---------------------------------------------------------------------
-- bronze.players_raw
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bronze.players_raw (
    id                 BIGSERIAL PRIMARY KEY,
    player_id          TEXT NOT NULL,
    full_name          TEXT,
    position           TEXT,
    team               TEXT,
    status             TEXT,
    season             INTEGER NOT NULL,
    source             TEXT NOT NULL,
    ingested_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    pipeline_run_id    BIGINT REFERENCES metadata.pipeline_runs(run_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_players_raw_player_season_source
    ON bronze.players_raw (player_id, season, source);

-- ---------------------------------------------------------------------
-- bronze.odds_raw
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bronze.odds_raw (
    id                 BIGSERIAL PRIMARY KEY,
    game_id            TEXT NOT NULL,
    sportsbook         TEXT,
    ts                 TEXT,
    spread             DOUBLE PRECISION,
    home_moneyline     DOUBLE PRECISION,
    away_moneyline     DOUBLE PRECISION,
    total              DOUBLE PRECISION,
    source             TEXT NOT NULL,
    ingested_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    pipeline_run_id    BIGINT REFERENCES metadata.pipeline_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_odds_raw_game
    ON bronze.odds_raw (game_id, ts);

-- ---------------------------------------------------------------------
-- bronze.plays_raw (created but populated starting Phase 3, since
-- play-by-play is large and not needed for the earliest game-level model)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bronze.plays_raw (
    id                       BIGSERIAL PRIMARY KEY,
    play_id                  TEXT,
    game_id                  TEXT NOT NULL,
    season                   INTEGER,
    week                     INTEGER,
    qtr                      INTEGER,
    game_seconds_remaining   DOUBLE PRECISION,
    down                     DOUBLE PRECISION,
    ydstogo                  DOUBLE PRECISION,
    yardline_100             DOUBLE PRECISION,
    posteam                  TEXT,
    defteam                  TEXT,
    play_type                TEXT,
    yards_gained             DOUBLE PRECISION,
    epa                      DOUBLE PRECISION,
    success                  DOUBLE PRECISION,
    pass_attempt             DOUBLE PRECISION,
    rush_attempt             DOUBLE PRECISION,
    interception             DOUBLE PRECISION,
    fumble_lost              DOUBLE PRECISION,
    touchdown                DOUBLE PRECISION,
    sack                     DOUBLE PRECISION,
    qb_hit                   DOUBLE PRECISION,  -- used as a pressure-rate proxy -- nflverse has no dedicated "pressure" flag
    first_down               DOUBLE PRECISION,
    third_down_converted     DOUBLE PRECISION,
    third_down_failed        DOUBLE PRECISION,
    passer_player_id         TEXT,
    rusher_player_id         TEXT,
    receiver_player_id       TEXT,
    source                   TEXT NOT NULL,
    ingested_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    pipeline_run_id          BIGINT REFERENCES metadata.pipeline_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_plays_raw_game
    ON bronze.plays_raw (game_id);

-- season/week/sack/qb_hit/first_down/third_down_converted/third_down_failed
-- were added after this table may already exist in a previously
-- initialized database -- CREATE TABLE IF NOT EXISTS above is a no-op
-- there, so these columns are backfilled explicitly (same pattern as
-- silver.games/silver.injuries in schema/002_silver.sql).
ALTER TABLE bronze.plays_raw ADD COLUMN IF NOT EXISTS season INTEGER;
ALTER TABLE bronze.plays_raw ADD COLUMN IF NOT EXISTS week INTEGER;
ALTER TABLE bronze.plays_raw ADD COLUMN IF NOT EXISTS sack DOUBLE PRECISION;
ALTER TABLE bronze.plays_raw ADD COLUMN IF NOT EXISTS qb_hit DOUBLE PRECISION;
ALTER TABLE bronze.plays_raw ADD COLUMN IF NOT EXISTS first_down DOUBLE PRECISION;
ALTER TABLE bronze.plays_raw ADD COLUMN IF NOT EXISTS third_down_converted DOUBLE PRECISION;
ALTER TABLE bronze.plays_raw ADD COLUMN IF NOT EXISTS third_down_failed DOUBLE PRECISION;

-- Idempotent-replace key: pbp ingestion deletes-then-reinserts a whole
-- season at a time (see src/ingest/run_pbp_ingestion.py) rather than
-- per-row upsert, since a season is ~45k+ rows -- a per-row SELECT-then-
-- branch loop (the pattern games_raw/injuries_raw use) doesn't scale to
-- that volume. This index exists for query performance and data-
-- integrity, not for upsert logic.
CREATE UNIQUE INDEX IF NOT EXISTS uq_plays_raw_game_play_source
    ON bronze.plays_raw (game_id, play_id, source);
