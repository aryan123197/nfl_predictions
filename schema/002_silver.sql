-- Silver layer: cleaned, normalized, analytics-ready data derived from
-- bronze. Unlike bronze (append-oriented), silver is idempotently
-- re-buildable -- re-running the silver transform for a season replaces
-- that season's rows rather than duplicating them, so silver always
-- reflects "latest known truth" for a given source row.
--
-- Responsibilities per design doc section 11: type normalization,
-- dedup, team ID normalization, player ID normalization, timestamp
-- normalization, null handling, validation, schema enforcement.

CREATE SCHEMA IF NOT EXISTS silver;

-- ---------------------------------------------------------------------
-- silver.teams
-- ---------------------------------------------------------------------
-- One row per *canonical* team ID. Historical relocations/renames
-- (e.g. OAK -> LV, SD -> LAC, STL -> LA) are normalized to the
-- canonical ID by the transform before rows ever reach this table --
-- see src/transform/team_aliases.py. This table is a distinct-values
-- register, not a full franchise-history table (no name/city columns
-- yet -- nflverse doesn't give us those directly, adding them is a
-- later, non-blocking enhancement).
CREATE TABLE IF NOT EXISTS silver.teams (
    team_id      TEXT PRIMARY KEY,
    first_season INTEGER,
    last_season  INTEGER,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------
-- silver.games
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.games (
    game_id       TEXT PRIMARY KEY,
    season        INTEGER NOT NULL,
    week          INTEGER,
    game_date     DATE,
    home_team_id  TEXT NOT NULL REFERENCES silver.teams(team_id),
    away_team_id  TEXT NOT NULL REFERENCES silver.teams(team_id),
    home_score    INTEGER,
    away_score    INTEGER,
    home_rest_days INTEGER,
    away_rest_days INTEGER,
    venue_id      TEXT,  -- not populated yet: nflverse schedule has no venue table (see DECISIONS.md follow-ups)
    status        TEXT NOT NULL,  -- scheduled|final
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_silver_games_season_week
    ON silver.games (season, week);

-- home_rest_days/away_rest_days were added after this table may already
-- exist in a previously-initialized database -- CREATE TABLE IF NOT
-- EXISTS above is a no-op there, so these columns are backfilled
-- explicitly. Safe to always run: a no-op once the column exists,
-- whether from a fresh CREATE TABLE or a prior run of this same ALTER.
ALTER TABLE silver.games ADD COLUMN IF NOT EXISTS home_rest_days INTEGER;
ALTER TABLE silver.games ADD COLUMN IF NOT EXISTS away_rest_days INTEGER;

-- ---------------------------------------------------------------------
-- silver.players
-- ---------------------------------------------------------------------
-- Current snapshot only (latest season seen per player_id). Historical
-- player-season rows stay in bronze.players_raw if ever needed.
CREATE TABLE IF NOT EXISTS silver.players (
    player_id   TEXT PRIMARY KEY,
    name        TEXT,
    position    TEXT,
    team_id     TEXT REFERENCES silver.teams(team_id),
    status      TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------
-- silver.injuries
-- ---------------------------------------------------------------------
-- Point-in-time by design -- one row per bronze report row, NOT
-- collapsed to "current status per player" (see README design notes:
-- collapsing here would make injury trend features impossible later).
-- bronze_id gives idempotent re-runs a stable upsert key (1:1 with the
-- bronze row it was derived from) without inventing a business key
-- that report data doesn't actually have.
CREATE TABLE IF NOT EXISTS silver.injuries (
    injury_id       BIGSERIAL PRIMARY KEY,
    bronze_id        BIGINT NOT NULL,
    player_id        TEXT,
    team_id          TEXT REFERENCES silver.teams(team_id),
    season           INTEGER,
    week             INTEGER,
    status           TEXT,
    injury_type      TEXT,
    reported_at      TIMESTAMPTZ,  -- often NULL: nflverse dropped this field for 2025 (see README design notes) -- (season, week) is the point-in-time fallback, see gold_transform.py
    expected_return  TIMESTAMPTZ,  -- not populated: nflverse injury reports don't carry this (see DECISIONS.md follow-ups)
    source           TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_silver_injuries_bronze_id
    ON silver.injuries (bronze_id);
CREATE INDEX IF NOT EXISTS idx_silver_injuries_player
    ON silver.injuries (player_id);

-- season/week were added after this table may already exist -- see the
-- same note on silver.games above.
ALTER TABLE silver.injuries ADD COLUMN IF NOT EXISTS season INTEGER;
ALTER TABLE silver.injuries ADD COLUMN IF NOT EXISTS week INTEGER;

-- ---------------------------------------------------------------------
-- silver.odds
-- ---------------------------------------------------------------------
-- V1 has exactly one derived source (nflverse consensus closing line,
-- embedded in the schedule file). `source` + `sportsbook` are carried
-- from day one per DECISIONS.md #2, so adding a second real odds
-- provider later is additive to this table, not a schema migration.
CREATE TABLE IF NOT EXISTS silver.odds (
    odds_id          BIGSERIAL PRIMARY KEY,
    game_id          TEXT NOT NULL REFERENCES silver.games(game_id),
    sportsbook       TEXT NOT NULL,
    source           TEXT NOT NULL,
    timestamp        TIMESTAMPTZ,
    spread           DOUBLE PRECISION,
    moneyline_home   DOUBLE PRECISION,
    moneyline_away   DOUBLE PRECISION,
    total            DOUBLE PRECISION
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_silver_odds_game_sportsbook_source
    ON silver.odds (game_id, sportsbook, source);
