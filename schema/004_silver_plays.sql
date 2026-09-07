-- Silver layer, Slice B addition: play-by-play (design doc section 11).
--
-- Brand new table (bronze.plays_raw existed as an unpopulated stub since
-- Phase 1 -- this is the first time it's actually used), so a plain
-- CREATE TABLE IF NOT EXISTS is sufficient here -- no ALTER TABLE
-- migration dance needed like schema/002_silver.sql's mid-flight column
-- additions.
--
-- Like bronze.plays_raw, this is rebuilt via bulk delete-then-insert per
-- season (src/transform/plays_transform.py), not the per-row upsert loop
-- silver_transform.py uses for games/injuries/players/odds -- a season
-- is ~45k+ play rows, and a per-row SELECT-then-branch loop doesn't
-- scale to that volume.

CREATE TABLE IF NOT EXISTS silver.plays (
    play_id                 TEXT,
    game_id                 TEXT NOT NULL REFERENCES silver.games(game_id),
    season                  INTEGER NOT NULL,
    week                    INTEGER,
    qtr                     INTEGER,
    down                    INTEGER,
    ydstogo                 INTEGER,
    yardline_100            INTEGER,
    posteam                 TEXT,  -- canonicalized via src/transform/team_aliases.py, same as silver.games
    defteam                 TEXT,
    play_type               TEXT,
    yards_gained            DOUBLE PRECISION,
    epa                     DOUBLE PRECISION,
    success                 BOOLEAN,
    pass_attempt            BOOLEAN,
    rush_attempt            BOOLEAN,
    interception            BOOLEAN,
    fumble_lost             BOOLEAN,
    touchdown               BOOLEAN,
    sack                    BOOLEAN,
    qb_hit                  BOOLEAN,
    first_down              BOOLEAN,
    third_down_converted    BOOLEAN,
    third_down_failed       BOOLEAN,
    passer_player_id        TEXT,
    rusher_player_id        TEXT,
    receiver_player_id      TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_silver_plays_game_play
    ON silver.plays (game_id, play_id);
CREATE INDEX IF NOT EXISTS idx_silver_plays_posteam_season_week
    ON silver.plays (posteam, season, week);
CREATE INDEX IF NOT EXISTS idx_silver_plays_defteam_season_week
    ON silver.plays (defteam, season, week);
