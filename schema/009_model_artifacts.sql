-- Phase 6 follow-up: persist the champion model artifact in the warehouse.
--
-- register_champion()/load_champion_model() (src/ml/registry.py) used to
-- read and write only a local models/ directory on disk. That's fine for a
-- laptop, but GitHub Actions runners are ephemeral -- models/ is gitignored
-- and every scheduled retrain/pipeline run starts from a fresh checkout, so
-- the trained champion was discarded the instant each job finished and
-- predict_upcoming_games() found no champion to serve from. Storing the
-- joblib bytes here, in the same Postgres warehouse everything else already
-- lives in, makes the champion durable across runs without adding a new
-- piece of infrastructure (S3/GCS/etc).
--
-- One row per registered version (append-only, like the models/<version>/
-- directories on disk) -- is_champion marks the single currently-active
-- one. A version's row is written once and never updated in place, so a
-- version name always refers to the same bytes: register_champion()
-- promoting a new champion clears the old flag and inserts a new row
-- rather than overwriting one.
CREATE TABLE IF NOT EXISTS metadata.model_artifacts (
    model_version   TEXT PRIMARY KEY,
    artifact        BYTEA NOT NULL,
    metadata_json   TEXT NOT NULL,
    is_champion     BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Partial unique index rather than a boolean check constraint: enforces
-- "at most one champion row" at the database level (belt-and-suspenders --
-- register_champion() also clears the old flag in the same transaction),
-- while leaving every non-champion row's is_champion=FALSE unconstrained.
CREATE UNIQUE INDEX IF NOT EXISTS model_artifacts_one_champion
    ON metadata.model_artifacts (is_champion)
    WHERE is_champion;
