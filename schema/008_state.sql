-- Phase 7: State tracking table for automated live season execution

CREATE TABLE IF NOT EXISTS metadata.pipeline_state (
    pipeline_name TEXT PRIMARY KEY,
    current_season INTEGER,
    current_week INTEGER,
    last_successful_run_id INTEGER REFERENCES metadata.pipeline_runs(run_id),
    last_run_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'idle',
    metadata_json TEXT
);

ALTER TABLE metadata.pipeline_state ADD COLUMN IF NOT EXISTS current_season INTEGER;
ALTER TABLE metadata.pipeline_state ADD COLUMN IF NOT EXISTS current_week INTEGER;
ALTER TABLE metadata.pipeline_state ADD COLUMN IF NOT EXISTS last_successful_run_id INTEGER REFERENCES metadata.pipeline_runs(run_id);
ALTER TABLE metadata.pipeline_state ADD COLUMN IF NOT EXISTS last_run_at TIMESTAMPTZ;
ALTER TABLE metadata.pipeline_state ADD COLUMN IF NOT EXISTS metadata_json TEXT;
