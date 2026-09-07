"""
metadata.pipeline_runs bookkeeping (design doc section 31), shared by
every ingestion entrypoint (run_ingestion.py, run_pbp_ingestion.py) so
each pipeline's runs are tracked the same way -- start a row, run the
pipeline, mark it success/failed on exit.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import text

from src.db import get_engine, qualified_table


def start_run(pipeline_name: str) -> int:
    engine = get_engine()
    table = qualified_table("metadata", "pipeline_runs")
    is_postgres = engine.dialect.name != "sqlite"
    returning = " RETURNING run_id" if is_postgres else ""

    with engine.begin() as conn:
        result = conn.execute(
            text(f"INSERT INTO {table} (pipeline_name, started_at, status) "
                 f"VALUES (:name, :started_at, 'running'){returning}"),
            {"name": pipeline_name, "started_at": datetime.now(timezone.utc).isoformat()},
        )
        if is_postgres:
            run_id = result.scalar_one()
        else:
            run_id = result.lastrowid
    return run_id


def finish_run(run_id: int, status: str, records_processed: int, error_message: str | None = None) -> None:
    engine = get_engine()
    table = qualified_table("metadata", "pipeline_runs")
    with engine.begin() as conn:
        conn.execute(
            text(f"UPDATE {table} SET completed_at = :completed_at, status = :status, "
                 f"records_processed = :records_processed, error_message = :error_message "
                 f"WHERE run_id = :run_id"),
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "records_processed": records_processed,
                "error_message": error_message,
                "run_id": run_id,
            },
        )
