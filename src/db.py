"""
Database connection + schema-aware helpers.

Production runs on PostgreSQL (per the design doc). For zero-setup
local development and for the test suite, DATABASE_URL can point at
SQLite instead -- SQLite doesn't support Postgres schemas
(`bronze.games_raw`), so this module transparently flattens
`schema.table` into `schema_table` when the dialect is sqlite. Nothing
above this module (providers, ingestion scripts) needs to know or
care which database it's talking to.

This is exactly the "replaceable infrastructure" principle from the
design doc in miniature: swap DATABASE_URL, not code.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

DEFAULT_SQLITE_URL = "sqlite:///./nfl_predict.db"


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_SQLITE_URL)


def make_engine(database_url: str | None = None) -> Engine:
    url = database_url or get_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, future=True)


_engine: Engine | None = None
_SessionLocal = None


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        _engine = make_engine()
        _SessionLocal = sessionmaker(bind=_engine, future=True)
    return _engine


@contextmanager
def session_scope() -> Iterator:
    global _SessionLocal
    get_engine()  # ensure initialized
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def qualified_table(schema: str, table: str) -> str:
    """Return the correct table reference for the active dialect.

    Postgres: 'bronze.games_raw'
    SQLite:   'bronze_games_raw' (SQLite has no schema support)
    """
    engine = get_engine()
    if engine.dialect.name == "sqlite":
        return f"{schema}_{table}"
    return f"{schema}.{table}"


SCHEMA_FILES = ["001_bronze.sql", "002_silver.sql"]


def init_schema() -> None:
    """Create all bronze + silver + metadata tables for the active database.

    For Postgres, this executes the schema/*.sql files directly, in
    order. For SQLite (local/dev/test only) it applies a translated
    version with schemas flattened and Postgres-only types swapped out,
    so `python -m src.ingest.run_ingestion` works with zero setup.
    """
    for filename in SCHEMA_FILES:
        apply_ddl_file(filename)


def apply_ddl_file(filename: str) -> None:
    engine = get_engine()
    ddl_path = os.path.join(os.path.dirname(__file__), "..", "schema", filename)
    with open(ddl_path) as f:
        raw_sql = f.read()

    if engine.dialect.name == "sqlite":
        sql = _translate_ddl_for_sqlite(raw_sql)
    else:
        sql = raw_sql

    with engine.begin() as conn:
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))


def _translate_ddl_for_sqlite(sql: str) -> str:
    """Best-effort translation of the Postgres DDL to SQLite for local dev.

    This is intentionally narrow -- it's a convenience for running the
    pipeline on a laptop with no setup, not a general Postgres->SQLite
    migrator. Production always uses the real Postgres DDL.
    """
    out = sql
    out = out.replace("CREATE SCHEMA IF NOT EXISTS bronze;", "")
    out = out.replace("CREATE SCHEMA IF NOT EXISTS metadata;", "")
    out = out.replace("CREATE SCHEMA IF NOT EXISTS silver;", "")
    out = out.replace("bronze.", "bronze_")
    out = out.replace("metadata.", "metadata_")
    out = out.replace("silver.", "silver_")
    out = out.replace("BIGSERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
    out = out.replace("TIMESTAMPTZ", "TEXT")
    out = out.replace("DATE", "TEXT")
    out = out.replace("now()", "CURRENT_TIMESTAMP")
    out = out.replace("DOUBLE PRECISION", "REAL")
    return out
