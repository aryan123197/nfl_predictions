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

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
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


def to_sql_target(schema: str, table: str) -> dict:
    """Return the {name, schema} kwargs for DataFrame.to_sql on the active
    dialect -- SQLite has no schema support (flattened table names, no
    `schema=` kwarg), Postgres uses a real schema. Use for bulk
    insert/replace of large tables (e.g. play-by-play) where a per-row
    upsert loop doesn't scale; qualified_table() remains what raw-SQL
    (SELECT/UPDATE/DELETE) statements should use.
    """
    engine = get_engine()
    if engine.dialect.name == "sqlite":
        return {"name": f"{schema}_{table}", "schema": None}
    return {"name": table, "schema": schema}


SCHEMA_FILES = ["001_bronze.sql", "002_silver.sql", "003_gold.sql",
                "004_silver_plays.sql", "005_gold_rolling_stats.sql", "006_ml.sql",
                "007_gold_player_stats.sql"]


def init_schema(force: bool = False) -> None:
    """Create all bronze + silver + metadata tables for the active database.

    For Postgres, this executes the schema/*.sql files directly, in
    order. For SQLite (local/dev/test only) it applies a translated
    version with schemas flattened and Postgres-only types swapped out,
    so `python -m src.ingest.run_ingestion` works with zero setup.
    """
    engine = get_engine()
    if not force and engine.dialect.name != "sqlite":
        try:
            with engine.connect() as conn:
                res = conn.execute(
                    text("SELECT 1 FROM information_schema.tables WHERE table_schema = 'ml' AND table_name = 'predictions'")
                ).fetchone()
                if res:
                    return
        except Exception:
            pass

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
            if not statement:
                continue
            try:
                conn.execute(text(statement))
            except OperationalError as exc:
                # SQLite's ALTER TABLE ADD COLUMN has no IF NOT EXISTS
                # clause (Postgres does) -- _translate_ddl_for_sqlite
                # strips it, so re-running an already-applied ADD COLUMN
                # against SQLite hits "duplicate column name" instead of
                # silently no-op'ing like it does on Postgres. Swallow
                # exactly that, to keep re-running init_schema() idempotent
                # on both dialects.
                if engine.dialect.name == "sqlite" and "duplicate column name" in str(exc).lower():
                    continue
                raise


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
    out = out.replace("CREATE SCHEMA IF NOT EXISTS gold;", "")
    out = out.replace("CREATE SCHEMA IF NOT EXISTS ml;", "")
    out = out.replace("bronze.", "bronze_")
    out = out.replace("metadata.", "metadata_")
    out = out.replace("silver.", "silver_")
    out = out.replace("gold.", "gold_")
    out = out.replace("ml.", "ml_")
    out = out.replace("BIGSERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
    out = out.replace("TIMESTAMPTZ", "TEXT")
    out = out.replace("DATE", "TEXT")
    out = out.replace("now()", "CURRENT_TIMESTAMP")
    out = out.replace("DOUBLE PRECISION", "REAL")
    # SQLite's ALTER TABLE ADD COLUMN has no IF NOT EXISTS clause (unlike
    # Postgres) -- stripped here, idempotency on rerun is instead handled
    # in apply_ddl_file() by swallowing the resulting "duplicate column
    # name" error.
    out = out.replace("ADD COLUMN IF NOT EXISTS", "ADD COLUMN")
    return out
