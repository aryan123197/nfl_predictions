"""Convenience script: create bronze + metadata tables without ingesting data.

Usage:
    python scripts/init_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import init_schema, get_database_url  # noqa: E402

if __name__ == "__main__":
    print(f"Initializing schema at {get_database_url()}")
    init_schema()
    print("Done.")
