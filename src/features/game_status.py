"""
Shared "is this game final" predicate.

Used by both src/features/elo.py and src/transform/gold_transform.py.
Point-in-time correctness (design doc section 19) depends on both
agreeing on exactly which games count as "played" -- keeping this in
one place means that definition can't silently drift between the two
call sites.
"""

from __future__ import annotations

import pandas as pd


def is_final_game(row) -> bool:
    return row.get("status") == "final" and pd.notna(row.get("home_score")) and pd.notna(row.get("away_score"))
