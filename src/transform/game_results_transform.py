"""
Phase 4: silver.games -> ml.game_results (design doc section 22).

Derives actual outcomes for completed games. Full rebuild each run
(delete + re-insert), matching the gold layer's pattern -- data volume
is small (thousands of games at most) and recomputing from scratch is
simpler than maintaining incremental state that could drift from
silver.games.

Usage:
    python -m src.transform.game_results_transform
"""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db import get_engine, init_schema, qualified_table
from src.features.game_status import is_final_game

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("game_results_transform")


def _read_table(engine: Engine, schema: str, table: str) -> pd.DataFrame:
    qualified = qualified_table(schema, table)
    return pd.read_sql(text(f"SELECT * FROM {qualified}"), engine)


def _compute_results(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=["game_id", "actual_home_score", "actual_away_score",
                                      "actual_margin", "home_team_won", "home_team_covered"])

    final_games = games[games.apply(is_final_game, axis=1)]
    spread_by_game = {r["game_id"]: r["current_spread"] for _, r in odds.iterrows()} if not odds.empty else {}

    rows = []
    for _, g in final_games.iterrows():
        home_score = int(g["home_score"])
        away_score = int(g["away_score"])
        margin = home_score - away_score

        if margin > 0:
            home_won = True
        elif margin < 0:
            home_won = False
        else:
            home_won = None  # tie -- neither team "won" (same convention as gold.team_game_stats)

        # Sign convention (empirically verified in format.ts): positive spread =
        # HOME team favored. Home covers if actual_margin > spread (i.e.
        # actual_margin - spread > 0). Underdog (negative spread) gets points added
        # (e.g. actual_margin - (-3.0) = actual_margin + 3.0 > 0).
        spread = spread_by_game.get(g["game_id"])
        covered = None
        if spread is not None and pd.notna(spread):
            ats_margin = margin - spread
            if ats_margin > 0:
                covered = True
            elif ats_margin < 0:
                covered = False

        rows.append({
            "game_id": g["game_id"],
            "actual_home_score": home_score,
            "actual_away_score": away_score,
            "actual_margin": margin,
            "home_team_won": home_won,
            "home_team_covered": covered,
        })
    return pd.DataFrame(rows)


def run() -> int:
    init_schema()
    engine = get_engine()

    games = _read_table(engine, "silver", "games")
    odds = _read_table(engine, "gold", "game_odds")
    results = _compute_results(games, odds)

    table = qualified_table("ml", "game_results")
    is_postgres = engine.dialect.name != "sqlite"
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {table}"))
        if not results.empty:
            results = results.where(pd.notna(results), None)
            cols = list(results.columns)
            if is_postgres:
                from psycopg2.extras import execute_values
                cur = conn.connection.cursor()
                col_list = ", ".join(f'"{c}"' for c in cols)
                sql = f"INSERT INTO {table} ({col_list}) VALUES %s"
                data = [tuple(r[c] for c in cols) for r in results.to_dict(orient="records")]
                execute_values(cur, sql, data, page_size=2000)
            else:
                placeholders = ", ".join(f":{c}" for c in cols)
                col_list = ", ".join(cols)
                conn.execute(text(f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"),
                             results.to_dict(orient="records"))
    logger.info("Rebuilt ml.game_results: %d rows", len(results))
    return len(results)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
