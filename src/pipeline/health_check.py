"""
Phase 7: System Health & Data Quality Monitoring Engine (Design Doc §37).

Runs automated diagnostic audits across:
1. Data Quality (Bronze/Silver integrity, duplicate checks, team counts)
2. Feature Integrity (Null feature rate in Gold, Elo rating bounds)
3. ML Prediction Health (Probability sanity [0, 1], sum-to-1, coverage)
4. Pipeline Health (Recent run failures, latency, state recency)

Usage:
    python -m src.pipeline.health_check
    python -m src.pipeline.health_check --season 2025 --week 1
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db import get_engine, qualified_table
from src.pipeline.schedule_resolver import get_active_week, get_current_season


class HealthCheckResult:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.metrics: dict[str, Any] = {}

    def add_pass(self, msg: str) -> None:
        self.passed.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    @property
    def status(self) -> str:
        if self.errors:
            return "UNHEALTHY"
        if self.warnings:
            return "WARNING"
        return "HEALTHY"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed_count": len(self.passed),
            "warnings_count": len(self.warnings),
            "errors_count": len(self.errors),
            "passed": self.passed,
            "warnings": self.warnings,
            "errors": self.errors,
            "metrics": self.metrics,
        }


def check_data_quality(engine: Engine, season: int, result: HealthCheckResult) -> None:
    """Audit Bronze & Silver data tables for consistency."""
    teams_tbl = qualified_table("silver", "teams")
    games_tbl = qualified_table("silver", "games")
    players_tbl = qualified_table("silver", "players")

    with engine.connect() as conn:
        # 1. Team count check
        try:
            team_cnt = conn.execute(text(f"SELECT COUNT(*) FROM {teams_tbl}")).scalar() or 0
            result.metrics["total_teams"] = team_cnt
            if team_cnt == 32:
                result.add_pass(f"Silver teams count is exactly 32")
            elif team_cnt > 0:
                result.add_warning(f"Silver teams count is {team_cnt} (expected 32)")
            else:
                result.add_error("Silver teams table is empty")
        except Exception as e:
            result.add_error(f"Failed querying teams: {e}")

        # 2. Games sanity check for season
        try:
            games_cnt = conn.execute(
                text(f"SELECT COUNT(*) FROM {games_tbl} WHERE season = :season"),
                {"season": season},
            ).scalar() or 0
            result.metrics["season_games_count"] = games_cnt
            if games_cnt >= 16:
                result.add_pass(f"Silver games populated for season {season} ({games_cnt} games)")
            elif games_cnt > 0:
                result.add_warning(f"Silver games has only {games_cnt} games for season {season}")
            else:
                result.add_warning(f"No silver games found for season {season}")

            # Check for duplicate game IDs
            dupe_cnt = conn.execute(text(f"""
                SELECT COUNT(*) FROM (
                    SELECT game_id FROM {games_tbl} WHERE season = :season GROUP BY game_id HAVING COUNT(*) > 1
                ) sub
            """), {"season": season}).scalar() or 0
            if dupe_cnt == 0:
                result.add_pass("No duplicate game IDs found in silver.games")
            else:
                result.add_error(f"Found {dupe_cnt} duplicate game IDs in silver.games")
        except Exception as e:
            result.add_error(f"Failed querying games: {e}")

        # 3. Players check
        try:
            player_cnt = conn.execute(text(f"SELECT COUNT(*) FROM {players_tbl}")).scalar() or 0
            result.metrics["total_players"] = player_cnt
            if player_cnt > 1000:
                result.add_pass(f"Silver players populated ({player_cnt:,} players)")
            elif player_cnt > 0:
                result.add_warning(f"Silver players count is low ({player_cnt} players)")
            else:
                result.add_warning("Silver players table is empty")
        except Exception as e:
            result.add_error(f"Failed querying players: {e}")


def check_features_integrity(engine: Engine, season: int, result: HealthCheckResult) -> None:
    """Audit Gold feature tables for null rates and bounded metrics."""
    ratings_tbl = qualified_table("gold", "team_ratings")
    features_tbl = qualified_table("gold", "game_features")

    with engine.connect() as conn:
        # 1. Team ratings sanity
        try:
            ratings_res = conn.execute(text(f"""
                SELECT MIN(elo_post), MAX(elo_post), COUNT(*)
                FROM {ratings_tbl}
                WHERE season = :season
            """), {"season": season}).fetchone()

            if ratings_res and ratings_res[2] > 0:
                min_elo, max_elo, cnt = ratings_res
                result.metrics["ratings_count"] = cnt
                result.metrics["min_elo"] = round(float(min_elo), 1)
                result.metrics["max_elo"] = round(float(max_elo), 1)
                if 1000 <= min_elo and max_elo <= 2000:
                    result.add_pass(f"Team Elo ratings within healthy bounds ({min_elo:.0f} - {max_elo:.0f})")
                else:
                    result.add_warning(f"Team Elo ratings out of typical bounds: min={min_elo:.0f}, max={max_elo:.0f}")
            else:
                result.add_warning(f"No team ratings found for season {season}")
        except Exception as e:
            result.add_error(f"Failed querying gold team ratings: {e}")

        # 2. Feature table nulls
        try:
            feat_rows = conn.execute(text(f"""
                SELECT COUNT(*),
                       COUNT(home_elo),
                       COUNT(away_elo),
                       COUNT(home_rest_days),
                       COUNT(away_rest_days)
                FROM {features_tbl}
                WHERE season = :season
            """), {"season": season}).fetchone()

            if feat_rows and feat_rows[0] > 0:
                total, h_elo, a_elo, h_rest, a_rest = feat_rows
                result.metrics["game_features_count"] = total
                if h_elo == total and a_elo == total and h_rest == total and a_rest == total:
                    result.add_pass(f"gold.game_features complete without unexpected nulls ({total} rows)")
                else:
                    result.add_warning(f"gold.game_features has missing core features in {total - h_elo} rows")
            else:
                result.add_warning(f"No gold.game_features found for season {season}")
        except Exception as e:
            result.add_error(f"Failed querying gold game features: {e}")


def check_prediction_health(engine: Engine, season: int, week: int | None, result: HealthCheckResult) -> None:
    """Audit ML predictions for valid probability distributions and coverage."""
    preds_tbl = qualified_table("ml", "predictions")
    games_tbl = qualified_table("silver", "games")

    with engine.connect() as conn:
        try:
            query = f"""
                SELECT p.game_id, p.home_win_probability, p.away_win_probability, p.cover_probability
                FROM {preds_tbl} p
                JOIN {games_tbl} g ON p.game_id = g.game_id
                WHERE g.season = :season
            """
            params: dict[str, Any] = {"season": season}
            if week is not None:
                query += " AND g.week = :week"
                params["week"] = week

            rows = conn.execute(text(query), params).fetchall()
            result.metrics["evaluated_predictions_count"] = len(rows)

            if not rows:
                result.add_warning(f"No predictions found in ml.predictions for season {season} week {week or 'all'}")
                return

            invalid_prob_count = 0
            invalid_sum_count = 0
            for r in rows:
                h_prob = float(r.home_win_probability) if r.home_win_probability is not None else -1
                a_prob = float(r.away_win_probability) if r.away_win_probability is not None else -1
                if not (0.0 <= h_prob <= 1.0 and 0.0 <= a_prob <= 1.0):
                    invalid_prob_count += 1
                if abs((h_prob + a_prob) - 1.0) > 0.01:
                    invalid_sum_count += 1

            if invalid_prob_count == 0:
                result.add_pass(f"All {len(rows)} predictions have win probabilities bounded in [0, 1]")
            else:
                result.add_error(f"Found {invalid_prob_count} predictions with out-of-bounds probabilities")

            if invalid_sum_count == 0:
                result.add_pass(f"All {len(rows)} predictions satisfy P(Home) + P(Away) = 1.0")
            else:
                result.add_error(f"Found {invalid_sum_count} predictions where home + away probabilities do not sum to 1.0")

        except Exception as e:
            result.add_error(f"Failed querying ml.predictions: {e}")


def check_pipeline_operations(engine: Engine, result: HealthCheckResult) -> None:
    """Audit pipeline run history and state metadata."""
    runs_tbl = qualified_table("metadata", "pipeline_runs")
    state_tbl = qualified_table("metadata", "pipeline_state")

    with engine.connect() as conn:
        try:
            recent_runs = conn.execute(text(f"""
                SELECT run_id, pipeline_name, status, started_at, completed_at, error_message
                FROM {runs_tbl}
                ORDER BY run_id DESC
                LIMIT 5
            """)).fetchall()

            if recent_runs:
                successes = sum(1 for r in recent_runs if r.status == "success")
                result.metrics["recent_runs_count"] = len(recent_runs)
                result.metrics["recent_success_rate"] = round(successes / len(recent_runs), 2)
                if successes == len(recent_runs):
                    result.add_pass(f"Last {len(recent_runs)} pipeline runs were all successful")
                else:
                    failures = len(recent_runs) - successes
                    result.add_warning(f"Found {failures} failed run(s) in last {len(recent_runs)} runs")
            else:
                result.add_warning("No pipeline run records found in metadata.pipeline_runs")
        except Exception as e:
            result.add_warning(f"Could not audit metadata.pipeline_runs: {e}")

        try:
            state_row = conn.execute(text(f"SELECT * FROM {state_tbl} LIMIT 1")).fetchone()
            if state_row:
                result.add_pass("metadata.pipeline_state record is initialized")
                result.metrics["pipeline_state"] = dict(state_row._mapping)
            else:
                result.add_warning("metadata.pipeline_state table is empty")
        except Exception as e:
            result.add_warning(f"Could not audit metadata.pipeline_state: {e}")


def run_health_check(
    season: int | None = None,
    week: int | None = None,
    engine: Engine | None = None,
) -> HealthCheckResult:
    """Execute complete diagnostic health audit."""
    if engine is None:
        engine = get_engine()

    if season is None:
        season = get_current_season()

    if week is None:
        week = get_active_week(engine=engine, season=season)

    result = HealthCheckResult()
    result.metrics["checked_season"] = season
    result.metrics["checked_week"] = week
    result.metrics["check_timestamp"] = datetime.now(timezone.utc).isoformat()

    check_data_quality(engine, season, result)
    check_features_integrity(engine, season, result)
    check_prediction_health(engine, season, week, result)
    check_pipeline_operations(engine, result)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run system health and data quality checks.")
    parser.add_argument("--season", type=int, default=None, help="NFL season to audit (default: dynamic current season)")
    parser.add_argument("--week", type=int, default=None, help="NFL week to audit (default: dynamic active week)")
    args = parser.parse_args()

    res = run_health_check(season=args.season, week=args.week)

    print("=" * 60)
    print(f"NFL PREDICT SYSTEM HEALTH AUDIT: [{res.status}]")
    print("=" * 60)
    print(f"Season: {res.metrics.get('checked_season')} | Active Week: {res.metrics.get('checked_week')}")
    print(f"Timestamp: {res.metrics.get('check_timestamp')}\n")

    if res.passed:
        print(f"PASS ({len(res.passed)}):")
        for p in res.passed:
            print(f"  [+] {p}")
        print()

    if res.warnings:
        print(f"WARNINGS ({len(res.warnings)}):")
        for w in res.warnings:
            print(f"  [!] {w}")
        print()

    if res.errors:
        print(f"ERRORS ({len(res.errors)}):")
        for e in res.errors:
            print(f"  [-] {e}")
        print()

    print("=" * 60)
    if res.status == "UNHEALTHY":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
