"""
Injury impact v0 -- design doc section 16, decision recorded in
DECISIONS.md #3.

Computes an estimated team injury impact score from silver.injuries,
using the versioned position/status weight table in
config/injury_impact_v0.yaml.

Point-in-time correctness (design doc section 19): callers must pass
only injury rows with reported_at strictly before the prediction
cutoff (game_date, for V1) -- this module does no filtering of its own,
by design, so the cutoff logic lives in one place (the gold transform)
rather than being silently re-implemented here.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "injury_impact_v0.yaml")


def load_weights(config_path: Optional[str] = None) -> dict:
    path = config_path or DEFAULT_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def _latest_report_per_player(injuries: pd.DataFrame) -> pd.DataFrame:
    """Collapse point-in-time injury rows to the latest report per player.

    silver.injuries is intentionally point-in-time (one row per report),
    so a team's *current* injury impact as of a cutoff needs the most
    recent report per player as of that cutoff -- not every report ever
    filed. Callers are responsible for pre-filtering to rows with
    reported_at < cutoff before calling this.
    """
    if injuries.empty:
        return injuries
    sortable = injuries.copy()
    sortable["reported_at"] = pd.to_datetime(sortable["reported_at"], errors="coerce")
    sortable = sortable.sort_values("reported_at")
    return sortable.groupby("player_id", as_index=False).last()


def compute_team_injury_impact(injuries_as_of_cutoff: pd.DataFrame, positions: pd.DataFrame,
                                weights: Optional[dict] = None) -> dict[str, float]:
    """Compute total injury impact per team from a set of injury rows.

    `injuries_as_of_cutoff` must already be filtered to reported_at <
    the desired cutoff timestamp (see module docstring) -- one row per
    (player, report). `positions` is silver.players (player_id ->
    position), used because silver.injuries' own `position`-adjacent
    field (report_primary_injury) is the injury type, not the player's
    position.

    Returns {team_id: total_impact}, where impact is negative (more
    negative = worse for the team), summed across that team's injured
    players as of the cutoff.
    """
    weights = weights or load_weights()
    position_weights = weights["position_weights"]
    default_position_weight = weights["default_position_weight"]
    status_multipliers = weights["status_multipliers"]
    default_status_multiplier = weights["default_status_multiplier"]

    if injuries_as_of_cutoff.empty:
        return {}

    latest = _latest_report_per_player(injuries_as_of_cutoff)
    latest = latest.merge(
        positions[["player_id", "position"]], on="player_id", how="left", suffixes=("", "_roster")
    )

    impact_by_team: dict[str, float] = {}
    for _, row in latest.iterrows():
        team_id = row.get("team_id")
        if team_id is None or pd.isna(team_id):
            continue
        position = row.get("position") if pd.notna(row.get("position")) else None
        status = row.get("status")

        position_weight = position_weights.get(position, default_position_weight)
        status_multiplier = status_multipliers.get(status, default_status_multiplier)
        impact_by_team[team_id] = impact_by_team.get(team_id, 0.0) + position_weight * status_multiplier

    return impact_by_team
