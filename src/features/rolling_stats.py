"""
Team rolling stats from play-by-play (design doc section 14).

Two-step computation, kept point-in-time correct throughout:

    1. compute_per_game_team_stats(plays) -- one row of raw sums/counts
       per (team_id, game_id), from that team's own offensive scrimmage
       plays (pass_attempt or rush_attempt), plus the opponent's
       offensive EPA allowed against them (defense). Raw sums, not
       rates -- rates are only computed once, after aggregating across
       a window (step 2).

    2. compute_rolling_stats(per_game, team_games) -- for every
       team-game (including still-scheduled ones, which need rolling
       stats too since gold.game_features covers upcoming predictions,
       not just completed games), sums that team's PRIOR games' raw
       sums within the chosen window, then divides once. Summing-then-
       dividing (not averaging per-game rates) keeps a 60-play game and
       a 90-play game weighted correctly relative to each other.
       Point-in-time correctness (design doc section 19): a game only
       ever enters another game's window after it has actually been
       played -- compute_per_game_team_stats only produces rows for
       games with real play data, so a scheduled/future game simply
       isn't in the history yet.

Only real scrimmage plays (pass_attempt or rush_attempt) count toward
offensive metrics -- kickoffs, punts, PATs, and timeouts are excluded,
matching the standard "offensive EPA/play" convention (nflverse assigns
EPA to those play types too, but on a different scale that isn't
comparable to a scrimmage snap).

Scope note: red_zone_td_rate is a per-play proxy (TD rate on snaps run
inside the 20), not a true per-drive red-zone efficiency (trips ending
in a TD / total red-zone trips) -- the latter needs drive-level
aggregation, deferred as a documented follow-up (see README). Player-
level features (design doc section 15) are out of scope entirely here
-- these are team aggregates only.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# design doc section 14 lists four candidate windows; scoped to two for
# V1 -- see schema/005_gold_rolling_stats.sql for the reasoning.
WINDOWS: dict[str, Optional[int]] = {"season_to_date": None, "last_4": 4}

EXPLOSIVE_PASS_YARDS = 15
EXPLOSIVE_RUSH_YARDS = 10

_KEY_COLUMNS = ["team_id", "game_id"]
# The two columns contributed by the defense-side groupby; everything else in
# _PER_GAME_COLUMNS comes from the offense side. Derived (not hand-listed)
# so the empty-DataFrame fallbacks below can't drift from the real schema
# when a metric is added.
_DEFENSE_COLUMNS = ["def_plays", "def_epa_sum"]
_PER_GAME_COLUMNS = _KEY_COLUMNS + [
    "plays", "epa_sum", "success_sum", "turnover_sum",
    "pass_plays", "pass_epa_sum", "rush_plays", "rush_epa_sum",
    "dropbacks", "sack_sum", "qb_hit_sum", "explosive_sum",
    "redzone_plays", "redzone_td_sum", "third_down_plays", "third_down_conversions",
] + _DEFENSE_COLUMNS
_OFFENSE_COLUMNS = [c for c in _PER_GAME_COLUMNS if c not in _DEFENSE_COLUMNS]


def compute_per_game_team_stats(plays: pd.DataFrame) -> pd.DataFrame:
    """One row per (team_id, game_id): raw sums/counts -- see module
    docstring for why these aren't rates yet."""
    if plays.empty:
        return pd.DataFrame(columns=_PER_GAME_COLUMNS)

    plays = plays.copy()
    for col in ("pass_attempt", "rush_attempt", "sack", "success", "touchdown",
                "interception", "fumble_lost", "qb_hit", "third_down_converted"):
        plays[col] = plays[col].fillna(False).astype(bool)

    scrimmage = plays[plays["pass_attempt"] | plays["rush_attempt"]].copy()
    # Verified against real 2025 data: nflverse sets pass_attempt=1 on
    # every sacked play (100% of 1352 sampled sacks), so `sack` here is
    # a defensive OR, not the primary signal -- kept in case that ever
    # changes, not because it's needed for current data.
    scrimmage["is_dropback"] = scrimmage["pass_attempt"] | scrimmage["sack"]
    scrimmage["is_explosive"] = (
        (scrimmage["pass_attempt"] & (scrimmage["yards_gained"] >= EXPLOSIVE_PASS_YARDS))
        | (scrimmage["rush_attempt"] & (scrimmage["yards_gained"] >= EXPLOSIVE_RUSH_YARDS))
    )
    scrimmage["is_redzone"] = scrimmage["yardline_100"] <= 20
    scrimmage["is_third_down"] = scrimmage["down"] == 3

    offense_rows = []
    for (team_id, game_id), group in scrimmage.groupby(["posteam", "game_id"]):
        dropbacks = group[group["is_dropback"]]
        pass_plays = group[group["pass_attempt"]]
        rush_plays = group[group["rush_attempt"]]
        redzone_plays = group[group["is_redzone"]]
        third_down_plays = group[group["is_third_down"]]
        offense_rows.append({
            "team_id": team_id, "game_id": game_id,
            "plays": len(group), "epa_sum": group["epa"].sum(),
            "success_sum": int(group["success"].sum()),
            "turnover_sum": int(group["interception"].sum() + group["fumble_lost"].sum()),
            "pass_plays": len(pass_plays), "pass_epa_sum": pass_plays["epa"].sum(),
            "rush_plays": len(rush_plays), "rush_epa_sum": rush_plays["epa"].sum(),
            "dropbacks": len(dropbacks), "sack_sum": int(dropbacks["sack"].sum()),
            "qb_hit_sum": int(dropbacks["qb_hit"].sum()),
            "explosive_sum": int(group["is_explosive"].sum()),
            "redzone_plays": len(redzone_plays), "redzone_td_sum": int(redzone_plays["touchdown"].sum()),
            "third_down_plays": len(third_down_plays),
            "third_down_conversions": int(third_down_plays["third_down_converted"].sum()),
        })
    offense = pd.DataFrame(offense_rows)

    defense_rows = []
    for (team_id, game_id), group in scrimmage.groupby(["defteam", "game_id"]):
        defense_rows.append({"team_id": team_id, "game_id": game_id,
                              "def_plays": len(group), "def_epa_sum": group["epa"].sum()})
    defense = pd.DataFrame(defense_rows)

    if offense.empty:
        offense = pd.DataFrame(columns=_OFFENSE_COLUMNS)
    if defense.empty:
        defense = pd.DataFrame(columns=_KEY_COLUMNS + _DEFENSE_COLUMNS)

    merged = offense.merge(defense, on=["team_id", "game_id"], how="outer")
    return merged


def _sum_history(history: list[dict], field: str) -> float:
    """Sum `field` across `history`, treating missing values as 0.

    Must skip NaN explicitly, not just rely on `or 0`: NaN is truthy in
    Python, so `g.get(field) or 0` passes it straight through, and NaN +
    anything is NaN -- one bad game silently poisons that stat for every
    later game in the window. NaN reaches here from the outer merge in
    compute_per_game_team_stats() when a team appears on only one side of
    the ball in a game (offense rows but no defense rows, or vice versa).
    Verified against the full real 2025 season (570 team-games) that this
    never happens with complete data -- a real game always has both teams
    on offense and defense -- so this guards partial ingestion, a truncated
    pbp file, or a forfeited game, not current production data.
    """
    total = 0.0
    for game in history:
        value = game.get(field)
        if value is None or not pd.notna(value):
            continue
        total += value
    return total


def _rate(numerator: float, denominator: float) -> Optional[float]:
    return (numerator / denominator) if denominator else None


def _summarize_window(team_id, game_id, season, week, window_name: str, history: list[dict]) -> dict:
    n = len(history)
    base = {"team_id": team_id, "game_id": game_id, "season": season, "week": week,
            "window": window_name, "games_included": n}
    if n == 0:
        return {**base, "epa_per_play": None, "off_epa": None, "def_epa": None, "pass_epa": None,
                "rush_epa": None, "success_rate": None, "turnover_rate": None, "sack_rate": None,
                "pressure_rate": None, "explosive_play_rate": None, "red_zone_td_rate": None,
                "third_down_rate": None}

    plays = _sum_history(history, "plays")
    pass_plays = _sum_history(history, "pass_plays")
    rush_plays = _sum_history(history, "rush_plays")
    dropbacks = _sum_history(history, "dropbacks")
    redzone_plays = _sum_history(history, "redzone_plays")
    third_down_plays = _sum_history(history, "third_down_plays")
    def_plays = _sum_history(history, "def_plays")

    epa_per_play = _rate(_sum_history(history, "epa_sum"), plays)
    return {
        **base,
        "epa_per_play": epa_per_play,
        "off_epa": epa_per_play,
        "def_epa": _rate(_sum_history(history, "def_epa_sum"), def_plays),
        "pass_epa": _rate(_sum_history(history, "pass_epa_sum"), pass_plays),
        "rush_epa": _rate(_sum_history(history, "rush_epa_sum"), rush_plays),
        "success_rate": _rate(_sum_history(history, "success_sum"), plays),
        "turnover_rate": _rate(_sum_history(history, "turnover_sum"), plays),
        "sack_rate": _rate(_sum_history(history, "sack_sum"), dropbacks),
        "pressure_rate": _rate(_sum_history(history, "qb_hit_sum"), dropbacks),
        "explosive_play_rate": _rate(_sum_history(history, "explosive_sum"), plays),
        "red_zone_td_rate": _rate(_sum_history(history, "redzone_td_sum"), redzone_plays),
        "third_down_rate": _rate(_sum_history(history, "third_down_conversions"), third_down_plays),
    }


def compute_rolling_stats(per_game: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """`team_games`: one row per (team_id, game_id, season, week) for
    EVERY game a team is scheduled for this season, final or not --
    rolling stats are produced for upcoming games too (that's the whole
    point of gold.game_features), computed from strictly-prior games
    that actually have play data in `per_game`.
    """
    if team_games.empty:
        return pd.DataFrame()

    per_game_by_key = {(r["team_id"], r["game_id"]): r for _, r in per_game.iterrows()}

    rows = []
    ordered = team_games.sort_values(["team_id", "season", "week", "game_id"], kind="stable")
    for team_id, team_rows in ordered.groupby("team_id", sort=False):
        history: list[dict] = []
        current_season = None
        for _, tg in team_rows.iterrows():
            if tg["season"] != current_season:
                history = []  # rolling stats reset at a season boundary -- no cross-season carryover
                current_season = tg["season"]

            for window_name in WINDOWS:
                window_size = WINDOWS[window_name]
                games_in_window = history if window_size is None else history[-window_size:]
                rows.append(_summarize_window(team_id, tg["game_id"], tg["season"], tg["week"],
                                               window_name, games_in_window))

            key = (team_id, tg["game_id"])
            if key in per_game_by_key:
                history.append(per_game_by_key[key])

    return pd.DataFrame(rows)
