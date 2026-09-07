"""
Team ID normalization for the silver transform.

nflverse (like most NFL data sources) uses the team's *current*
abbreviation for old seasons in some files and the *historical*
abbreviation in others, depending on the file and the season it covers.
Franchise relocations mean the same team shows up under multiple
abbreviations across seasons. Silver should have exactly one canonical
ID per franchise so joins and rolling stats don't silently split a
team's history in two.

This is a small, explicit, versioned map -- not a general geocoding or
fuzzy-matching solution. It only needs to cover relocations that
actually appear in the seasons this pipeline ingests. Extend it if a
new alias shows up in bronze data (a KeyError-free unknown team ID is
NOT silently dropped -- see `canonical_team_id`, it passes unknown IDs
through unchanged and callers can decide whether to warn/log on rare
teams).
"""

from __future__ import annotations

# alias -> canonical ID. Canonical IDs match the abbreviation nflverse
# uses for the current team as of the design doc's target season (2026).
TEAM_ALIASES: dict[str, str] = {
    "OAK": "LV",   # Raiders: Oakland -> Las Vegas (2020)
    "SD": "LAC",   # Chargers: San Diego -> Los Angeles (2017)
    "STL": "LA",   # Rams: St. Louis -> Los Angeles (2016)
}


def canonical_team_id(raw_team_id: str | None) -> str | None:
    """Map a raw team abbreviation to its canonical silver.teams ID.

    Unknown / already-canonical IDs pass through unchanged.
    """
    if raw_team_id is None:
        return None
    return TEAM_ALIASES.get(raw_team_id, raw_team_id)
