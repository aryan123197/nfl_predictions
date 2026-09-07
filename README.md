# NFL Predict

An end-to-end data engineering + ML platform that ingests NFL data, generates
point-in-time predictions, evaluates them, and retrains continuously across
the 2026 season. Built as a data platform first, a prediction model second.

Full design doc: see `NFL_Predict___Adaptive_NFL_Game_Prediction_Platform.md`.
This README tracks **implementation status** against that design.

---

## Status at a glance

| Phase | What | Status |
|---|---|---|
| 1 | Data ingestion (provider → bronze) | ✅ **Working** — see below |
| 2 | Bronze → Silver → Gold transforms | 🟡 **Silver working** — Gold deferred to Phase 3, see below |
| 3 | Point-in-time feature engineering | ⬜ Not started |
| 4 | ML training (XGBoost baseline) | ⬜ Not started |
| 5 | Walk-forward backtesting (2025) | ⬜ Not started |
| 6 | Automation (GitHub Actions) | ⬜ Not started |
| 7 | Live 2026 data connection | ⬜ Not started |
| 8 | MLOps (MLflow, model registry, monitoring) | ⬜ Not started |
| 9 | Application (FastAPI + React) | ⬜ Not started |
| 10 | Advanced learning (online/RL) | ⬜ Not started (explicit non-goal for V1) |

---

## What's actually built right now (Phase 1)

A working, tested ingestion pipeline that pulls real NFL data — games,
scores, closing lines, and injury reports — from the free
[nflverse-data](https://github.com/nflverse/nflverse-data) GitHub releases
and lands it in a bronze layer, with pipeline-run tracking and idempotent
upserts.

```
src/providers/base.py            NFLDataProvider abstract interface
src/providers/nflverse_provider.py   Concrete free-data implementation
src/db.py                        Postgres (prod) / SQLite (local dev) connection layer
src/ingest/run_ingestion.py      CLI entrypoint: provider → bronze
schema/001_bronze.sql            Bronze + metadata schema (Postgres DDL)
scripts/init_db.py               Create tables without ingesting
scripts/backfill_seasons.py      Bulk-ingest a range of historical seasons
tests/test_provider.py           Unit + live-network integration tests
tests/test_ingestion.py          Offline ingestion logic tests (in-memory SQLite)
```

**Verified working end-to-end** against live nflverse data: fetched real
2025 Week 1 games (scores, spread lines, moneylines), 197 injury report
rows, and 25,066 player records; confirmed games upsert correctly on
re-run (no duplicates) while injury snapshots append correctly (each
report is a point-in-time record). 13/13 tests pass, including 8 that hit
the live data source.

---

## What's actually built right now (Phase 2 — Silver only)

Bronze → Silver transform: cleans, deduplicates, and normalizes the raw
bronze rows into analytics-ready tables, idempotently re-runnable (unlike
bronze, re-running a season updates silver rows in place instead of
duplicating them).

```
schema/002_silver.sql              Silver schema (teams, games, players, injuries, odds)
src/transform/team_aliases.py      Franchise relocation map (OAK→LV, SD→LAC, STL→LA)
src/transform/silver_transform.py  CLI entrypoint: bronze → silver
tests/test_silver_transform.py     Offline transform logic tests (in-memory SQLite)
```

**Verified working end-to-end** against the same live 2025 Week 1 data used
to validate Phase 1: 32 teams, 16 games, 16 odds rows, 24,828 players, and
197 injury reports normalized into silver, with no stale team aliases
(OAK/SD/STL) surviving the transform. 16/16 offline tests pass (8 from
Phase 1 + 8 new).

**Gold layer is deferred to Phase 3.** The design doc's Gold tables
(`team_rolling_stats`, `team_ratings`, `injury_impact`, `game_features`,
etc.) depend on feature-engineering logic — rolling stats, Elo, the
injury-weight config from `DECISIONS.md` #3 — not just cleaned data, so
building them under "Phase 2" would blur data cleaning with feature
engineering. The two Gold tables that *are* mechanically derivable from
silver alone (`game_odds`, `team_game_stats`) are left for Phase 3 too, to
keep Gold as one coherent slice built alongside the features that depend
on it.

**Design decisions carried into this slice** (see `DECISIONS.md` for full
reasoning):
- `silver.injuries` stays point-in-time (one row per bronze report row,
  upserted on `bronze_id`) — never collapsed to "current status per
  player" — so injury trend features stay possible in Phase 3.
- `silver.odds` carries `source` and `sportsbook` columns from day one
  (Decision #2), even though V1 has exactly one derived odds source, so a
  second sportsbook is additive later rather than a schema migration.
- Team ID normalization (`src/transform/team_aliases.py`) is a small,
  explicit, versioned map — not general fuzzy matching — covering only
  the relocations that actually appear in ingested seasons (Raiders,
  Chargers, Rams).

### Quickstart

```bash
git clone <this-repo>
cd nfl_predict
pip install -r requirements.txt

# Zero-setup local dev: defaults to a SQLite file, no Postgres needed
python -m src.ingest.run_ingestion --season 2025 --week 1

# Or against real Postgres:
docker compose up -d
cp .env.example .env   # uncomment the local Postgres DATABASE_URL line
python -m src.ingest.run_ingestion --season 2025

# Backfill several seasons for later model training
python scripts/backfill_seasons.py --start 2020 --end 2025

# Run tests (offline only, e.g. in CI without network):
pytest -m "not integration"
# Full suite including live network calls:
pytest
```

### Design decisions worth knowing about

- **Provider abstraction pays off immediately.** nflverse changed the
  `injuries_{season}.csv` schema between 2024 and 2025 (dropped a
  `date_modified` column) — this was caught by the test suite and fixed in
  one place (`nflverse_provider.py`) without touching ingestion logic.
  This is the exact failure mode the abstraction in the design doc (§8)
  exists to contain.
- **SQLite for local dev, Postgres for production**, same code path. This
  goes a step further than the design doc's "avoid cloud infra until local
  is proven" (§5) — you don't even need Docker running to iterate on
  ingestion logic. `src/db.py` transparently flattens `schema.table` →
  `schema_table` for SQLite since it has no schema support; the real
  Postgres DDL (`schema/001_bronze.sql`) is what actually ships to
  production.
- **Games upsert, injuries append.** Games have a natural key (`game_id` +
  `source`) and represent current truth — re-ingesting should update, not
  duplicate. Injury reports are point-in-time snapshots by design (a
  player's status *as reported on a given day*) — collapsing them to
  "current status" happens in the Silver transform (Phase 2), not here.
  Losing that raw history in bronze would make injury *trend* features
  (e.g. "downgraded from Full to Limited this week") impossible later.
- **Play-by-play is deliberately excluded from Phase 1.** The pbp file is
  40-80MB/season and isn't needed until team/player rolling-stat features
  (Phase 3). `get_plays()` is implemented on the provider so it's ready
  when needed, but `run_ingestion.py` doesn't call it yet.

---

## Review notes on the design doc (for Phase 2+ planning)

A few gaps worth closing before building further phases, based on reading
the full doc:

1. **Model promotion criteria isn't fully specified.** §28 shows the
   *process* (train candidate → evaluate → compare → promote) and one
   example threshold (ATS accuracy), but doesn't define the actual
   decision rule — e.g., is it "any improvement," "improvement beyond
   noise on a held-out set," or "improvement across ≥2 of {accuracy, Brier
   score, ATS}"? Worth deciding before Phase 8, since a single noisy ATS
   percentage point could otherwise cause you to "improve" your way into
   a worse model.
2. **Odds source reconciliation isn't addressed.** §8's provider
   abstraction handles swapping *entire* providers, but §7 lists odds as
   its own category, implying multiple sportsbooks eventually. If/when a
   second odds source is added, Silver needs an explicit precedence rule
   (e.g. "prefer Pinnacle close over consensus, fall back to consensus if
   missing") rather than implicit "whichever loaded last wins."
3. **Injury impact model (§16) needs a v0 before it can be learned.** The
   doc correctly defers "learned from historical data" to later, but the
   "simple player-value model" for V1 isn't defined at all — even a rough
   position-weighted heuristic (e.g. QB = -8 to -12, starting OL/DL = -3
   to -5, WR1/CB1 = -2 to -3, bench = ~0) needs to be written down and
   versioned like any other feature, since it will silently bias every
   prediction until it's replaced.
4. **Weekly retraining cadence (§26, §39) isn't validated against data
   volume.** Early in the season (Week 1-3) there's very little
   *current-season* signal to retrain on — worth deciding whether early
   weeks retrain at all, or lean more heavily on prior-season priors, to
   avoid overfitting a model to 1-2 games of new data.

None of these block starting Phase 2 (Bronze → Silver → Gold), but #1 and
#3 will need answers before Phase 4 (ML training) and #4 before Phase 6
(automation).

---

## Next steps

The natural next slice is **Phase 2**: `bronze.games_raw` /
`bronze.injuries_raw` → `silver.games` / `silver.injuries` / `silver.teams`,
applying the type normalization, deduplication, and ID normalization called
for in §11 of the design doc. That unlocks Phase 3 (rolling stats + the
`gold.game_features` table) and is a natural place to pick this back up.
