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
| 2 | Bronze → Silver → Gold transforms | ⬜ Not started |
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
