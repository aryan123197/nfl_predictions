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
| 2 | Bronze → Silver → Gold transforms | 🟡 **Silver working** — Gold moved into Phase 3 |
| 3 | Point-in-time feature engineering | ✅ **Slices A + B working** — Elo, injury impact, rolling EPA stats, full `gold.game_features` (player features deferred) |
| 4 | ML training (XGBoost baseline) | 🟡 **Win probability working** — score/spread prediction deferred |
| 5 | Walk-forward backtesting (2025) | ✅ **Working** — week-by-week simulation with Decision #1 promotion gate & Decision #4 cadence |
| 6 | Automation (GitHub Actions) | ✅ **Working** — scheduled data & prediction pipeline + weekly retraining & promotion gate |
| 7 | Live 2026 data connection | ⬜ Not started |
| 8 | MLOps (MLflow, model registry, monitoring) | ⬜ Not started |
| 9 | Application (FastAPI + React) | 🟡 **Slice A working** — read-only API + React UI over silver/gold; prediction panel wired |
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

## What's actually built right now (Phase 3 — Slice A)

Silver → Gold: Elo team ratings, injury impact v0, and a first
`gold.game_features` table — everything derivable without play-by-play
data. See `DECISIONS.md` for the reasoning behind every design choice
below.

```
config/injury_impact_v0.yaml        Versioned position/status weight table (Decision #3)
src/features/elo.py                 Elo rating computation (design doc §17)
src/features/injury_impact.py       Injury impact heuristic (design doc §16)
schema/003_gold.sql                 Gold schema, Slice A tables only
src/transform/gold_transform.py     CLI entrypoint: silver → gold
tests/test_elo.py                   Elo unit tests (no DB)
tests/test_injury_impact.py         Injury impact unit tests (no DB)
tests/test_gold_transform.py        Point-in-time correctness tests (in-memory SQLite)
```

**What's in `gold.game_features`:** `home_elo`/`away_elo`/`elo_difference`,
`home_injury_impact`/`away_injury_impact`, `home_rest_days`/`away_rest_days`,
`home_recent_form`/`away_recent_form` (win % over the last 4 completed
games), and `opening_spread`/`current_spread`/`spread_movement` (V1 has
one odds snapshot, so movement is always 0 — see Decision #2).
`temperature`/`wind` columns exist per the design doc's example feature
set but stay NULL — no weather provider yet. EPA-based columns
(`home_off_epa`, `away_off_epa`, `home_def_epa`, `away_def_epa`) were
added in Slice B, see below — `home_qb_epa`/`away_qb_epa` are still not
in the table, since those are player-level (design doc §15), out of
scope until a future player-features slice.

**Point-in-time correctness (design doc §19) is the core guarantee
here**, verified two ways: unit tests on the pure Elo/injury-impact
functions, and DB-backed tests that a *later* week's rating/injury/form
never leaks into an *earlier* game's features. Verified against live
2025 Weeks 1-2 data too — this is how a real gap was caught during
implementation: nflverse dropped injury report timestamps for 2025
entirely (see Decision #5), which would have silently zeroed out every
injury feature without the live check.

**Verified working end-to-end** against live 2025 Weeks 1-2 nflverse
data: 32 teams, 32 games, 64 team-ratings rows, 64 injury-impact rows,
32 `game_odds` rows, 32 `game_features` rows. 40/40 offline tests pass
as of the bug-fix pass after this slice (see `DECISIONS.md` for what
was found and fixed in a follow-up code review: NaN-into-INTEGER
crashes, a tie-scoring bug, an injury data "Frankenstein row" bug, and
a silent schema-migration gap).

---

## What's actually built right now (Phase 3 — Slice B)

Play-by-play ingestion through rolling team stats — the EPA-dependent
half of Phase 3 that Slice A deferred. See `DECISIONS.md` #6 for the
full reasoning behind every scope choice below.

```
src/ingest/pipeline_metadata.py     Shared metadata.pipeline_runs bookkeeping (extracted from run_ingestion.py)
src/ingest/run_pbp_ingestion.py     CLI entrypoint: provider → bronze.plays_raw (full season, bulk replace)
schema/001_bronze.sql               bronze.plays_raw gained season/week/sack/qb_hit/first_down/third_down columns
schema/004_silver_plays.sql         New silver.plays table
src/transform/plays_transform.py    CLI entrypoint: bronze → silver.plays (bulk replace, not per-row upsert)
schema/005_gold_rolling_stats.sql   New gold.team_rolling_stats table + game_features EPA columns
src/features/rolling_stats.py       Point-in-time rolling stat computation (design doc §14)
tests/test_rolling_stats.py         Unit tests: weighted averaging, window boundaries, season resets
```

**What's in `gold.team_rolling_stats`:** two windows (`season_to_date`,
`last_4`) × 12 metrics per team per game — EPA/play, offensive EPA,
defensive EPA (allowed), pass EPA, rush EPA, success rate, turnover
rate, sack rate, pressure rate (a `qb_hit`-based proxy — nflverse has no
dedicated pressure flag), explosive play rate, red-zone TD rate (a
per-play proxy, not true per-drive efficiency), and third-down rate.
Only `season_to_date` is promoted into `gold.game_features`
(`home_off_epa`/`away_off_epa`/`home_def_epa`/`away_def_epa`) — `last_4`
stays available in `gold.team_rolling_stats` for future experimentation.

**A real, previously-undetected bug was caught here too:** `get_plays()`
had never actually been run against live data before this slice (Phase 1's
README flagged it as "implemented but not called yet"). It didn't work —
`_fetch_csv`'s `pd.read_csv` had no `compression` argument, so it tried
to parse the gzip-compressed play-by-play file as raw text and failed
immediately with a `UnicodeDecodeError`. Caught by testing against real
data before building anything on top of it, not by a unit test (offline
tests can't catch a bug in *fetching* real data). Fixed by passing
`compression="gzip"` when the source filename ends in `.gz`. See
`DECISIONS.md` #6.

**Verified working end-to-end** against live 2025 season data: 48,771
play-by-play rows ingested and transformed, 128 `gold.team_rolling_stats`
rows and full EPA columns in `gold.game_features` for Weeks 1-2, with
Week 1 correctly showing NULL EPA (no prior games) and Week 2 showing
real values reflecting each team's Week 1 performance. 52/52 offline
tests pass (40 existing + 11 new, plus 1 from the follow-up code-review
fix pass below).

**Fixed in a follow-up code review of this slice** (all three verified
against the real 2025 season before and after, and all three latent
rather than actively firing — see the commit message for the full
reasoning):
- `rolling_stats._sum_history` used `value or 0`, which does *not* fall
  back for `NaN` (NaN is truthy). A team appearing on only one side of
  the ball in a game gets NaN from the offense/defense outer merge, and
  since NaN + anything is NaN, one such game would silently poison that
  stat for every *later* game in the window. Confirmed this never occurs
  across all 570 real 2025 team-games — a complete NFL game always has
  both teams on offense and defense — so it guards partial ingestion, a
  truncated pbp file, or a forfeit, not current data.
- `run_pbp_ingestion._replace_plays` bulk-wrote `bronze.plays_raw`'s
  INTEGER `season`/`week`/`qtr` columns without the NaN→NULL coercion
  applied everywhere else in this slice — the same failure class the
  Slice A review fixed via `_int_or_none`, missed on the newer `to_sql`
  path (SQLite coerces NaN silently, Postgres rejects it).
- The empty-DataFrame fallback column lists in
  `compute_per_game_team_stats` were hand-duplicated from
  `_PER_GAME_COLUMNS` and could drift from it whenever a metric is
  added; they're now derived from it.

**Still deferred:** player-level features (design doc §15 — QB/WR/
defensive player stats aggregated to team level), true per-drive
red-zone efficiency, and weather data (no provider). None of these
block Phase 4 (ML training), which can now start against a genuinely
complete team-level feature table.

---

## Continuous integration (a slice of Phase 6)

Two workflows, split along the line the test suite already draws between
offline and live-data tests:

```
.github/workflows/tests.yml         PR gate: the 52 offline tests, Python 3.11 + 3.12
.github/workflows/integration.yml   Weekly + on-demand: the 6 live nflverse tests
```

**The PR gate is offline-only on purpose.** `pytest -m "not integration"`
needs no network and no database — `src/db.py` defaults to a local SQLite
file — so a nflverse outage or a slow 40-80MB play-by-play download can
never make an unrelated PR look broken. Verified by running the suite with
the network blackholed: 52/52 still pass.

**The live tests get a schedule instead of a gate**, because upstream data
changes are a real recurring failure mode here that offline tests are
structurally incapable of catching — both times it happened, only live
data caught it (nflverse dropping `date_modified`, per `DECISIONS.md` #5;
`get_plays()` never having worked at all, per #6). A Monday-morning run
turns "discovered while building the next slice" into "discovered the
Monday after upstream changed." `workflow_dispatch` is enabled so a slice
can still be verified against live data on demand.

This covers CI only. The Phase 6 design-doc work — scheduled *pipeline*
runs (weekly ingestion, retraining cadence per `DECISIONS.md` #4) — is
still ahead, and needs Phase 4 to exist first.

---

## What's actually built right now

### Phase 4 — Win probability baseline
Features → XGBoost → Predictions (design doc §24), scoped to win
probability only — see `DECISIONS.md` #7 for details.

```
schema/006_ml.sql                     ml.predictions (insert-only) + ml.game_results
src/transform/game_results_transform.py   silver.games -> ml.game_results (actual outcomes, ATS cover)
src/ml/features.py                    FEATURE_COLUMNS (explicit, versioned) + training-frame loader
src/ml/train.py                       CLI entrypoint: train, evaluate, save model, write predictions
scripts/backfill_seasons.py           Extended to run the FULL pipeline across a season range
tests/test_ml_train.py                Split logic + full-pipeline integration tests (in-memory SQLite)
```

**Usage:**
```bash
# Populate multi-season history
python scripts/backfill_seasons.py --start 2020 --end 2025

# Train, evaluate on holdout season, write predictions
python -m src.ml.train
```

### Phase 5 — Walk-forward backtesting (2025)
Simulates sequential real-world operation and retraining across the 2025 season —
design doc §25 (walk-forward training), §26 (continuous learning), §28 (model promotion).

```
src/ml/promotion.py                Candidate promotion gate (Decision #1 multi-metric tolerance rules)
src/ml/backtest.py                 CLI entrypoint: sequential week-by-week simulation loop
tests/test_backtest.py             Unit tests + full walk-forward SQLite simulation
```

**Usage:**
```bash
# Run week-by-week backtest simulation on the 2025 season
python -m src.ml.backtest --season 2025

# Custom starting week or retraining threshold
python -m src.ml.backtest --season 2025 --start-week 1 --retrain-start-week 4
```

**What it does:**
1. Trains an initial baseline champion model on all historical seasons strictly prior to 2025 (`season < 2025`).
2. Iterates week-by-week ($W = 1 \dots 18$):
   - Generates pre-game out-of-sample predictions for Week $W$ using the active champion model.
   - Inserts predictions into `ml.predictions` tagged with `model_version=f"backtest_{season}_w{W}_{champ_version}"`.
   - Resolves actual results from `ml.game_results` and scores out-of-sample accuracy, log loss, and Brier score.
   - Enforces the Decision #4 retraining cadence: Weeks 1–3 skip current-season retraining to avoid overfitting on tiny samples. From Week 4 onward, trains candidate models on all data up to Week $W$.
   - Evaluates candidate models against the champion on validation games using Decision #1's promotion gate (`src/ml/promotion.py`). If the candidate improves without exceeding regression tolerances, it is promoted to active champion.
3. Aggregates cumulative season performance and exports full lineage logs and summary JSON to `models/backtests/backtest_{season}_{timestamp}.json`.

---

### Phase 9 — Slice A: Read-only API + React UI
A read-only FastAPI service over silver/gold and a React UI on top of it —
design doc §34 (architecture), §35 (prediction UI).

```
requirements-api.txt              API-only dependencies
src/api/main.py                   FastAPI app: the §34 endpoints
src/api/queries.py                Read-only SQL over silver + gold
src/api/schemas.py                Pydantic response models
src/api/predictions.py            The Phase 4 boundary
tests/test_api.py                 16 offline API tests against SQLite
frontend/                         Vite + React + TypeScript
  src/components/WeekView.tsx     Season/week pickers + game grid
  src/components/GameDetail.tsx   §35 game page
  src/components/PredictionPanel.tsx  Win probability, predicted score, model vs market
  src/components/FeatureComparison.tsx  Head-to-head gold.game_features
  src/components/format.test.ts   14 unit tests
```

### Running the API & Frontend
```bash
pip install -r requirements.txt -r requirements-api.txt
uvicorn src.api.main:app --reload      # http://127.0.0.1:8000/docs

---

### Phase 6 — Automated Pipelines & Continuous Retraining (GitHub Actions)
Fully automated, scheduled, cloud-native orchestration (design doc §26, §28, §29, §30, §31) with lineage tracking in `metadata.pipeline_runs` and active champion management in `models/champion`.

```
src/ml/registry.py                    Champion model loader, promotion registry, & prediction generator
src/pipeline/run_pipeline.py          CLI/Workflow entrypoint: Ingest -> Silver -> Gold -> Predict
src/pipeline/run_retrain.py           CLI/Workflow entrypoint: Results -> Evaluation -> Retrain -> Promotion Gate
.github/workflows/pipeline.yml        Scheduled data & prediction refresh (Wed/Fri/Sun 14:00 UTC)
.github/workflows/retrain.yml         Scheduled weekly retraining & candidate promotion (Tue 06:00 UTC)
.github/workflows/tests.yml           PR gate: offline test suite across Python 3.11/3.12 + frontend
.github/workflows/integration.yml     Weekly live-data canary against upstream nflverse
tests/test_pipeline.py                5 offline integration tests with SQLite & mock providers
```

**Usage:**
```bash
# Run full automated data & prediction pipeline
python -m src.pipeline.run_pipeline --season 2025 --week 1

# Run weekly outcome evaluation & candidate retraining loop
python -m src.pipeline.run_retrain --season 2025 --week 6

# Force retrain during early season (overriding Decision #4's Week 1-3 skip)
python -m src.pipeline.run_retrain --season 2025 --week 2 --force-retrain
```

**Workflows:**
1. **Data Ingestion & Prediction Refresh (`.github/workflows/pipeline.yml`)**:
   - Runs Wednesdays, Fridays, and Sundays before kickoff.
   - Refreshes odds, injury reports, silver, and gold features, then generates predictions using the active champion model.
2. **Weekly Model Retraining (`.github/workflows/retrain.yml`)**:
   - Runs Tuesdays after Monday Night Football.
   - Syncs game results, checks Decision #4's cadence gate, trains candidate models on cumulative data, and promotes candidates to champion via Decision #1's multi-metric gate.

---

## Next steps

**Phase 7** (connecting live 2026 data sources) and **Phase 8** (MLOps with MLflow and advanced monitoring) are the next milestones on the roadmap. Model spread/margin predictions and player-level feature extensions (design doc §15) remain available as follow-up modeling enhancements.
