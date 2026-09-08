# Design Decisions

Tracks decisions made on open questions from the design doc review notes
(see README "Review notes on the design doc"). Each entry should be updated
in place if a decision changes, with a note on why.

---

## 1. Model promotion criteria (design doc §28)

**Status:** Decided and implemented (`src/ml/promotion.py`, Phase 5).

**Problem:** §28 defines the promotion *process* (train candidate → evaluate
→ compare → promote) but not the *decision rule*. A single noisy ATS
percentage point over a ~270-game season is not a reliable signal on its
own.

**Decision:** Track three metrics on validation evaluation — accuracy, Brier
score, and log loss. A candidate is promoted over the incumbent champion only if:

- it does not regress on **any** of the three by more than calibrated noise
  tolerances (`log_loss <= +0.015`, `brier_score <= +0.010`, `accuracy >= -0.020`), **and**
- it strictly improves on **at least one** of the three metrics compared to the champion.

**Why not simpler alternatives:**
- "Any improvement" (e.g. the §28 example of ATS 54.1% → 56.2%) is too
  permissive — a single noisy metric can promote a worse model.
- "Improvement across all metrics" is too strict, especially early in the
  season when held-out sets are small.

Implemented in `src/ml/promotion.py::evaluate_candidate_promotion`.

---

## 2. Odds source reconciliation (design doc §7 / §8)

**Status:** Decided, partially implementable now (schema shape), full
reconciliation logic deferred until a second odds source exists.

**Problem:** §8's provider abstraction handles swapping *entire*
providers, but §7 lists odds as its own data category, implying multiple
sportsbooks eventually. Without an explicit precedence rule, Silver would
default to "whichever loaded last wins," which is not reproducible.

**Decision:** V1 ships with a single free odds feed (derived from
nflverse games data), so full reconciliation logic isn't needed yet. But
`silver.odds` gets a `source` column and a `precedence` config (a simple
ordered list in code, e.g. `["pinnacle", "consensus", "nflverse"]`) from
day one, so adding a second source later is additive rather than a Silver
schema migration.

**Open follow-up:** When a second odds source is added, implement the
actual precedence resolution logic in the Silver transform (Phase 2) using
the `source` column that's already in place.

---

## 3. Injury impact v0 (design doc §16)

**Status:** Decided, not yet implemented (needed before Phase 3 feature
engineering / Phase 4 training).

**Problem:** §16 says "the initial implementation can use a simple
player-value model" but never defines it. This will silently bias every
prediction until it's replaced with a learned model, so it needs to be
versioned and auditable like any other feature — not an ad hoc constant
buried in code.

**Decision:** Ship a versioned, position-weighted lookup table as a real
config file (`config/injury_impact_v0.yaml`), not inlined in Python, so
changes are visible in git history and diffable:

```yaml
version: v0
weights:
  QB1: -10.0
  OL_starter: -4.0
  DL_starter: -4.0
  WR1: -3.0
  CB1: -3.0
  RB1: -2.0
  TE1: -2.0
  S_starter: -2.0
  bench: 0.0
```

(Values above are placeholders pending a first pass at Phase 3 — the
point of this decision is the *mechanism* — versioned file, not the exact
numbers, which should be revisited once real data is available.)

**Open follow-up:** Values should be sanity-checked against actual
historical spread movement around injury announcements before Phase 4
training, not just intuition.

---

## 4. Weekly retraining cadence (design doc §26 / §39)

**Status:** Decided, needed before Phase 6 (automation).

**Problem:** §26's "Weekly: Retrain model" cadence isn't validated
against data volume. Early in the season (Weeks 1-3) there's very little
*current-season* signal — retraining weekly from Week 1 risks overfitting
a model to 1-2 games of new data.

**Decision:** Do not retrain on current-season data until **Week 4** of
the season. Before Week 4, use the prior-season-trained model with a
light shrinkage/Bayesian adjustment toward early-season results, rather
than a full retrain. From Week 4 onward, follow the §26 weekly retrain
cadence as designed.

**Implication for Phase 3:** `gold.game_features` needs to expose "weeks
into season" as a first-class feature so this rule can be applied
consistently in both training and serving.

---

## 5. Injury point-in-time cutoff when `reported_at` is missing

**Status:** Decided and implemented (Phase 3 Slice A).

**Problem:** Decision #3's injury impact calculation needs a cutoff
timestamp (design doc §19: only injuries reported before the prediction
time may count). `silver.injuries.reported_at` comes from nflverse's
`date_modified` field — which, per the Phase 1 README notes, nflverse
dropped from the 2025 injuries file entirely. Verified against live
2025 data during Slice A implementation: `reported_at` is NULL for
every real 2025 row, not just an edge case. A naive `reported_at <
game_date` filter silently excludes every injury, producing an
always-zero injury impact feature with no error to signal it.

**Decision:** Prefer `reported_at` when present. When it's NULL, fall
back to `(season, week) <= this game's (season, week)` — an injury
report is always filed during the days leading into that week's games,
so a report from the same week necessarily precedes kickoff. Coarser
than a real timestamp (week-level instead of hour-level), but still
strictly point-in-time safe: it can only look backward, never forward.
Implemented in `src/transform/gold_transform.py::_compute_injury_impact_rows`.

**Open follow-up:** If a future data source restores real report
timestamps, prefer them automatically (the code already does) — no
further action needed unless the fallback needs removing entirely.

---

## 6. Rolling stats scope: two windows, offense-focused, per-play proxies

**Status:** Decided and implemented (Phase 3 Slice B).

**Problem:** Design doc §14 lists four candidate rolling windows
(season-to-date, last 8/4/2 games) without mandating all of them, and
§14's metric list (EPA/play, success rate, sack rate, pressure rate,
explosive play rate, red-zone efficiency, third-down efficiency, ...)
includes some metrics nflverse's play-by-play doesn't directly support
without either extra complexity (true red-zone efficiency needs
per-drive aggregation) or an outright missing field (no dedicated
"pressure" flag).

**Decision:**
- **Two windows only:** `season_to_date` and `last_4`, not all four
  candidates. Stored in long format in `gold.team_rolling_stats` (one
  row per team/game/window) so adding a third window later is new rows,
  not a schema change.
- **Season-to-date rolling sub-metrics promoted into `gold.game_features`:**
  In addition to `home_off_epa`/`away_off_epa`/`home_def_epa`/`away_def_epa`,
  the granular splits (`pass_epa`, `rush_epa`, `turnover_rate`, `pressure_rate`,
  `explosive_play_rate`, `third_down_rate`, `success_rate`) for both home and away
  are promoted into `gold.game_features` and `src/ml/features.py::FEATURE_COLUMNS`.
  In 2025 walk-forward backtesting, adding these features boosted out-of-sample accuracy
  from 63.73% to 64.79% and improved log loss from 0.6586 to 0.6560.
- **Offense-focused:** every metric except `def_epa` is computed from a
  team's own offensive plays. There's no full defensive breakdown
  (defensive success rate, defensive sack rate, etc.) — just the one
  defensive EPA-allowed summary number, matching what `gold.game_features`
  actually needs per the design doc's §18 example.
- **`pressure_rate` uses `qb_hit`** as a proxy — nflverse's public
  play-by-play has no dedicated "pressure" flag.
- **`red_zone_td_rate` is a per-play proxy** (TD rate on snaps run
  inside the 20), not true per-drive red-zone efficiency (trips ending
  in a TD / total red-zone trips). The latter needs drive-level
  aggregation (nflverse does have a `drive` column, so this is
  buildable later, just deferred).
- **Explosive play thresholds:** 15+ yards on a pass, 10+ yards on a
  run — the standard NFL analytics convention.
- **Player-level features (design doc §15) are entirely out of scope**
  here — `gold.team_rolling_stats` is team aggregates only. This is why
  `home_qb_epa`/`away_qb_epa` from §18's example feature list aren't in
  `gold.game_features` yet.

**Also fixed during implementation:** `get_plays()` never actually
worked before this slice — `_fetch_csv`'s `pd.read_csv` call had no
`compression` argument, so it silently tried to parse the gzip-
compressed play-by-play file as raw text (pandas infers compression
from a file path's suffix, which a `BytesIO` payload doesn't have).
Caught by running it against live data before building anything on top
of it; fixed by passing `compression="gzip"` explicitly when the source
filename ends in `.gz`.

**Open follow-up:** Per-drive red-zone efficiency and a true pressure
metric (if nflverse or a future provider ever exposes one) are natural
additions once there's a concrete modeling need for them.

---

## 7. Phase 4 scope: win probability only, single split, no MLflow yet

**Status:** Decided and implemented (Phase 4).

**Problem:** The design doc's own phase breakdown scopes Phase 4 tightly
("Features → XGBoost → Predictions") and keeps Phase 5 ("simulate 2025
week-by-week") separate. But §20 describes three prediction types (win
probability, score, spread) and §27 calls for MLflow versioning, so it
would be easy to over-scope Phase 4 into building all of that at once.

**Decision:**
- **Win probability only.** `predicted_home_score`/`predicted_away_score`/
  `predicted_margin`/`cover_probability` exist as columns on
  `ml.predictions` (per §21's schema) but are left NULL — score and
  spread prediction are a follow-up slice that shares this same
  train/evaluate/predict scaffolding once it exists.
- **One train/holdout split, not the walk-forward loop.** Training uses
  `src/ml/train.py::split_train_holdout` — a season-based split by
  default (train on every season strictly before a holdout season,
  evaluate on the holdout season), falling back to a within-season
  week-based split when only one season of data exists. The full
  week-by-week retrain-and-predict simulation (design doc §25) is
  Phase 5's job, built around this same training code, not
  reimplemented here.
- **Model versioning is a plain `models/<version>/metadata.json`**,
  not MLflow. MLflow is explicitly Phase 8 (MLOps) per this project's
  own phase table — Phase 4 tracks the same fields §27 calls for
  (training data, features, hyperparameters, metrics, git commit) in a
  format that's trivial to later import into MLflow, without taking on
  that dependency before there's a real need for experiment comparison
  across many runs.
- **Feature set is `src/ml/features.py::FEATURE_COLUMNS`** — an
  explicit, hand-maintained list, not "every numeric column in
  `gold.game_features`" — so a trained model's feature set is always
  traceable to a specific, intentional list rather than silently
  changing whenever a new gold column is added.
- **Missing values are left as NaN, not imputed.** XGBoost handles
  missing values natively (learns a default split direction per node),
  which was one of the reasons XGBoost was chosen as the baseline in
  the first place — a large fraction of real rows have NaN
  `off_epa`/`def_epa` (a team's first 1-3 games of a season, before
  rolling stats exist), and imputing with e.g. 0 would assert
  "league-average," a much stronger and more arbitrary claim than
  "unknown."

**Also found and fixed while implementing this slice** (both only
surfaced by backfilling and training against real multi-season data,
not by unit tests against synthetic fixtures):
- `gold_transform.py`'s injury-impact cutoff compared a tz-aware
  `reported_at` (real pre-2025 seasons have full ISO8601 timestamps
  with a UTC offset) against a tz-naive `game_date`, crashing with
  `TypeError: Cannot compare tz-naive and tz-aware datetime-like
  objects`. 2025-only testing never exercised this path, since 2025
  dropped `date_modified` (`reported_at`) entirely (Decision #5) —
  every `reported_at` was NULL, never a real timestamp. Fixed by
  normalizing to naive UTC (`utc=True` then `.dt.tz_localize(None)`)
  right after parsing.
- `split_features_target` crashed on a training set where an entire
  feature column happened to be all-NULL (no numeric value anywhere in
  it, so pandas gives it `object` dtype instead of `float64` — XGBoost
  rejects `object` dtype outright). Fixed with an explicit
  `.astype(float)`.
- `model_version` had only second-level timestamp precision, so two
  training runs within the same second (a realistic case for an
  automated retraining pipeline, and for local testing) would collide
  on the same version string. Fixed by appending a short random suffix.

**Open follow-up:** Score/spread prediction builds on this scaffolding rather than
replacing it.

---

## 8. Walk-forward backtesting simulation & lineage tracking (Phase 5)

**Status:** Decided and implemented (Phase 5).

**Problem:** Backtesting a season must strictly mirror real-world sequential
operation (§25) — model $M_0$ predicts Week 1 before kickoff, outcomes are
revealed, and weekly retraining begins only once sufficient current-season
sample exists (Decision #4: Week 4+). Furthermore, point-in-time prediction
integrity must be preserved in `ml.predictions` without overwriting historical
forecasts.

**Decision:**
- `src/ml/backtest.py::run_walk_forward_backtest` runs the week-by-week
  loop across the target season (e.g. 2025).
- In Weeks 1–3, current-season retraining is skipped per Decision #4.
- In Weeks 4+, candidate models are trained on all data up to the completed
  week and evaluated against the incumbent champion on recent validation games
  using Decision #1's promotion gate (`src/ml/promotion.py`).
- Every weekly prediction is written to `ml.predictions` tagged with
  `model_version=f"backtest_{season}_w{week:02d}_{champion_version}"`.
- The full audit trail of model promotions, weekly metrics, and cumulative
  season calibration metrics is saved to `models/backtests/backtest_{season}_{timestamp}.json`.

---

## Revision history

- 2026-09-07: Initial decisions recorded for all four open questions from
  the design doc review notes.
- 2026-09-07: Added decision #5 (injury point-in-time cutoff fallback),
  found while implementing and live-testing Phase 3 Slice A.
- 2026-09-07: Added decision #6 (rolling stats scope), recorded while
  implementing Phase 3 Slice B.
- 2026-09-08: Added decision #7 (Phase 4 scope), recorded while
  implementing and live-backfill-testing Phase 4.
- 2026-09-08: Updated decision #1 with calibrated promotion tolerances and
  added decision #8 (walk-forward backtesting), recorded while implementing Phase 5.
