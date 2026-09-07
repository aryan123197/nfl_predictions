# Design Decisions

Tracks decisions made on open questions from the design doc review notes
(see README "Review notes on the design doc"). Each entry should be updated
in place if a decision changes, with a note on why.

---

## 1. Model promotion criteria (design doc §28)

**Status:** Decided, not yet implemented (needed before Phase 4/8).

**Problem:** §28 defines the promotion *process* (train candidate → evaluate
→ compare → promote) but not the *decision rule*. A single noisy ATS
percentage point over a ~270-game season is not a reliable signal on its
own.

**Decision:** Track three metrics on a held-out set — ATS accuracy, Brier
score, and log loss. A candidate is promoted only if:

- it does not regress on **any** of the three by more than a small
  tolerance (exact tolerance TBD when Phase 4 lands and we have real
  variance estimates to calibrate against), **and**
- it improves on **at least one** of the three beyond a bootstrap
  confidence interval (not just a point-estimate improvement).

**Why not simpler alternatives:**
- "Any improvement" (e.g. the §28 example of ATS 54.1% → 56.2%) is too
  permissive — a single noisy metric can promote a worse model.
- "Improvement across all metrics" is too strict, especially early in the
  season when held-out sets are small.

**Open follow-up:** Pick the actual tolerance and bootstrap method once
Phase 4 produces real metric variance to calibrate against.

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

## Revision history

- 2026-09-07: Initial decisions recorded for all four open questions from
  the design doc review notes.
- 2026-09-07: Added decision #5 (injury point-in-time cutoff fallback),
  found while implementing and live-testing Phase 3 Slice A.
