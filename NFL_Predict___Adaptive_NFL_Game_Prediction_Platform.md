# NFL Predict — Adaptive NFL Game Prediction Platform

**Document Type:** Product & Technical Design Document  
**Version:** 1.0  
**Target Season:** 2026 NFL Season  
**Initial Infrastructure Budget:** $0–$10/month  
**Primary Goal:** Build a production-style data engineering + machine learning platform that automatically ingests NFL data, generates point-in-time predictions, evaluates those predictions, and continuously improves throughout the 2026 season.

---

# 1. Executive Summary

NFL Predict is an end-to-end data engineering, machine learning, and MLOps platform designed to predict NFL games throughout the 2026 season.

The system will predict:

1. Which team will win
2. Probability that each team wins
3. Predicted score
4. Predicted point differential
5. Probability that a team covers the betting spread
6. Model confidence
7. Key factors contributing to the prediction

The system will begin with historical NFL data from previous seasons to train and backtest models. It will then operate continuously during the 2026 season.

The core system will automatically:

- Ingest new NFL data
- Ingest injury updates
- Ingest betting lines
- Ingest game results
- Update team and player statistics
- Generate new point-in-time features
- Make predictions
- Record predictions before games begin
- Evaluate predictions after games finish
- Retrain models periodically
- Compare new models against the production model
- Deploy a new model only when it meets predefined criteria

The first implementation should prioritize **low cost, simplicity, reproducibility, and strong engineering fundamentals** rather than immediately building a complex cloud architecture.

---

# 2. Project Goals

## Primary Goals

Build a real-world ML/data engineering system demonstrating:

- REST API ingestion
- ETL/ELT
- PostgreSQL
- PySpark
- Data modeling
- Incremental data processing
- Feature engineering
- Machine learning
- Model evaluation
- MLflow
- Automated workflows
- CI/CD
- MLOps
- Explainable predictions
- Point-in-time correctness
- Continuous model improvement

## Secondary Goals

Eventually demonstrate:

- Azure
- Databricks
- Delta Lake
- Cloud deployment
- Advanced online learning
- Reinforcement learning experimentation

---

# 3. Non-Goals for V1

The first version will NOT attempt to:

- Build a sophisticated reinforcement learning system
- Process real-time betting transactions
- Guarantee profitable betting performance
- Build a high-frequency trading-style betting system
- Use expensive premium NFL data APIs
- Deploy Kubernetes
- Maintain large always-running cloud clusters

Reinforcement learning should only be considered after the supervised-learning system and automated pipeline are working reliably.

---

# 4. High-Level Architecture

```text
                         EXTERNAL DATA SOURCES
                                  |
             +--------------------+--------------------+
             |                    |                    |
         NFL Data            Injury Data          Odds Data
             |                    |                    |
             +--------------------+--------------------+
                                  |
                                  v
                        +--------------------+
                        |  INGESTION LAYER   |
                        | Python / REST APIs |
                        +---------+----------+
                                  |
                                  v
                        +--------------------+
                        |    BRONZE DATA     |
                        | Raw source data    |
                        +---------+----------+
                                  |
                               PySpark
                                  |
                                  v
                        +--------------------+
                        |    SILVER DATA     |
                        | Cleaned/normalized |
                        +---------+----------+
                                  |
                               PySpark
                                  |
                                  v
                        +--------------------+
                        |     GOLD DATA      |
                        | Analytics tables   |
                        +---------+----------+
                                  |
                                  v
                        +--------------------+
                        |  FEATURE PIPELINE  |
                        +---------+----------+
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
             MODEL TRAINING               PREDICTION
                    |                           |
                  MLflow                       |
                    |                           |
                    +-------------+-------------+
                                  |
                                  v
                            PostgreSQL
                                  |
                                  v
                              FastAPI
                                  |
                                  v
                               React
```

---

# 5. Initial Infrastructure

The initial system should run primarily locally.

## Local Components

- Python
- PostgreSQL
- Docker
- PySpark
- MLflow
- XGBoost / LightGBM
- FastAPI
- React
- Git
- GitHub Actions

Initial storage:

```text
PostgreSQL
```

Historical files can additionally be stored as:

```text
Parquet
```

The project should avoid cloud infrastructure until the local implementation is proven.

---

# 6. Production Architecture — Initial Cheap Version

```text
                     GitHub Repository
                            |
                            v
                    GitHub Actions
                            |
                    Scheduled Workflow
                            |
                            v
                    Python ETL Runner
                            |
                +-----------+-----------+
                |                       |
                v                       v
           External APIs          PostgreSQL
                                        |
                              +---------+---------+
                              |                   |
                              v                   v
                         ML Pipeline          FastAPI
                                                  |
                                                  v
                                                React
```

GitHub Actions will act as the initial scheduler/orchestrator.

The database should be hosted remotely once live 2026 automation begins so that the pipeline can run even when the developer's computer is turned off.

---

# 7. Data Sources

The system should support pluggable data providers.

Potential data categories:

## Games

- Game ID
- Season
- Week
- Date
- Home team
- Away team
- Venue
- Game status
- Final score

## Play-by-Play

- Play ID
- Game ID
- Quarter
- Game clock
- Down
- Distance
- Field position
- Play type
- Yards
- EPA
- Success
- Pass/rush
- Turnover
- Touchdown
- Player involvement

## Players

- Player ID
- Name
- Position
- Team
- Status

## Player Statistics

- Snap count
- Passing statistics
- Rushing statistics
- Receiving statistics
- Defensive statistics
- EPA
- Success rate

## Injuries

- Player ID
- Team
- Position
- Injury
- Status
- Reported timestamp
- Expected return
- Source

## Betting

- Game ID
- Sportsbook
- Timestamp
- Spread
- Moneyline
- Total

## Weather

- Temperature
- Wind
- Precipitation
- Weather condition
- Timestamp

---

# 8. Data Provider Abstraction

The downstream pipeline must not depend directly on a specific API provider.

Create an abstraction:

```python
class NFLDataProvider:
    def get_games(self, ...):
        pass

    def get_plays(self, ...):
        pass

    def get_players(self, ...):
        pass

    def get_injuries(self, ...):
        pass

    def get_odds(self, ...):
        pass
```

Different providers should implement this interface.

Example:

```text
NFLDataProvider
       |
       +--- NFLVerseProvider
       |
       +--- LiveNFLProvider
       |
       +--- OddsProvider
```

This allows free historical data sources to be replaced by paid live APIs without rewriting downstream processing.

---

# 9. Data Architecture

The system will follow a simplified medallion architecture.

```text
Bronze
  |
  v
Silver
  |
  v
Gold
  |
  v
Features
```

PostgreSQL schemas should represent the layers:

```text
bronze
silver
gold
ml
metadata
```

---

# 10. Bronze Layer

Bronze stores raw or minimally modified source data.

Example tables:

```text
bronze.games_raw
bronze.plays_raw
bronze.players_raw
bronze.injuries_raw
bronze.odds_raw
bronze.weather_raw
```

Bronze should preserve source information for reproducibility.

Common metadata:

```text
source
ingested_at
source_timestamp
pipeline_run_id
raw_payload_hash
```

Bronze data should generally be append-oriented.

---

# 11. Silver Layer

Silver contains cleaned and normalized data.

Tables:

```text
silver.games
silver.plays
silver.players
silver.player_game_stats
silver.injuries
silver.odds
silver.weather
silver.teams
```

Responsibilities:

- Data type normalization
- Deduplication
- Team ID normalization
- Player ID normalization
- Timestamp normalization
- Null handling
- Validation
- Schema enforcement

---

# 12. Gold Layer

Gold contains analytics-ready entities.

Tables:

```text
gold.team_game_stats
gold.team_rolling_stats
gold.player_game_stats
gold.player_rolling_stats
gold.team_ratings
gold.injury_impact
gold.game_odds
gold.game_features
```

Gold should be designed around ML and analytical use cases rather than source APIs.

---

# 13. Core Database Model

## `silver.games`

```text
game_id
season
week
game_date
home_team_id
away_team_id
home_score
away_score
venue_id
status
created_at
updated_at
```

## `silver.plays`

```text
play_id
game_id
play_timestamp
quarter
game_clock
down
distance
yard_line
offense_team_id
defense_team_id
play_type
yards
epa
success
pass
rush
turnover
touchdown
passer_id
rusher_id
receiver_id
```

## `silver.players`

```text
player_id
name
position
team_id
status
```

## `silver.injuries`

```text
injury_id
player_id
team_id
status
injury_type
reported_at
expected_return
source
```

## `silver.odds`

```text
odds_id
game_id
sportsbook
timestamp
spread
moneyline_home
moneyline_away
total
```

---

# 14. Team Rolling Statistics

The system should calculate rolling statistics using only information available before a game.

Potential windows:

- Season-to-date
- Last 8 games
- Last 4 games
- Last 2 games

Features include:

```text
EPA/play
Offensive EPA
Defensive EPA
Pass EPA
Rush EPA
Success rate
Turnover rate
Sack rate
Pressure rate
Explosive play rate
Red-zone efficiency
Third-down efficiency
```

---

# 15. Player Features

Potential QB features:

```text
EPA/play
Completion %
Success rate
Sack rate
Pressure rate
Turnover rate
```

WR features:

```text
Targets
Target share
Yards/route
EPA
Reception rate
Snap share
```

Defensive player features:

```text
Snap share
Pressure rate
Tackles
Coverage metrics
Interceptions
```

Player features should eventually be aggregated into team-level features.

---

# 16. Injury Features

The system should not treat every injured player equally.

Instead, calculate an estimated team impact.

Example:

```text
QB1   -10.0
WR1    -3.0
LT     -4.0
CB1    -2.5

Total injury impact = -19.5
```

The initial implementation can use a simple player-value model.

Later, player impact can be learned from historical data.

---

# 17. Team Ratings

Maintain dynamic team ratings.

Initial implementation:

```text
Elo rating
```

Potential future ratings:

```text
Offensive rating
Defensive rating
QB rating
Power rating
```

These ratings update after completed games.

---

# 18. Feature Table

`gold.game_features` contains one row per prediction opportunity.

Example:

```text
game_id
season
week

home_team_id
away_team_id

home_elo
away_elo
elo_difference

home_off_epa
away_off_epa

home_def_epa
away_def_epa

home_qb_epa
away_qb_epa

home_injury_impact
away_injury_impact

home_recent_form
away_recent_form

home_rest_days
away_rest_days

temperature
wind

opening_spread
current_spread
spread_movement
```

The final feature set may eventually contain 50–200 features.

---

# 19. Point-in-Time Correctness

This is a critical requirement.

A prediction must only use information that was available before the prediction timestamp.

For every prediction:

```text
prediction_timestamp < game_start
```

Every feature must satisfy:

```text
feature_timestamp <= prediction_timestamp
```

No future information may leak into training or prediction.

This requirement should be enforced through tests.

---

# 20. Prediction System

The system initially produces three predictions.

## Win Prediction

```text
P(home_team_wins)
P(away_team_wins)
```

Example:

```text
KC = 68%
BUF = 32%
```

## Score Prediction

```text
Predicted home score = 27.3
Predicted away score = 23.1
```

## Spread Prediction

```text
Predicted margin = +4.2
```

Given:

```text
Market spread = KC -2.5
```

the model can calculate its difference from the market.

---

# 21. Prediction Table

Create:

```text
ml.predictions
```

Columns:

```text
prediction_id
game_id
model_version
prediction_timestamp

home_win_probability
away_win_probability

predicted_home_score
predicted_away_score

predicted_margin

market_spread
cover_probability

feature_snapshot_id
```

Predictions must NEVER be overwritten.

Each prediction is a historical record of what the model believed at a particular point in time.

---

# 22. Game Results

After games finish:

```text
ml.game_results
```

Columns:

```text
game_id
actual_home_score
actual_away_score
actual_margin
home_team_won
home_team_covered
result_timestamp
```

---

# 23. Model Evaluation

Join:

```text
predictions
+
actual_results
```

to calculate:

## Win prediction

- Accuracy
- Log loss
- Brier score
- Calibration

## Score prediction

- MAE
- RMSE

## Spread

- Against-the-spread accuracy
- Margin MAE

The system should track these metrics over time.

---

# 24. Model Training

Start with supervised learning.

Initial candidates:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. LightGBM

XGBoost should be the primary baseline model.

Do NOT start with reinforcement learning.

---

# 25. Walk-Forward Training

Historical backtesting must simulate real-world operation.

Example:

```text
2020–2024 → initial training
2025 Week 1 → predict
2025 Week 1 → reveal result
2025 Week 2 → retrain/update
2025 Week 2 → predict
...
2025 Week 18
```

The model must never see future games during a prediction.

This creates a realistic simulation of the 2026 production system.

---

# 26. Continuous Learning

The system should distinguish between:

## Fast feature updates

These can happen frequently:

- Injuries
- Odds
- Weather
- Roster status

## Model retraining

This should happen less frequently.

Initial schedule:

```text
Daily:
Data ingestion

Every few hours:
Injury / odds updates

Before games:
Generate predictions

After games:
Ingest results

Weekly:
Retrain model
```

Continuous learning should NOT mean retraining the model every time an API changes.

---

# 27. Model Versioning

Use MLflow.

Example:

```text
Model v1.0
Preseason 2026

Model v1.1
After Week 1

Model v1.2
After Week 2

Model v1.3
After Week 3
```

Track:

```text
Training data version
Feature version
Hyperparameters
Metrics
Model artifact
Git commit
```

---

# 28. Model Promotion

Never automatically replace the production model without evaluation.

Process:

```text
Training data
      |
      v
Train candidate model
      |
      v
Evaluate candidate
      |
      +------> Worse → Reject
      |
      v
Better
      |
      v
Register candidate
      |
      v
Production
```

Example:

```text
Production v1.3
ATS accuracy = 54.1%

Candidate v1.4
ATS accuracy = 56.2%

Candidate passes threshold
        |
        v
Deploy v1.4
```

---

# 29. Automated Pipeline

Initial workflow:

```text
NFL_DATA_PIPELINE

1. ingest_games
2. ingest_plays
3. ingest_players
4. ingest_injuries
5. ingest_odds
6. ingest_weather

        ↓

7. validate_data

        ↓

8. transform_silver

        ↓

9. build_gold_tables

        ↓

10. build_features

        ↓

11. generate_predictions

        ↓

12. store_predictions
```

Weekly ML workflow:

```text
MODEL_TRAINING_PIPELINE

1. build_training_dataset
2. validate_point_in_time_correctness
3. train_models
4. evaluate_models
5. log_to_mlflow
6. compare_to_production
7. promote_if_better
```

---

# 30. Scheduling

Initial production schedule:

```text
Injuries:
Every 2–3 hours

Odds:
Every 1 hour

Weather:
Every 3–6 hours

Game/PBP:
Every 30–60 minutes during games

Post-game:
After game completion

Predictions:
Before game kickoff

Model retraining:
Weekly
```

GitHub Actions can initially provide scheduling.

---

# 31. Pipeline Metadata

Create:

```text
metadata.pipeline_runs
```

Columns:

```text
run_id
pipeline_name
started_at
completed_at
status
records_processed
error_message
```

Also create:

```text
metadata.pipeline_state
```

Example:

```text
pipeline_name
last_successful_run
last_processed_timestamp
status
```

This allows incremental ingestion.

---

# 32. Incremental Ingestion

The ingestion system should avoid reprocessing the entire dataset.

If the API supports timestamps:

```text
last_successful_timestamp
        |
        v
API request for changed data
```

If not:

```text
Pull recent time window
        |
        v
Deduplicate
        |
        v
UPSERT/MERGE
```

PostgreSQL should use appropriate unique constraints and indexes.

---

# 33. Database Indexing

Important indexes:

```sql
CREATE INDEX idx_games_season_week
ON silver.games(season, week);

CREATE INDEX idx_plays_game
ON silver.plays(game_id);

CREATE INDEX idx_player_stats_player
ON silver.player_game_stats(player_id);

CREATE INDEX idx_injuries_player
ON silver.injuries(player_id);

CREATE INDEX idx_odds_game_timestamp
ON silver.odds(game_id, timestamp);

CREATE INDEX idx_predictions_game
ON ml.predictions(game_id);
```

---

# 34. Application Architecture

The frontend should NOT directly access the data warehouse.

Architecture:

```text
React
  |
  v
FastAPI
  |
  v
PostgreSQL
```

Example endpoints:

```text
GET /games
GET /games/{game_id}
GET /games/week/{week}
GET /predictions/{game_id}
GET /teams/{team_id}
GET /players/{player_id}
GET /model/performance
GET /model/explanation/{game_id}
```

---

# 35. Prediction UI

A game page should show:

```text
Kansas City Chiefs
       vs
Buffalo Bills

Win Probability

KC 68%
BUF 32%

Predicted Score

KC 27
BUF 23

Spread

Market: KC -2.5
Model: KC -4.2

Cover Probability

KC: 61%

Confidence: Medium
```

---

# 36. Explainability

Use SHAP or another explainability method.

Example:

```text
Why does the model favor KC?

+ Home field advantage
+ QB EPA
+ Offensive EPA
+ Defensive EPA
+ Recent performance
- WR injury
- Rest disadvantage
```

The explanation should be generated from actual model features rather than an LLM inventing a narrative.

An LLM can optionally convert structured SHAP results into natural language later.

---

# 37. Monitoring

Monitor:

## Data quality

- Missing games
- Missing players
- Duplicate games
- Invalid scores
- API failures
- Stale injury data

## Pipeline health

- Runtime
- Failure rate
- Records processed
- API response errors

## ML health

- Accuracy
- Brier score
- Calibration
- Spread accuracy
- Prediction distribution
- Feature drift

---

# 38. Testing

Tests should cover:

## Data tests

```text
game_id uniqueness
valid team IDs
valid timestamps
non-negative scores
```

## ETL tests

```text
deduplication
schema validation
incremental loading
```

## Feature tests

Most importantly:

```text
No feature may use information after prediction_timestamp.
```

## ML tests

```text
Model can train
Model outputs valid probabilities
Probabilities sum to 1
Prediction schema is valid
```

---

# 39. 2026 Production Lifecycle

Before Week 1:

```text
Historical data
      ↓
Train initial model
      ↓
2026 preseason information
      ↓
Model v1.0
```

Week 1:

```text
Injuries / odds / weather
      ↓
Features
      ↓
Prediction
      ↓
Freeze prediction
      ↓
Game occurs
      ↓
Ingest result
      ↓
Evaluate
```

Week 2:

```text
Week 1 results
+
Historical data
+
Current injuries
+
Current odds
      ↓
Updated features
      ↓
Model / prediction
```

Repeat through Week 18.

---

# 40. Reinforcement Learning — Future Phase

RL should not be part of V1.

Potential future use cases:

- Dynamic prediction weighting
- Adaptive strategy selection
- Model ensemble weighting
- Contextual bandits
- Decision-making under uncertainty

First establish whether supervised learning actually improves with additional data.

If online learning is introduced, it should be evaluated against the existing supervised-learning baseline.

---

# 41. Future Cloud Architecture

After the local version works:

```text
                        Azure
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
      ADLS          Databricks       PostgreSQL
        |                |                |
        v                v                |
     Bronze           Silver              |
        |                |                |
        +--------------> Gold <------------+
                         |
                         v
                     MLflow
                         |
                         v
                    Model Registry
                         |
                         v
                     FastAPI
                         |
                         v
                       React
```

Local PostgreSQL should eventually be replaced or supplemented by cloud storage depending on scale.

---

# 42. Development Strategy

Build incrementally.

## Phase 1 — Data ingestion

Build:

```text
API
 ↓
Python
 ↓
PostgreSQL
```

## Phase 2 — Data architecture

Build:

```text
Bronze
 ↓
Silver
 ↓
Gold
```

## Phase 3 — Features

Build:

```text
Gold
 ↓
Point-in-time features
```

## Phase 4 — ML

Build:

```text
Features
 ↓
XGBoost
 ↓
Predictions
```

## Phase 5 — Backtesting

Simulate 2025 week-by-week.

## Phase 6 — Automation

GitHub Actions.

## Phase 7 — Live 2026

Connect live data sources.

## Phase 8 — MLOps

MLflow + model registry + monitoring.

## Phase 9 — Application

FastAPI + React.

## Phase 10 — Advanced learning

Experiment with online learning / RL.

---

# 43. Initial Success Criteria

The MVP is considered successful when:

1. Historical NFL data can be ingested automatically.
2. Data is stored in PostgreSQL.
3. Bronze/Silver/Gold layers are implemented.
4. Feature engineering is automated.
5. Point-in-time correctness is enforced.
6. The model predicts game winners.
7. The model predicts scores.
8. The model predicts spread coverage probabilities.
9. Every prediction is stored with a timestamp and model version.
10. Actual results are automatically matched to predictions.
11. Model performance is automatically calculated.
12. A 2025 walk-forward simulation works end-to-end.
13. The same pipeline can be switched to live 2026 data.
14. The system can run without the developer's computer being turned on.
15. The initial production infrastructure costs approximately $0–$10/month.

---

# 44. Engineering Principles

The implementation should prioritize:

### Reproducibility

Every prediction should be reproducible from:

```text
model version
feature snapshot
prediction timestamp
training data version
```

### Point-in-time correctness

Never allow future information into a prediction.

### Incremental processing

Don't repeatedly process all historical data.

### Separation of concerns

Keep:

```text
ingestion
transformation
features
ML
API
frontend
```

separate.

### Replaceable infrastructure

The system should eventually be migratable from:

```text
Local PostgreSQL
```

to:

```text
Azure + Databricks + Delta Lake
```

without rewriting the core business logic.

### Cost efficiency

Don't introduce expensive infrastructure unless the system requires it.

---

# 45. Final Architecture Philosophy

The project should be treated as a **data platform first and a prediction model second**.

The most important pipeline is:

```text
External Data
     ↓
Reliable Ingestion
     ↓
Raw Data
     ↓
Clean Data
     ↓
Analytics Data
     ↓
Point-in-Time Features
     ↓
ML Model
     ↓
Prediction
     ↓
Actual Result
     ↓
Evaluation
     ↓
New Training Data
     ↓
Improved Model
```

The ultimate goal is a system that can operate continuously throughout the 2026 NFL season with minimal manual intervention.