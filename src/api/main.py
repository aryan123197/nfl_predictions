"""
FastAPI application -- design doc sections 34-35 (Phase 9).

Read-only HTTP surface over silver + gold, so the React frontend never talks
to the warehouse directly (section 34's explicit requirement). Endpoint names
follow section 34's list.

Run locally:
    pip install -r requirements-api.txt
    uvicorn src.api.main:app --reload
    # interactive docs at http://127.0.0.1:8000/docs

Not implemented yet, and why:
    GET /players/{player_id}            player-level features are out of
                                        scope until a player-features slice
                                        exists (design doc section 15).
    GET /model/explanation/{game_id}    SHAP explanations need a trained
                                        model (Phase 4) and the
                                        explainability work in Phase 8.
Both return 501 rather than 404 -- the route is real and specified, it just
has nothing behind it yet, and a 404 would suggest the endpoint was dropped.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api import predictions as predictions_repo
from src.api import queries, schemas

app = FastAPI(
    title="NFL Predict API",
    description="Read-only API over the silver/gold warehouse for the prediction UI.",
    version="0.1.0",
)

# The Vite dev server runs on a different origin (5173) than uvicorn (8000),
# so local development needs CORS. Scoped to localhost dev origins rather
# than "*" -- in a real deployment the frontend is served from the same
# origin and this list should be replaced by that origin explicitly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _to_game(row: dict, prediction: Optional[dict]) -> schemas.Game:
    """Split one joined DB row into the nested Game/GameFeatures shape.

    The query returns a flat row (one JOIN); the API returns features nested
    under the game, so the frontend gets "what the model sees" as one object
    rather than twenty sibling keys it has to know how to group.
    """
    features = schemas.GameFeatures(**{k: row.get(k) for k in schemas.GameFeatures.model_fields})
    game_date = row.get("game_date")
    return schemas.Game(
        game_id=row["game_id"],
        season=row["season"],
        week=row.get("week"),
        # game_date is DATE on Postgres and TEXT on SQLite (see db.py's DDL
        # translation), so normalize to an ISO string at the boundary rather
        # than letting the dialect leak into the JSON.
        game_date=str(game_date) if game_date is not None else None,
        status=row["status"],
        home_team_id=row["home_team_id"],
        away_team_id=row["away_team_id"],
        home_score=row.get("home_score"),
        away_score=row.get("away_score"),
        home_rest_days=row.get("home_rest_days"),
        away_rest_days=row.get("away_rest_days"),
        features=features,
        prediction=_to_prediction(prediction),
    )


def _to_prediction(row: Optional[dict]) -> Optional[schemas.Prediction]:
    if not row:
        return None
    data = dict(row)
    timestamp = data.get("prediction_timestamp")
    data["prediction_timestamp"] = str(timestamp) if timestamp is not None else None
    return schemas.Prediction(**data)


@app.get("/health", response_model=schemas.Health, tags=["meta"])
def health() -> schemas.Health:
    ready = queries.warehouse_ready()
    return schemas.Health(
        status="ok",
        warehouse_ready=ready,
        predictions_available=predictions_repo.predictions_available(),
        # Guarded: querying seasons before the pipeline has ever run would
        # hit a missing table.
        seasons=queries.list_seasons() if ready else [],
    )


@app.get("/games", response_model=list[schemas.Game], tags=["games"])
def get_games(
    season: Optional[int] = Query(None, description="Filter to one season"),
    week: Optional[int] = Query(None, description="Filter to one week"),
) -> list[schemas.Game]:
    rows = queries.list_games(season=season, week=week)
    preds = predictions_repo.get_predictions_for_games([r["game_id"] for r in rows])
    return [_to_game(r, preds.get(r["game_id"])) for r in rows]


@app.get("/games/week/{week}", response_model=list[schemas.Game], tags=["games"])
def get_games_for_week(
    week: int,
    season: int = Query(..., description="Required: week numbers repeat every season"),
) -> list[schemas.Game]:
    rows = queries.list_games(season=season, week=week)
    preds = predictions_repo.get_predictions_for_games([r["game_id"] for r in rows])
    return [_to_game(r, preds.get(r["game_id"])) for r in rows]


@app.get("/games/{game_id}", response_model=schemas.Game, tags=["games"])
def get_game(game_id: str) -> schemas.Game:
    row = queries.get_game(game_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No game with id {game_id!r}")
    return _to_game(row, predictions_repo.get_prediction(game_id))


@app.get("/predictions/{game_id}", response_model=schemas.Prediction, tags=["predictions"])
def get_prediction(game_id: str) -> schemas.Prediction:
    row = predictions_repo.get_prediction(game_id)
    if row is None:
        # 404 is correct for both "no model yet" and "this game is
        # unpredicted": the resource genuinely does not exist. The detail
        # says which, so a client can tell a missing model from a missing
        # game without guessing.
        detail = (
            "No predictions table yet -- model training (Phase 4) has not landed."
            if not predictions_repo.predictions_available()
            else f"No prediction recorded for game {game_id!r}."
        )
        raise HTTPException(status_code=404, detail=detail)
    return _to_prediction(row)


@app.get("/seasons", response_model=list[int], tags=["meta"])
def get_seasons() -> list[int]:
    return queries.list_seasons() if queries.warehouse_ready() else []


@app.get("/seasons/{season}/weeks", response_model=list[int], tags=["meta"])
def get_weeks(season: int) -> list[int]:
    return queries.list_weeks(season)


@app.get("/teams", response_model=list[schemas.Team], tags=["teams"])
def get_teams() -> list[schemas.Team]:
    return [schemas.Team(**t) for t in queries.list_teams()]


@app.get("/teams/{team_id}", response_model=schemas.Team, tags=["teams"])
def get_team(team_id: str) -> schemas.Team:
    team = queries.get_team(team_id)
    if team is None:
        raise HTTPException(status_code=404, detail=f"No team with id {team_id!r}")
    rating = queries.get_team_rating(team_id)
    # elo_post, not elo_pre: the rating AFTER the team's most recent rated
    # game is its current strength. elo_pre would be one game stale.
    return schemas.Team(**team, elo=rating.get("elo_post") if rating else None)


@app.get("/model/performance", response_model=schemas.ModelPerformance, tags=["model"])
def model_performance() -> schemas.ModelPerformance:
    # Deliberately reports unavailability rather than zeros. Phase 4/5 will
    # fill this from evaluated predictions vs. actual results (design doc
    # section 23), against the promotion criteria in DECISIONS.md #1.
    if not predictions_repo.predictions_available():
        return schemas.ModelPerformance(
            available=False,
            reason="No model has been trained yet (Phase 4 not landed).",
        )
    return schemas.ModelPerformance(
        available=False,
        reason="Predictions exist but evaluation metrics are not computed yet (Phase 5).",
    )


@app.get("/players/{player_id}", tags=["players"])
def get_player(player_id: str):
    raise HTTPException(
        status_code=501,
        detail="Player-level features are not implemented yet (design doc section 15).",
    )


@app.get("/model/explanation/{game_id}", tags=["model"])
def get_explanation(game_id: str):
    raise HTTPException(
        status_code=501,
        detail="SHAP explanations require a trained model (Phase 4) and Phase 8 explainability work.",
    )
