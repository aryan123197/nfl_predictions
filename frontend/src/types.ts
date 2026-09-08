/**
 * Mirrors src/api/schemas.py.
 *
 * Every optional field here is `| null` rather than `?`, deliberately: the
 * API always sends the key, and its value being null is meaningful ("not
 * known"), distinct from a field being absent. Typing them as `?` would let
 * `undefined` and `null` blur together and invite `?? 0` defaults -- which is
 * precisely the substitution this app must never make. A missing Elo is not
 * an Elo of zero.
 */

export interface GameFeatures {
  home_elo: number | null;
  away_elo: number | null;
  elo_difference: number | null;
  home_injury_impact: number | null;
  away_injury_impact: number | null;
  home_recent_form: number | null;
  away_recent_form: number | null;
  home_off_epa: number | null;
  away_off_epa: number | null;
  home_def_epa: number | null;
  away_def_epa: number | null;
  opening_spread: number | null;
  current_spread: number | null;
  spread_movement: number | null;
}

export interface Prediction {
  game_id: string;
  model_version: string | null;
  prediction_timestamp: string | null;
  home_win_probability: number | null;
  away_win_probability: number | null;
  predicted_home_score: number | null;
  predicted_away_score: number | null;
  predicted_margin: number | null;
  market_spread: number | null;
}

export interface Game {
  game_id: string;
  season: number;
  week: number | null;
  game_date: string | null;
  status: string;
  home_team_id: string;
  away_team_id: string;
  home_score: number | null;
  away_score: number | null;
  home_rest_days: number | null;
  away_rest_days: number | null;
  features: GameFeatures;
  /** null means no model has predicted this game -- render an empty state. */
  prediction: Prediction | null;
}

export interface Team {
  team_id: string;
  first_season: number | null;
  last_season: number | null;
  elo: number | null;
}

export interface Health {
  status: string;
  warehouse_ready: boolean;
  predictions_available: boolean;
  seasons: number[];
}

export interface ModelPerformance {
  available: boolean;
  reason: string | null;
  model_version: string | null;
  games_evaluated: number | null;
  accuracy: number | null;
  brier_score: number | null;
  log_loss: number | null;
  ats_accuracy: number | null;
}
