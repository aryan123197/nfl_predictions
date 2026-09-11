/**
 * The only module that talks to the API.
 *
 * Requests go to /api/*, which the Vite dev server proxies to uvicorn (see
 * vite.config.ts) and which a real deployment would serve from the same
 * origin. Nothing here hardcodes a host, so no build-time API URL is needed.
 */

import type {
  BettingSlate,
  Game,
  Health,
  ModelPerformance,
  PredictionExplanation,
  SystemHealthAudit,
  Team,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function get<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`);
  } catch (cause) {
    // A network-level failure is nearly always "the API isn't running" during
    // development. Say that, rather than surfacing a bare "Failed to fetch"
    // that sends the reader looking for a bug in their data.
    throw new ApiError(
      "Could not reach the API. Is it running? (uvicorn src.api.main:app --reload)",
      0,
    );
  }

  if (!response.ok) {
    // FastAPI puts the useful text in `detail` -- surface it, since those
    // messages deliberately name the missing phase (e.g. "Phase 4 has not
    // landed") rather than being generic.
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // Non-JSON error body; the status line is the best we have.
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

export const api = {
  health: () => get<Health>("/health"),
  seasons: () => get<number[]>("/seasons"),
  weeks: (season: number) => get<number[]>(`/seasons/${season}/weeks`),
  gamesForWeek: (season: number, week: number) =>
    get<Game[]>(`/games/week/${week}?season=${season}`),
  game: (gameId: string) => get<Game>(`/games/${encodeURIComponent(gameId)}`),
  team: (teamId: string) => get<Team>(`/teams/${encodeURIComponent(teamId)}`),
  modelPerformance: () => get<ModelPerformance>("/model/performance"),
  explanation: (gameId: string) =>
    get<PredictionExplanation>(`/model/explanation/${encodeURIComponent(gameId)}`),
  systemHealth: () => get<SystemHealthAudit>("/monitoring/health"),
  bettingRecommendations: (season?: number, week?: number) => {
    const params = new URLSearchParams();
    if (season !== undefined) params.append("season", String(season));
    if (week !== undefined) params.append("week", String(week));
    const qs = params.toString();
    return get<BettingSlate>(`/betting/recommendations${qs ? `?${qs}` : ""}`);
  },
};


