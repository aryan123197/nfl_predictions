/**
 * Derived, display-ready views of a Game. Pure functions, no fetching.
 *
 * Everything here obeys the same rule as format.ts: a null never becomes a
 * number. A game with no prediction yields `prob: null`, `edge: null` and
 * tier "none" -- the UI renders an absence rather than a 50%.
 */

import type { Game } from "../../types";
import { isMissing, pct, spread } from "../format";

export type Tier = "strong" | "lean" | "none";

export interface DerivedGame {
  game: Game;
  /** Model home win probability, or null when unpredicted. */
  prob: number | null;
  /** Team code the model favours; null without a probability. */
  favCode: string | null;
  favPct: string;
  awayWidth: string;
  homeWidth: string;
  /** Market line actually priced against, falling back to the current line. */
  market: number | null;
  marketLabel: string;
  modelSpreadLabel: string;
  /** predicted_margin - market, in points. Positive = model likes home more. */
  edge: number | null;
  edgeLabel: string;
  tier: Tier;
  tierClass: string;
  tierLabel: string;
  predScore: string;
  predCompact: string;
  isFinal: boolean;
  resultLabel: string;
}

export function tierOf(edge: number | null): Tier {
  if (edge === null) return "none";
  const a = Math.abs(edge);
  return a >= 2 ? "strong" : a >= 1 ? "lean" : "none";
}

export function derive(game: Game): DerivedGame {
  const p = game.prediction;
  const home = p?.home_win_probability ?? null;
  const prob = isMissing(home) ? null : (home as number);

  const market = !isMissing(p?.market_spread)
    ? (p!.market_spread as number)
    : isMissing(game.features.current_spread)
    ? null
    : (game.features.current_spread as number);

  const margin = isMissing(p?.predicted_margin) ? null : (p!.predicted_margin as number);
  const edge = margin !== null && market !== null ? margin - market : null;
  const tier = tierOf(edge);

  const homeFav = prob !== null && prob >= 0.5;
  const isFinal = game.status === "final";

  return {
    game,
    prob,
    favCode: prob === null ? null : homeFav ? game.home_team_id : game.away_team_id,
    favPct: prob === null ? "—" : pct(homeFav ? prob : 1 - prob),
    awayWidth: prob === null ? "0%" : `${((1 - prob) * 100).toFixed(1)}%`,
    homeWidth: prob === null ? "0%" : `${(prob * 100).toFixed(1)}%`,
    market,
    marketLabel: spread(market, game.home_team_id, game.away_team_id),
    modelSpreadLabel: spread(margin, game.home_team_id, game.away_team_id),
    edge,
    edgeLabel: edge === null ? "—" : `${edge > 0 ? "+" : ""}${edge.toFixed(1)}`,
    tier,
    tierClass: tier === "strong" ? "tier-strong" : tier === "lean" ? "tier-lean" : "tier-none",
    tierLabel: tier === "strong" ? "Strong" : tier === "lean" ? "Lean" : "No edge",
    predScore:
      isMissing(p?.predicted_away_score) || isMissing(p?.predicted_home_score)
        ? "—"
        : `${game.away_team_id} ${(p!.predicted_away_score as number).toFixed(1)} – ${game.home_team_id} ${(p!.predicted_home_score as number).toFixed(1)}`,
    predCompact:
      isMissing(p?.predicted_away_score) || isMissing(p?.predicted_home_score)
        ? "—"
        : `${(p!.predicted_away_score as number).toFixed(1)}–${(p!.predicted_home_score as number).toFixed(1)}`,
    isFinal,
    resultLabel:
      isFinal && !isMissing(game.away_score) && !isMissing(game.home_score)
        ? `${game.away_score}–${game.home_score} F`
        : "",
  };
}

/** Kickoff, short form: "Sun 1:00p". Falls back to the em-dash on a null date. */
export function kickoff(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso.slice(0, 10);
  const day = d.toLocaleDateString(undefined, { weekday: "short" });
  const time = d
    .toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })
    .replace(/\s?(AM|PM)/i, (m) => m.trim().toLowerCase()[0]);
  return `${day} ${time}`;
}

/** The n games whose model number disagrees most with the market. */
export function featuredGames(games: DerivedGame[], n = 3): DerivedGame[] {
  return games
    .filter((d) => d.edge !== null)
    .sort((a, b) => Math.abs(b.edge as number) - Math.abs(a.edge as number))
    .slice(0, n);
}
