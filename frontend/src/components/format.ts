/**
 * Formatting helpers.
 *
 * The rule this file exists to enforce: a null NEVER becomes a number. Every
 * formatter returns the em-dash placeholder for null/undefined, so an
 * unknown Elo renders as "—" and can't be mistaken for a real value. This is
 * the UI-layer counterpart to the NaN-handling discipline in the pipeline --
 * same principle, same reason.
 */

export const MISSING = "—";

export const isMissing = (v: number | null | undefined): boolean =>
  v === null || v === undefined || Number.isNaN(v);

export function num(v: number | null | undefined, digits = 1): string {
  return isMissing(v) ? MISSING : (v as number).toFixed(digits);
}

/** Elo is conventionally shown as a whole number. */
export function elo(v: number | null | undefined): string {
  return isMissing(v) ? MISSING : Math.round(v as number).toString();
}

export function pct(v: number | null | undefined, digits = 0): string {
  return isMissing(v) ? MISSING : `${((v as number) * 100).toFixed(digits)}%`;
}

/** Signed, for values where direction matters (EPA, margin, spread). */
export function signed(v: number | null | undefined, digits = 2): string {
  if (isMissing(v)) return MISSING;
  const n = v as number;
  return `${n > 0 ? "+" : ""}${n.toFixed(digits)}`;
}

/**
 * A spread as a bettor reads it: "KC -2.5" (KC favoured by 2.5).
 *
 * SIGN CONVENTION -- verified empirically against all 285 completed 2025
 * games rather than assumed, because getting it backwards silently names
 * the wrong favourite on every single game:
 *
 *   current_spread > 0  =>  HOME team favoured
 *
 * Evidence: corr(current_spread, home_margin) = +0.506; home teams won 66.7%
 * of games with a positive spread vs 35.0% with a negative one; and the
 * largest positive spreads are heavy home favourites (BUF +15.5 at home vs
 * NO). This matches nflverse's `spread_line`, which is stated from the home
 * team's perspective.
 *
 * Display flips the sign, since the favourite is conventionally shown with a
 * minus: a stored +2.5 (home favoured by 2.5) renders as "KC -2.5".
 */
export function spread(
  value: number | null | undefined,
  homeTeam: string,
  awayTeam: string,
): string {
  if (isMissing(value)) return MISSING;
  const n = value as number;
  if (n === 0) return "PK"; // pick'em -- a real outcome, not a missing value
  const favourite = n > 0 ? homeTeam : awayTeam;
  return `${favourite} ${(-Math.abs(n)).toFixed(1)}`;
}

export function gameDate(iso: string | null): string {
  if (!iso) return MISSING;
  // Parse as a plain calendar date: an NFL game's date shouldn't shift a day
  // because the viewer is in a different timezone than the parser's default.
  const [y, m, d] = iso.slice(0, 10).split("-").map(Number);
  if (!y || !m || !d) return iso;
  return new Date(Date.UTC(y, m - 1, d)).toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}
