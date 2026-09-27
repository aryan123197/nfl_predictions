import type { DerivedGame } from "./derive";
import { isMissing } from "../format";

/**
 * How the model did on the most recent completed week.
 *
 * Computed client-side from the previous week's final games, because no
 * per-week performance endpoint exists yet (/model/performance is
 * holdout-wide). If you add GET /model/performance/week/{w}, swap this for
 * that and drop the derivation.
 *
 * Renders nothing when the previous week has no finals -- week 1 of a season
 * has no recap, and an empty bar reading "0-0" would look like a failure.
 */
export default function RecapBar({ week, games }: { week: number | null; games: DerivedGame[] }) {
  const finals = games.filter(
    (d) => d.isFinal && !isMissing(d.game.home_score) && !isMissing(d.game.away_score) && d.prob !== null,
  );
  if (finals.length === 0) return null;

  let su = 0;
  let ats = 0;
  let atsGraded = 0;
  let brierSum = 0;
  let maeSum = 0;
  let maeGraded = 0;

  const chips = finals.map((d) => {
    const g = d.game;
    const homeMargin = (g.home_score as number) - (g.away_score as number);
    const homeWon = homeMargin > 0;
    const pickedHome = (d.prob as number) >= 0.5;
    const hit = homeMargin === 0 ? false : pickedHome === homeWon;
    if (hit) su += 1;
    brierSum += Math.pow((d.prob as number) - (homeWon ? 1 : 0), 2);

    if (d.market !== null) {
      atsGraded += 1;
      const coveredHome = homeMargin > d.market;
      const modelLikesHome = (d.edge ?? 0) > 0;
      if (coveredHome === modelLikesHome) ats += 1;
    }
    const margin = g.prediction?.predicted_margin;
    if (!isMissing(margin)) {
      maeGraded += 1;
      maeSum += Math.abs((margin as number) - homeMargin);
    }

    return {
      id: g.game_id,
      label: `${hit ? "✓" : "✗"} ${pickedHome ? g.home_team_id : g.away_team_id}`,
      hit,
    };
  });

  return (
    <div className="recap-bar">
      <span className="label">Week {week} results</span>
      <span>
        straight up <strong>{su}–{finals.length - su}</strong>
      </span>
      {atsGraded > 0 && (
        <span>
          ATS <strong>{ats}–{atsGraded - ats}</strong>
        </span>
      )}
      <span>
        brier <strong>{(brierSum / finals.length).toFixed(3)}</strong>
      </span>
      {maeGraded > 0 && (
        <span>
          mean margin err <strong>{(maeSum / maeGraded).toFixed(1)} pts</strong>
        </span>
      )}
      <span className="board-spacer" />
      {chips.slice(0, 6).map((c) => (
        <span key={c.id} className={c.hit ? "hit" : "miss"}>
          {c.label}
        </span>
      ))}
    </div>
  );
}
