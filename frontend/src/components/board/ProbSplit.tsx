import type { DerivedGame } from "./derive";

/**
 * The two-segment win-probability bar. Away on the left, home on the right --
 * the same order as every other pairing in the app.
 *
 * Renders nothing but an empty track when the game has no prediction, so an
 * unpredicted game never shows a 50/50 split.
 */
export default function ProbSplit({
  d,
  size,
  labels = false,
}: {
  d: DerivedGame;
  size: "sm" | "row" | "lg";
  labels?: boolean;
}) {
  const { game, prob, awayWidth, homeWidth } = d;
  return (
    <div
      className={`prob-split ${size}`}
      role="img"
      aria-label={
        prob === null
          ? "No win probability for this game"
          : `Win probability: ${game.away_team_id} ${d.favCode === game.away_team_id ? d.favPct : ""}, home ${game.home_team_id}`
      }
    >
      {prob !== null && (
        <>
          <i className="away" style={{ width: awayWidth }}>
            {labels && prob <= 0.85 ? game.away_team_id : null}
          </i>
          <i className="home" style={{ width: homeWidth }}>
            {labels && prob >= 0.15 ? game.home_team_id : null}
          </i>
        </>
      )}
    </div>
  );
}
