import type { DerivedGame } from "./derive";
import { kickoff } from "./derive";
import { elo } from "../format";
import ProbSplit from "./ProbSplit";

/** One dense row in the full slate. */
export default function SlateRow({
  d,
  onOpen,
}: {
  d: DerivedGame;
  onOpen: (gameId: string) => void;
}) {
  const g = d.game;
  const awayFav = d.favCode === g.away_team_id;
  const homeFav = d.favCode === g.home_team_id;

  return (
    <button
      className="slate-row"
      onClick={() => onOpen(g.game_id)}
      aria-label={`${g.away_team_id} at ${g.home_team_id}, ${g.status}`}
    >
      <span className="num" style={{ color: d.isFinal ? "var(--pos)" : undefined }}>
        {d.isFinal ? "Final" : kickoff(g.game_date)}
      </span>

      <span className="matchup">
        <span className={`code ${awayFav ? "fav" : ""}`}>{g.away_team_id}</span>
        <span className="num xs">{elo(g.features.away_elo)}</span>
        <span className="num xs">at</span>
        <span className={`code ${homeFav ? "fav" : ""}`}>{g.home_team_id}</span>
        <span className="num xs">{elo(g.features.home_elo)}</span>
        {d.resultLabel && <span className="num">{d.resultLabel}</span>}
      </span>

      <span className="prob-cell">
        <ProbSplit d={d} size="row" />
        <span className="fav-pct">{d.favCode ? `${d.favCode} ${d.favPct}` : "not predicted"}</span>
      </span>

      <span className="cell">{d.predCompact}</span>
      <span className="cell right">{d.marketLabel}</span>
      <span className={`cell right ${d.tierClass}`}>{d.edgeLabel}</span>
    </button>
  );
}
