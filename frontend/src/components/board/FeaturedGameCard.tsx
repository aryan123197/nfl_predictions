import type { DerivedGame } from "./derive";
import { kickoff } from "./derive";
import { elo } from "../format";
import ProbSplit from "./ProbSplit";

/** One of the top model-vs-market disagreements, shown large. */
export default function FeaturedGameCard({
  d,
  onOpen,
}: {
  d: DerivedGame;
  onOpen: (gameId: string) => void;
}) {
  const g = d.game;
  return (
    <button className="featured-card" onClick={() => onOpen(g.game_id)}>
      <div className="top">
        <span>{d.isFinal ? d.resultLabel : kickoff(g.game_date)}</span>
        <span className={d.tierClass}>
          {d.tierLabel} · {d.edgeLabel}
        </span>
      </div>

      <div className="prob">
        <b>{d.favPct}</b>
        <span>{d.favCode ?? "no model"}</span>
      </div>

      <div className="teams">
        {g.away_team_id} {elo(g.features.away_elo)} at {g.home_team_id} {elo(g.features.home_elo)}
      </div>

      <ProbSplit d={d} size="sm" />

      <div className="foot">
        <span>{d.predCompact}</span>
        <span>mkt {d.marketLabel}</span>
      </div>
    </button>
  );
}
