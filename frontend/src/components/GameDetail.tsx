import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import type { Game } from "../types";
import ExplanationPanel from "./ExplanationPanel";
import FeatureComparison from "./FeatureComparison";
import PredictionPanel from "./PredictionPanel";
import { ErrorState, EmptyState } from "./States";
import { gameDate, isMissing, spread } from "./format";
import { useAsync } from "./useAsync";

export default function GameDetail({ predictionsAvailable }: { predictionsAvailable: boolean }) {
  const { gameId = "" } = useParams();
  const { data: game, error, status, loading } = useAsync<Game>(
    () => api.game(gameId),
    [gameId],
  );

  if (loading) return <div className="card state">Loading…</div>;
  // A 404 is a missing game, not a failure -- say so plainly rather than
  // showing a red error box for a mistyped URL.
  if (status === 404) {
    return <EmptyState title="Game not found" detail={`No game with id ${gameId}.`} />;
  }
  if (error) return <ErrorState message={error} />;
  if (!game) return null;

  const isFinal = game.status === "final";

  return (
    <>
      <Link to=".." relative="path" className="back-link">
        ← All games
      </Link>

      <section className="card matchup-hero">
        <div className="teams">
          <span className="team">{game.away_team_id}</span>
          <span className="vs">at</span>
          <span className="team">{game.home_team_id}</span>
        </div>

        {isFinal && !isMissing(game.away_score) && !isMissing(game.home_score) && (
          <div className="final-score">
            {game.away_score} – {game.home_score}
          </div>
        )}

        <div className="meta">
          {gameDate(game.game_date)} · Season {game.season}
          {game.week !== null ? ` · Week ${game.week}` : ""} ·{" "}
          <span className={`badge ${isFinal ? "final" : "scheduled"}`}>{game.status}</span>
        </div>
        <div className="meta">
          Market: {spread(game.features.current_spread, game.home_team_id, game.away_team_id)}
        </div>
      </section>

      <PredictionPanel game={game} predictionsAvailable={predictionsAvailable} />
      <ExplanationPanel
        gameId={game.game_id}
        homeTeamId={game.home_team_id}
        awayTeamId={game.away_team_id}
      />
      <FeatureComparison game={game} />
    </>
  );
}

