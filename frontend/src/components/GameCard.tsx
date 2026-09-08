import { Link } from "react-router-dom";
import type { Game } from "../types";
import { elo, isMissing, pct, spread } from "./format";

/**
 * One game in the week list.
 *
 * Shows the market spread and Elo because those exist today; the win
 * probability line appears only when a model has actually predicted the
 * game. There is deliberately no placeholder percentage -- an unpredicted
 * game simply doesn't show that row.
 */
export default function GameCard({ game }: { game: Game }) {
  const isFinal = game.status === "final";
  const homeWon =
    isFinal && !isMissing(game.home_score) && !isMissing(game.away_score)
      ? (game.home_score as number) > (game.away_score as number)
      : null;

  const prob = game.prediction?.home_win_probability ?? null;

  return (
    <Link
      to={`/games/${encodeURIComponent(game.game_id)}`}
      className="card game-card"
      aria-label={`${game.away_team_id} at ${game.home_team_id}, ${game.status}`}
    >
      <div className="game-card-top">
        <span className={`badge ${isFinal ? "final" : "scheduled"}`}>{game.status}</span>
        <span>{spread(game.features.current_spread, game.home_team_id, game.away_team_id)}</span>
      </div>

      {/* Away first, then home -- the conventional "AWAY at HOME" reading order. */}
      <Row
        team={game.away_team_id}
        score={game.away_score}
        elo={game.features.away_elo}
        won={homeWon === null ? null : !homeWon}
        isFinal={isFinal}
      />
      <Row
        team={game.home_team_id}
        score={game.home_score}
        elo={game.features.home_elo}
        won={homeWon}
        isFinal={isFinal}
      />

      {prob !== null && (
        <div className="game-card-top" style={{ marginTop: 12, marginBottom: 0 }}>
          <span>model</span>
          <span>
            {game.home_team_id} {pct(prob)}
          </span>
        </div>
      )}
    </Link>
  );
}

function Row({
  team,
  score,
  elo: rating,
  won,
  isFinal,
}: {
  team: string;
  score: number | null;
  elo: number | null;
  won: boolean | null;
  isFinal: boolean;
}) {
  // Only dim the loser once a game is final; before that, neither team is
  // "losing" and dimming one would imply a result that doesn't exist.
  const emphasis = won === null ? "" : won ? "winner" : "loser";
  return (
    <div className="matchup-row">
      <span className={`team-name ${emphasis}`}>
        {team} <span style={{ color: "var(--text-faint)", fontWeight: 400, fontSize: 12 }}>{elo(rating)}</span>
      </span>
      <span className={`score ${isFinal ? emphasis : "dim"}`}>
        {isMissing(score) ? "—" : score}
      </span>
    </div>
  );
}
