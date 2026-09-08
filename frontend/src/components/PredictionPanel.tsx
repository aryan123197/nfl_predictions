import type { Game } from "../types";
import { isMissing, num, pct, signed, spread } from "./format";
import { InlineEmpty } from "./States";

/**
 * Design doc section 35's prediction UI: win probability, predicted score,
 * model spread vs market, and the model's edge.
 *
 * When no prediction exists this renders an explicit explanation instead of
 * the panel. That is the single most important behaviour in this component:
 * a prediction UI showing invented numbers is worse than one showing none,
 * because a reader cannot tell a real forecast from a placeholder.
 */
export default function PredictionPanel({
  game,
  predictionsAvailable,
}: {
  game: Game;
  predictionsAvailable: boolean;
}) {
  const p = game.prediction;

  if (!p) {
    return (
      <section className="card panel">
        <h2>Prediction</h2>
        {predictionsAvailable ? (
          <InlineEmpty
            title="Not predicted"
            detail="A model exists, but it hasn't produced a prediction for this game yet."
          />
        ) : (
          <InlineEmpty
            title="No model yet"
            detail="Model training (Phase 4) hasn't landed, so nothing has predicted this game. Numbers will appear here once a model writes to ml.predictions — none are shown in the meantime."
          />
        )}
      </section>
    );
  }

  const home = p.home_win_probability;
  const away = p.away_win_probability;
  const hasProb = !isMissing(home) && !isMissing(away);

  // Market spread from the prediction row if the model recorded the line it
  // priced against; otherwise the current line from gold features.
  const market = !isMissing(p.market_spread) ? p.market_spread : game.features.current_spread;

  // "Edge" = how far the model's expected home margin sits from the
  // market's. Both are stated in the same units and the same direction
  // (positive = home favoured -- see the sign convention documented and
  // empirically verified in format.ts::spread), so this is a plain
  // difference with no negation. A positive edge means the model likes the
  // home team more than the market does.
  const edge =
    !isMissing(p.predicted_margin) && !isMissing(market)
      ? (p.predicted_margin as number) - (market as number)
      : null;

  return (
    <section className="card panel">
      <h2>Prediction</h2>
      <p className="panel-note">
        {p.model_version ? `Model ${p.model_version}` : "Model version unknown"}
        {p.prediction_timestamp ? ` · predicted ${p.prediction_timestamp.slice(0, 16).replace("T", " ")}` : ""}
      </p>

      {hasProb ? (
        <div style={{ marginBottom: 20 }}>
          {/* Away left, home right -- consistent with the hero, the game
              cards and the feature table. The percentage is hidden inside a
              segment narrower than ~15% (where it would overflow or clip)
              and read from the legend below instead, which always shows
              both. */}
          <div className="prob-bar" role="img"
               aria-label={`Win probability: ${game.away_team_id} ${pct(away)}, ${game.home_team_id} ${pct(home)}`}>
            <div className="seg away" style={{ width: `${(away as number) * 100}%` }}>
              {(away as number) >= 0.15 && pct(away)}
            </div>
            <div className="seg home" style={{ width: `${(home as number) * 100}%` }}>
              {(home as number) >= 0.15 && pct(home)}
            </div>
          </div>
          <div className="prob-legend">
            <span>
              {game.away_team_id} {pct(away)}
            </span>
            <span>
              {game.home_team_id} {pct(home)} <span style={{ color: "var(--text-faint)" }}>(home)</span>
            </span>
          </div>
        </div>
      ) : (
        <InlineEmpty
          title="No win probability"
          detail="This prediction row has no win probabilities recorded."
        />
      )}

      <div className="kv-grid">
        <div className="kv">
          <div className="k">Predicted score</div>
          {/* Away first, matching the hero's final score and every other
              pairing in the app. Team codes are included because a bare
              "23.1 – 27.3" is exactly the kind of value a reader will
              attribute to the wrong side. */}
          <div className="v" style={{ fontSize: 15 }}>
            {isMissing(p.predicted_home_score) && isMissing(p.predicted_away_score) ? (
              "—"
            ) : (
              <>
                {game.away_team_id} {num(p.predicted_away_score)}
                <span style={{ color: "var(--text-faint)" }}> – </span>
                {game.home_team_id} {num(p.predicted_home_score)}
              </>
            )}
          </div>
        </div>
        <div className="kv">
          <div className="k">Model spread</div>
          <div className="v">
            {/* predicted_margin uses the same direction as the market
                spread (positive = home favoured, design doc section 20's
                "+4.2" for a favoured home team), so it needs no negation. */}
            {spread(p.predicted_margin, game.home_team_id, game.away_team_id)}
          </div>
        </div>
        <div className="kv">
          <div className="k">Market spread</div>
          <div className="v">{spread(market, game.home_team_id, game.away_team_id)}</div>
        </div>
        <div className="kv">
          <div className="k">Edge vs market</div>
          <div className="v" style={{ color: edge === null ? undefined : "var(--text)" }}>
            {edge === null ? "—" : `${signed(edge, 1)} pts`}
          </div>
        </div>
      </div>
    </section>
  );
}
