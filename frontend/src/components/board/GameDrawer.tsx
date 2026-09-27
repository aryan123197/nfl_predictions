import { api } from "../../api";
import type { PredictionExplanation } from "../../types";
import { useAsync } from "../useAsync";
import { num, signed } from "../format";
import { gameDate } from "../format";
import type { DerivedGame } from "./derive";
import ProbSplit from "./ProbSplit";

/**
 * The inline drill-down. Expands in place above the slate rather than
 * navigating away, so the reader keeps the week in view; /games/:gameId
 * still works as a deep link via GameDetail.
 *
 * The factor bars come from /model/explanation. When that call fails or the
 * game has no explanation, the factor column simply isn't rendered -- no
 * invented contributions.
 */
export default function GameDrawer({
  d,
  onClose,
}: {
  d: DerivedGame;
  onClose: () => void;
}) {
  const g = d.game;
  const { data: exp } = useAsync<PredictionExplanation>(
    () => api.explanation(g.game_id),
    [g.game_id],
  );

  const factors = exp
    ? [...exp.top_positive_factors, ...exp.top_negative_factors].slice(0, 6)
    : [];
  const maxAbs = factors.reduce((m, f) => Math.max(m, Math.abs(f.contribution)), 0) || 1;

  return (
    <section className="game-drawer">
      <div className="head">
        <div style={{ flex: 1 }}>
          <h2>
            {g.away_team_id} at {g.home_team_id}
          </h2>
          <div className="meta">
            {gameDate(g.game_date)} · market {d.marketLabel} · model {d.modelSpreadLabel}
          </div>
        </div>
        <button className="board-btn" onClick={onClose}>
          Close
        </button>
      </div>

      <div className="cols">
        <div>
          <div className="prob-legend">
            <span>
              {g.away_team_id} {d.prob === null ? "—" : `${Math.round((1 - d.prob) * 100)}%`}
            </span>
            <span>Win probability</span>
            <span>
              {g.home_team_id} {d.prob === null ? "—" : `${Math.round(d.prob * 100)}%`}
            </span>
          </div>

          <ProbSplit d={d} size="lg" labels />

          <div className="stat-grid">
            <div className="stat">
              <div className="k">Predicted score</div>
              <div className="v">{d.predScore}</div>
            </div>
            <div className="stat">
              <div className="k">Edge vs market</div>
              <div className={`v ${d.tierClass}`}>
                {d.edge === null ? "—" : `${signed(d.edge, 1)} pts`}
              </div>
            </div>
            <div className="stat">
              <div className="k">Confidence</div>
              <div className={`v ${d.tierClass}`}>{d.tierLabel}</div>
            </div>
          </div>

          <p className="note">
            {g.prediction?.model_version
              ? `Model ${g.prediction.model_version}`
              : "Model version unknown"}
            {g.prediction?.prediction_timestamp
              ? ` · predicted ${g.prediction.prediction_timestamp.slice(0, 16).replace("T", " ")}`
              : ""}
          </p>
        </div>

        {factors.length > 0 && (
          <div>
            <div className="board-eyebrow">What moved the number</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
              {factors.map((f) => {
                const w = `${Math.min(100, (Math.abs(f.contribution) / maxAbs) * 100)}%`;
                const positive = f.contribution > 0;
                return (
                  <div className="factor" key={f.feature} title={f.description}>
                    <span className="name">{f.display_name}</span>
                    <span className="track">
                      <span className="lhs">{!positive && <i style={{ width: w }} />}</span>
                      <span className="rhs">{positive && <i style={{ width: w }} />}</span>
                    </span>
                    <span className="val">{num(f.value, 2)}</span>
                  </div>
                );
              })}
            </div>
            <p className="note">
              SHAP contributions, home-team perspective. Baseline{" "}
              {Math.round(exp!.base_probability * 100)}% before features.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
