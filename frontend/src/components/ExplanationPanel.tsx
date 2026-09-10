import { api } from "../api";
import type { PredictionExplanation } from "../types";
import { useAsync } from "./useAsync";

interface ExplanationPanelProps {
  gameId: string;
  homeTeamId: string;
  awayTeamId: string;
}

export default function ExplanationPanel({
  gameId,
  homeTeamId,
  awayTeamId,
}: ExplanationPanelProps) {
  const { data: exp, loading, error, status } = useAsync<PredictionExplanation>(
    () => api.explanation(gameId),
    [gameId],
  );

  if (loading) {
    return (
      <section className="card panel explanation-panel">
        <h2>Model Explainability</h2>
        <p className="panel-note">Computing TreeSHAP feature attributions…</p>
        <div className="skeleton" style={{ height: 120 }} />
      </section>
    );
  }

  if (status === 404 || error || !exp) {
    return null; // Degrade gracefully if game has no explanation or model not ready
  }

  const maxContrib = Math.max(
    ...exp.top_positive_factors.map((f) => Math.abs(f.contribution)),
    ...exp.top_negative_factors.map((f) => Math.abs(f.contribution)),
    0.1,
  );

  return (
    <section className="card panel explanation-panel">
      <div className="explanation-header">
        <div>
          <h2>Model Explainability & Key Drivers</h2>
          <p className="panel-note">
            TreeSHAP feature attributions explaining why the model gives{" "}
            <strong>{exp.favored_team}</strong> a{" "}
            <strong>{(exp.win_probability * 100).toFixed(1)}%</strong> win probability.
          </p>
        </div>
        <div className="base-baseline">
          <span className="base-label">Baseline Prior</span>
          <span className="base-value">{(exp.base_probability * 100).toFixed(1)}%</span>
        </div>
      </div>

      <div className="factors-container">
        {/* Positive Factors (Favor Home) */}
        <div className="factor-column">
          <h3 className="factor-col-title favor-home">
            ▲ Factors Favoring {homeTeamId}
          </h3>
          {exp.top_positive_factors.length === 0 ? (
            <div className="factor-empty">No dominant positive factors</div>
          ) : (
            exp.top_positive_factors.map((factor) => {
              const widthPct = Math.min(100, Math.round((Math.abs(factor.contribution) / maxContrib) * 100));
              return (
                <div key={factor.feature} className="factor-card">
                  <div className="factor-info">
                    <span className="factor-name">{factor.display_name}</span>
                    <span className="factor-contrib positive">
                      +{factor.contribution.toFixed(3)}
                    </span>
                  </div>
                  <div className="factor-desc">{factor.description}</div>
                  <div className="factor-bar-wrapper">
                    <div
                      className="factor-bar favor-home-bar"
                      style={{ width: `${widthPct}%` }}
                    />
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Negative Factors (Favor Away) */}
        <div className="factor-column">
          <h3 className="factor-col-title favor-away">
            ▼ Factors Favoring {awayTeamId}
          </h3>
          {exp.top_negative_factors.length === 0 ? (
            <div className="factor-empty">No dominant negative factors</div>
          ) : (
            exp.top_negative_factors.map((factor) => {
              const widthPct = Math.min(100, Math.round((Math.abs(factor.contribution) / maxContrib) * 100));
              return (
                <div key={factor.feature} className="factor-card">
                  <div className="factor-info">
                    <span className="factor-name">{factor.display_name}</span>
                    <span className="factor-contrib negative">
                      {factor.contribution.toFixed(3)}
                    </span>
                  </div>
                  <div className="factor-desc">{factor.description}</div>
                  <div className="factor-bar-wrapper">
                    <div
                      className="factor-bar favor-away-bar"
                      style={{ width: `${widthPct}%` }}
                    />
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </section>
  );
}
