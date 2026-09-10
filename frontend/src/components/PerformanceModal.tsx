import { useEffect } from "react";
import { api } from "../api";
import type { ModelPerformance } from "../types";
import { useAsync } from "./useAsync";

interface PerformanceModalProps {
  onClose: () => void;
}

export default function PerformanceModal({ onClose }: PerformanceModalProps) {
  const { data: perf, loading, error } = useAsync<ModelPerformance>(
    () => api.modelPerformance(),
    [],
  );

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-content card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2>Model Performance & Evaluation</h2>
            <p className="panel-note">
              Holdout out-of-fold and backtested metrics for the production champion model.
            </p>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close modal">
            ✕
          </button>
        </div>

        {loading && <div className="state">Loading model metrics…</div>}
        {error && <div className="state error">{error}</div>}

        {perf && !perf.available && (
          <div className="empty-inline" style={{ marginTop: 12 }}>
            <strong>No Performance Metrics Available</strong>
            {perf.reason || "Model evaluation has not run yet."}
          </div>
        )}

        {perf && perf.available && (
          <>
            <div className="perf-version-bar">
              <span className="perf-badge">Champion</span>
              <span className="perf-model-id">{perf.model_version || "Production"}</span>
              <span className="spacer" />
              <span className="perf-eval-count">
                <strong>{perf.games_evaluated ?? "—"}</strong> games evaluated
              </span>
            </div>

            <div className="perf-grid">
              <div className="perf-metric-card">
                <span className="metric-label">Win/Loss Accuracy</span>
                <span className="metric-value">
                  {perf.accuracy !== null ? `${(perf.accuracy * 100).toFixed(1)}%` : "—"}
                </span>
                <span className="metric-sub">Straight-up winner accuracy</span>
              </div>

              <div className="perf-metric-card">
                <span className="metric-label">Brier Score</span>
                <span className="metric-value">
                  {perf.brier_score !== null ? perf.brier_score.toFixed(4) : "—"}
                </span>
                <span className="metric-sub">Probability calibration (lower is better)</span>
              </div>

              <div className="perf-metric-card">
                <span className="metric-label">Log Loss</span>
                <span className="metric-value">
                  {perf.log_loss !== null ? perf.log_loss.toFixed(4) : "—"}
                </span>
                <span className="metric-sub">Binary cross-entropy loss</span>
              </div>

              <div className="perf-metric-card">
                <span className="metric-label">Margin MAE</span>
                <span className="metric-value">
                  {perf.mae_margin !== null ? `${perf.mae_margin.toFixed(2)} pts` : "—"}
                </span>
                <span className="metric-sub">Mean absolute error on game margin</span>
              </div>

              <div className="perf-metric-card">
                <span className="metric-label">Against the Spread (ATS)</span>
                <span className="metric-value">
                  {perf.ats_accuracy !== null ? `${(perf.ats_accuracy * 100).toFixed(1)}%` : "—"}
                </span>
                <span className="metric-sub">Pick accuracy vs closing spread</span>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
