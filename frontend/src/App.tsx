import { useState } from "react";
import { Link, Route, Routes, useLocation } from "react-router-dom";
import { api } from "./api";
import GameDetail from "./components/GameDetail";
import PerformanceModal from "./components/PerformanceModal";
import WeekBoard from "./components/board/WeekBoard";
import { ErrorState } from "./components/States";
import { useAsync } from "./components/useAsync";
import type { Health, SystemHealthAudit } from "./types";

/**
 * App shell.
 *
 * /health is fetched once here and its result threaded down, because two
 * facts from it shape the whole UI: which seasons exist (drives the pickers)
 * and whether any model exists at all (drives how an absent prediction is
 * explained).
 */
export default function App() {
  const [showPerfModal, setShowPerfModal] = useState(false);
  // The board renders its own masthead (health pill + performance button),
  // so the shell's header would be a second one on the index route.
  const onBoard = useLocation().pathname === "/";
  const { data: health, error, loading } = useAsync<Health>(() => api.health(), []);
  const { data: sysHealth } = useAsync<SystemHealthAudit>(
    () => api.systemHealth(),
    [],
  );

  return (
    <div className="shell">
      {!onBoard && (
        <header className="masthead">
          <h1>
            <Link to="/">NFL Predict</Link>
          </h1>
          <span className="sub">point-in-time game predictions</span>
          <span className="spacer" />

          <div className="header-actions">
            {sysHealth && (
              <span
                className={`health-pill ${
                  sysHealth.status === "healthy"
                    ? "healthy"
                    : sysHealth.status === "warning"
                    ? "warning"
                    : "error"
                }`}
                title={`${sysHealth.passed_count} checks passed, ${sysHealth.warnings_count} warnings, ${sysHealth.errors_count} errors`}
              >
                <span className="dot" />
                {sysHealth.status === "healthy" ? "Pipelines Healthy" : `Health: ${sysHealth.status}`}
              </span>
            )}

            <button
              className="perf-btn"
              onClick={() => setShowPerfModal(true)}
              title="View production model holdout and evaluation metrics"
            >
              📊 Model Performance
            </button>
          </div>
        </header>
      )}

      {loading && <div className="card state">Connecting to the API…</div>}
      {error && <ErrorState message={error} />}

      {health && !health.warehouse_ready && (
        <ErrorState message="The API is up, but this database has no warehouse tables. Run the ingestion pipeline first." />
      )}

      {health?.warehouse_ready && (
        <Routes>
          <Route
            path="/"
            element={
              <WeekBoard
                seasons={health.seasons}
                onOpenPerformance={() => setShowPerfModal(true)}
              />
            }
          />
          <Route
            path="/games/:gameId"
            element={<GameDetail predictionsAvailable={health.predictions_available} />}
          />
        </Routes>
      )}

      {showPerfModal && <PerformanceModal onClose={() => setShowPerfModal(false)} />}
    </div>
  );
}

