import { Link, Route, Routes } from "react-router-dom";
import { api } from "./api";
import GameDetail from "./components/GameDetail";
import WeekView from "./components/WeekView";
import { ErrorState } from "./components/States";
import { useAsync } from "./components/useAsync";
import type { Health } from "./types";

/**
 * App shell.
 *
 * /health is fetched once here and its result threaded down, because two
 * facts from it shape the whole UI: which seasons exist (drives the pickers)
 * and whether any model exists at all (drives how an absent prediction is
 * explained). Fetching it per-view would mean each panel independently
 * guessing at why it has no data.
 */
export default function App() {
  const { data: health, error, loading } = useAsync<Health>(() => api.health(), []);

  return (
    <div className="shell">
      <header className="masthead">
        <h1>
          <Link to="/">NFL Predict</Link>
        </h1>
        <span className="sub">point-in-time game predictions</span>
        <span className="spacer" />
        {health && (
          <span className="sub">
            {health.predictions_available ? "model connected" : "no model yet — Phase 4 pending"}
          </span>
        )}
      </header>

      {loading && <div className="card state">Connecting to the API…</div>}
      {error && <ErrorState message={error} />}

      {health && !health.warehouse_ready && (
        <ErrorState message="The API is up, but this database has no warehouse tables. Run the ingestion pipeline first." />
      )}

      {health?.warehouse_ready && (
        <Routes>
          <Route path="/" element={<WeekView seasons={health.seasons} />} />
          <Route
            path="/games/:gameId"
            element={<GameDetail predictionsAvailable={health.predictions_available} />}
          />
        </Routes>
      )}
    </div>
  );
}
