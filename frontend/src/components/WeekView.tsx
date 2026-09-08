import { useEffect, useState } from "react";
import { api } from "../api";
import type { Game } from "../types";
import GameCard from "./GameCard";
import { EmptyState, ErrorState, Loading } from "./States";
import { useAsync } from "./useAsync";

/**
 * The week list: season/week pickers plus a grid of games.
 *
 * Both pickers are populated from the API (/seasons, /seasons/{s}/weeks)
 * rather than a hardcoded range, so the UI can only ever offer weeks this
 * database actually has -- picking a week with no data is impossible by
 * construction instead of being an empty state to explain.
 */
export default function WeekView({ seasons }: { seasons: number[] }) {
  const [season, setSeason] = useState<number | null>(seasons[0] ?? null);
  const [week, setWeek] = useState<number | null>(null);

  const weeksState = useAsync<number[]>(
    () => (season === null ? Promise.resolve([]) : api.weeks(season)),
    [season],
  );
  const weeks = weeksState.data ?? [];

  // Default to the first available week, and re-default when the season
  // changes to one that doesn't contain the currently-selected week.
  useEffect(() => {
    if (weeks.length === 0) return;
    if (week === null || !weeks.includes(week)) setWeek(weeks[0]);
  }, [weeks, week]);

  const gamesState = useAsync<Game[]>(
    () =>
      season === null || week === null
        ? Promise.resolve([])
        : api.gamesForWeek(season, week),
    [season, week],
  );

  if (seasons.length === 0) {
    return (
      <EmptyState
        title="No games ingested yet"
        detail="This database has no games. Run the ingestion and transform pipelines first, e.g. python -m src.ingest.run_ingestion --season 2025 followed by the silver and gold transforms."
      />
    );
  }

  return (
    <>
      <div className="controls">
        <label className="field">
          Season
          <select
            value={season ?? ""}
            onChange={(e) => {
              setSeason(Number(e.target.value));
              // Clear the week so the effect above picks the new season's
              // first week rather than holding a week that may not exist.
              setWeek(null);
            }}
          >
            {seasons.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          Week
          <select
            value={week ?? ""}
            onChange={(e) => setWeek(Number(e.target.value))}
            disabled={weeks.length === 0}
          >
            {weeks.map((w) => (
              <option key={w} value={w}>
                {w}
              </option>
            ))}
          </select>
        </label>

        <span className="spacer" />
        {gamesState.data && (
          <span style={{ fontSize: 13, color: "var(--text-faint)" }}>
            {gamesState.data.length} game{gamesState.data.length === 1 ? "" : "s"}
          </span>
        )}
      </div>

      {gamesState.loading && <Loading />}
      {gamesState.error && <ErrorState message={gamesState.error} />}

      {gamesState.data && gamesState.data.length === 0 && !gamesState.loading && (
        <EmptyState
          title="No games this week"
          detail={`Season ${season} week ${week} has no games in the warehouse.`}
        />
      )}

      {gamesState.data && gamesState.data.length > 0 && (
        <div className="game-grid">
          {gamesState.data.map((game) => (
            <GameCard key={game.game_id} game={game} />
          ))}
        </div>
      )}
    </>
  );
}
