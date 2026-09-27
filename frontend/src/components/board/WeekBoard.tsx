import { useEffect, useMemo, useState } from "react";
import { api } from "../../api";
import type { Game, SystemHealthAudit } from "../../types";
import { EmptyState, ErrorState, Loading } from "../States";
import { useAsync } from "../useAsync";
import { derive, featuredGames } from "./derive";
import FeaturedGameCard from "./FeaturedGameCard";
import GameDrawer from "./GameDrawer";
import RecapBar from "./RecapBar";
import SlateRow from "./SlateRow";
import WeekStrip from "./WeekStrip";
import "../../board.css";

/**
 * Drop-in replacement for WeekView: same data, restructured around the model.
 *
 * Three changes of substance over the old card grid:
 *   1. Win probability is the primary number, shown as a bar on every game.
 *   2. The games where the model most disagrees with the market are promoted
 *      out of the grid; the rest collapse into a dense slate.
 *   3. Selecting a game expands a drill-down in place instead of navigating.
 */
export default function WeekBoard({
  seasons,
  onOpenPerformance,
}: {
  seasons: number[];
  onOpenPerformance?: () => void;
}) {
  const [season, setSeason] = useState<number | null>(seasons[0] ?? null);
  const [week, setWeek] = useState<number | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const { data: sysHealth } = useAsync<SystemHealthAudit>(() => api.systemHealth(), []);

  const weeksState = useAsync<number[]>(
    () => (season === null ? Promise.resolve([]) : api.weeks(season)),
    [season],
  );
  const weeks = weeksState.data ?? [];

  useEffect(() => {
    if (weeks.length === 0) return;
    // Default to the latest week with data, not the first: opening on week 1
    // in December is never what the reader wanted.
    if (week === null || !weeks.includes(week)) setWeek(Math.max(...weeks));
  }, [weeks, week]);

  const gamesState = useAsync<Game[]>(
    () =>
      season === null || week === null
        ? Promise.resolve([])
        : api.gamesForWeek(season, week),
    [season, week],
  );

  // Previous week, for the recap bar.
  const prevWeek = week !== null ? weeks.filter((w) => w < week).pop() ?? null : null;
  const prevState = useAsync<Game[]>(
    () =>
      season === null || prevWeek === null
        ? Promise.resolve([])
        : api.gamesForWeek(season, prevWeek),
    [season, prevWeek],
  );

  const derived = useMemo(() => (gamesState.data ?? []).map(derive), [gamesState.data]);
  const featured = useMemo(() => featuredGames(derived, 3), [derived]);
  const featuredIds = new Set(featured.map((d) => d.game.game_id));
  const rest = derived.filter((d) => !featuredIds.has(d.game.game_id));
  const open = derived.find((d) => d.game.game_id === selected) ?? null;

  // Clear a stale selection when the week changes.
  useEffect(() => setSelected(null), [season, week]);

  if (seasons.length === 0) {
    return (
      <EmptyState
        title="No games ingested yet"
        detail="This database has no games. Run the ingestion and transform pipelines first."
      />
    );
  }

  const status = sysHealth?.status ?? "healthy";

  return (
    <div className="board">
      <header className="board-masthead">
        <h1>NFL Predict</h1>
        <span className="tag">point-in-time model</span>
        <span className="board-spacer" />
        {sysHealth && (
          <span
            className={`board-pill ${status === "healthy" ? "" : status}`}
            title={`${sysHealth.passed_count} passed, ${sysHealth.warnings_count} warnings, ${sysHealth.errors_count} errors`}
          >
            <span className="dot" />
            {status === "healthy" ? "Pipelines healthy" : `Health: ${status}`}
          </span>
        )}
        {onOpenPerformance && (
          <button className="board-btn" onClick={onOpenPerformance}>
            Model performance
          </button>
        )}
      </header>

      <WeekStrip
        season={season}
        seasons={seasons}
        weeks={weeks}
        week={week}
        gameCount={gamesState.data?.length ?? null}
        onSeason={(s) => {
          setSeason(s);
          setWeek(null);
        }}
        onWeek={setWeek}
      />

      <RecapBar week={prevWeek} games={(prevState.data ?? []).map(derive)} />

      {open && <GameDrawer d={open} onClose={() => setSelected(null)} />}

      {gamesState.loading && <Loading />}
      {gamesState.error && <ErrorState message={gamesState.error} />}

      {week !== null && gamesState.data && gamesState.data.length === 0 && !gamesState.loading && (
        <EmptyState
          title="No games this week"
          detail={`Season ${season} week ${week} has no games in the warehouse.`}
        />
      )}

      {featured.length > 0 && (
        <section className="board-section">
          <div className="board-eyebrow">Biggest model–market disagreements</div>
          <div className="featured-grid">
            {featured.map((d) => (
              <FeaturedGameCard key={d.game.game_id} d={d} onOpen={setSelected} />
            ))}
          </div>
        </section>
      )}

      {rest.length > 0 && (
        <section className="board-section">
          <div className="board-eyebrow">Full slate</div>
          <div className="slate-head">
            <span>Kickoff</span>
            <span>Matchup</span>
            <span>Model win probability</span>
            <span>Pred score</span>
            <span className="right">Market</span>
            <span className="right">Edge</span>
          </div>
          {rest.map((d) => (
            <SlateRow key={d.game.game_id} d={d} onOpen={setSelected} />
          ))}
        </section>
      )}
    </div>
  );
}
