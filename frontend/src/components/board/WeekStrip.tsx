/**
 * Week picker as a strip. Only weeks the API reported for this season are
 * enabled, so selecting a week with no data is impossible by construction --
 * same guarantee the old <select> gave, one click instead of two.
 */
export default function WeekStrip({
  season,
  seasons,
  weeks,
  week,
  gameCount,
  onSeason,
  onWeek,
}: {
  season: number | null;
  seasons: number[];
  weeks: number[];
  week: number | null;
  gameCount: number | null;
  onSeason: (s: number) => void;
  onWeek: (w: number) => void;
}) {
  const maxWeek = weeks.length ? Math.max(...weeks) : 0;
  // Render a full 18-week rail so the season's shape is visible; weeks the
  // warehouse doesn't have are shown disabled rather than hidden.
  const rail = Array.from({ length: Math.max(maxWeek, 18) }, (_, i) => i + 1);

  return (
    <div className="week-strip">
      <label className="season">
        <select
          value={season ?? ""}
          onChange={(e) => onSeason(Number(e.target.value))}
          style={{ background: "none", border: 0, color: "inherit", font: "inherit" }}
        >
          {seasons.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>
      <span className="rule" />

      {rail.map((w) => {
        const available = weeks.includes(w);
        return (
          <button
            key={w}
            className="week-chip"
            aria-current={w === week}
            disabled={!available}
            title={available ? `Week ${w}` : `Week ${w} has no games in the warehouse`}
            onClick={() => onWeek(w)}
          >
            W{w}
          </button>
        );
      })}

      <span className="board-spacer" />
      {gameCount !== null && (
        <span className="count">
          {gameCount} game{gameCount === 1 ? "" : "s"} · week {week}
        </span>
      )}
    </div>
  );
}
