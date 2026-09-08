import type { Game } from "../types";
import { elo, isMissing, num, pct, signed } from "./format";

/**
 * The head-to-head feature table -- what the model actually sees.
 *
 * This is the closest thing the app has to explainability until SHAP lands
 * (design doc section 36), so it shows the real gold.game_features values
 * rather than a summarised narrative.
 *
 * `higherIsBetter` drives which side is emphasised. Defensive EPA is the
 * one inverted metric: it's EPA *allowed*, so lower is better. Getting that
 * backwards would make good defences look bad, so it's explicit per row
 * rather than inferred.
 */
interface Row {
  label: string;
  home: number | null;
  away: number | null;
  format: (v: number | null) => string;
  higherIsBetter: boolean;
}

export default function FeatureComparison({ game }: { game: Game }) {
  const f = game.features;

  const rows: Row[] = [
    { label: "Elo", home: f.home_elo, away: f.away_elo, format: elo, higherIsBetter: true },
    {
      label: "Off EPA/play",
      home: f.home_off_epa,
      away: f.away_off_epa,
      format: (v) => signed(v, 3),
      higherIsBetter: true,
    },
    {
      label: "Def EPA/play",
      home: f.home_def_epa,
      away: f.away_def_epa,
      format: (v) => signed(v, 3),
      // EPA allowed: lower is a better defence.
      higherIsBetter: false,
    },
    {
      label: "Recent form",
      home: f.home_recent_form,
      away: f.away_recent_form,
      format: (v) => pct(v),
      higherIsBetter: true,
    },
    {
      label: "Injury impact",
      home: f.home_injury_impact,
      away: f.away_injury_impact,
      format: (v) => num(v, 1),
      // Injury impact is negative (points of expected value lost), so a
      // number closer to zero -- i.e. higher -- is the healthier team.
      higherIsBetter: true,
    },
    {
      label: "Rest days",
      home: game.home_rest_days,
      away: game.away_rest_days,
      format: (v) => (isMissing(v) ? "—" : String(v)),
      higherIsBetter: true,
    },
  ];

  const allMissing = rows.every((r) => isMissing(r.home) && isMissing(r.away));

  return (
    <section className="card panel">
      <h2>Features</h2>
      <p className="panel-note">
        Point-in-time: every value uses only games played strictly before this one.
      </p>

      {allMissing ? (
        <div className="empty-inline">
          <strong>No features yet</strong>
          The gold transform hasn&apos;t produced features for this game. Run{" "}
          <code>python -m src.transform.gold_transform</code> for this season.
        </div>
      ) : (
        <>
          {/* Away left, home right -- same order as the hero ("DAL at PHI")
              and the game cards, so a reader never has to re-check which
              column is which. */}
          <div className="stat-row" style={{ paddingBottom: 4 }}>
            <div className="v away" style={{ fontWeight: 700, fontFamily: "inherit" }}>
              {game.away_team_id}
            </div>
            <div className="label" />
            <div className="v home" style={{ fontWeight: 700, fontFamily: "inherit" }}>
              {game.home_team_id}
            </div>
          </div>
          {rows.map((row) => (
            <StatRow key={row.label} row={row} />
          ))}
        </>
      )}
    </section>
  );
}

function StatRow({ row }: { row: Row }) {
  const comparable = !isMissing(row.home) && !isMissing(row.away);
  let homeBetter = false;
  let awayBetter = false;
  if (comparable) {
    const h = row.home as number;
    const a = row.away as number;
    if (h !== a) {
      const homeWins = row.higherIsBetter ? h > a : h < a;
      homeBetter = homeWins;
      awayBetter = !homeWins;
    }
  }

  return (
    <div className="stat-row">
      <div className={`v away ${awayBetter ? "better" : ""} ${isMissing(row.away) ? "null" : ""}`}>
        {row.format(row.away)}
      </div>
      <div className="label">{row.label}</div>
      <div className={`v home ${homeBetter ? "better" : ""} ${isMissing(row.home) ? "null" : ""}`}>
        {row.format(row.home)}
      </div>
    </div>
  );
}
