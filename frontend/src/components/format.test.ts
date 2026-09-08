import { describe, expect, it } from "vitest";
import { MISSING, elo, num, pct, signed, spread } from "./format";

/**
 * These tests exist because of one specific bug: the spread sign convention
 * was implemented backwards, which named the WRONG FAVOURITE on every game
 * in the app while looking completely plausible. It was caught by checking
 * the rendered value against real 2025 results, not by reading the code.
 *
 * The convention below is the empirically verified one (see format.ts), and
 * these cases pin it so a future refactor can't quietly re-invert it.
 */
describe("spread sign convention", () => {
  it("renders a POSITIVE stored spread as the home team favoured", () => {
    // Verified against real data: GB (home) -3.5 vs WAS, GB won 27-18.
    expect(spread(3.5, "GB", "WAS")).toBe("GB -3.5");
  });

  it("renders a NEGATIVE stored spread as the away team favoured", () => {
    expect(spread(-6, "NYJ", "BUF")).toBe("BUF -6.0");
  });

  it("renders the largest real home favourite correctly", () => {
    // BUF at home vs NO, stored +15.5 -- BUF are the favourites, not NO.
    expect(spread(15.5, "BUF", "NO")).toBe("BUF -15.5");
  });

  it("treats a zero spread as a pick'em, not a missing value", () => {
    expect(spread(0, "KC", "BUF")).toBe("PK");
  });

  it("shows missing for a null spread rather than inventing a line", () => {
    expect(spread(null, "KC", "BUF")).toBe(MISSING);
  });
});

/**
 * The other rule worth pinning: null never becomes a number. A prediction UI
 * that renders an unknown value as 0 is actively misleading.
 */
describe("null handling", () => {
  it.each([
    ["num", () => num(null)],
    ["elo", () => elo(null)],
    ["pct", () => pct(null)],
    ["signed", () => signed(null)],
  ])("%s renders null as the missing placeholder, not 0", (_label, fn) => {
    expect(fn()).toBe(MISSING);
  });

  it("renders a real zero as zero, not as missing", () => {
    // The distinction that makes the above meaningful: 0.0 EPA is a real
    // measurement and must not be hidden as "—".
    expect(num(0)).toBe("0.0");
    expect(signed(0)).toBe("0.00");
    expect(pct(0)).toBe("0%");
  });

  it("treats NaN as missing", () => {
    expect(num(NaN)).toBe(MISSING);
  });
});

describe("number formatting", () => {
  it("rounds elo to a whole number", () => {
    expect(elo(1508.43)).toBe("1508");
  });

  it("signs positive values explicitly so direction is unambiguous", () => {
    expect(signed(0.162, 3)).toBe("+0.162");
    expect(signed(-0.061, 3)).toBe("-0.061");
  });

  it("formats a probability as a percentage", () => {
    expect(pct(0.68)).toBe("68%");
  });
});
