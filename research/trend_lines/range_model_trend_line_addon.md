# Range model + trend line add-on (fitted 2026-11-03, NOT used by any bot)

The current range bot model (`kalshi_btc_bot/range_paper.py`, `RangeModel`, `data/range_cdf.json`) is unchanged.
This file records the optional trend line add-on that was fitted on top of it.

## Current model (unchanged)
For a band [lo, hi] with m minutes left:
- s100 = std of the last 100 one-minute log returns of BRTI (closes ending at t-1); s10 = std of the last 10.
- vol state vb = 0 / 1 / 2 for log(s10/s100) < -0.3 / between / > 0.3.
- time bucket = 30 if m > 20, 15 if m > 8, else 5.
- scale = s100 * sqrt(m - 0.5)
- P(YES) = F(ln((hi + 0.01) / P) / scale) - F(ln(lo / P) / scale), where F = empirical CDF of standardized
  returns for (time bucket, vb) from range_cdf.json and P = BRTI now.

## Add-on: shift the price in the direction of a recent trend line break
    P' = P * exp(k * d * s100 * sqrt(m - 0.5))
then use P' in place of P in the formula above.
- d = +1 if the most recent trend line break was a falling resistance line broken UP,
  -1 if a rising support line broken DOWN, 0 if no break in the last W minutes.
- Best version (lowest Brier on Dec 2025 - Aug 2026): swing size N = 300 s, W = 15 min, k = -0.02.
  (k is in units of the model's own move size; -0.02 = a 2% fade of the break, effectively zero.)
- Every other swing size / window fitted k = 0 (no shift) or -0.02.

## Trend line definition (same as the 15-min trend line tests)
- Swing low at second i: BRTI[i] is the lowest of i-N .. i+N; known at i+N. Swing highs mirror.
- Support = line through the latest swing low and an earlier, lower swing low (up to 10 back, within 4 h)
  with no print more than $15 below the line between them; the anchor with the most touches (within $15) wins,
  longest on ties. Resistance mirrors with falling highs.
- Break = first print $5 beyond the line after the latest swing point is confirmed.

## Result (band-minutes in the last 30 min, 14 bands nearest the T-30 price, spread <= 5c)
| | Dec-Aug Brier | Sep-Oct Brier |
|---|---|---|
| Kalshi mid | 0.05780 | 0.05233 |
| Current model | 0.05814 | 0.05268 |
| Current model + trend line add-on | 0.05814 | 0.05268 |

Conclusion: trend lines add nothing to the range model. Per-cent scores: strategy/range_model_trend_line_fit.xlsx,
chart strategy/range_model_trend_line_by_price.png.
