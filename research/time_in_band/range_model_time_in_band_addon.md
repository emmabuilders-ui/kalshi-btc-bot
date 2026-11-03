# Range model + time-in-band add-on (fitted 2026-11-03, NOT used by any bot)

The current range model (`RangeModel`, range_cdf.json) is unchanged. This is an optional layer on top of it.

Inputs for a band [lo, hi] at minute mark t (all from 1-second BRTI before t, no look-ahead):
- h  = share of seconds this hour so far (hour start to t) with lo <= BRTI <= hi
- ph = share of the previous hour with BRTI inside the band
- q  = share of the last 15 minutes inside the band
- Lm = logit(current model YES chance) = ln(p / (1 - p))

Formula (logistic, fitted on Dec 2025 - Aug 2026 band-minutes, last 30 min, 14 nearest bands, spread <= 5c):

    z = 0.0517 + 1.0786*Lm - 0.2418*h - 0.0071*ph + 0.3802*q + 0.0642*h*Lm + 0.0262*ph*Lm - 0.0828*q*Lm
    P(YES) = 1 / (1 + exp(-z))

Exact coefficients: range_model_time_in_band_addon.json ("time_in_band_addon").
Recalibration alone (no time-in-band): z = intercept + b*Lm, also in the json ("recalibration_only").

Brier (lower is better):
| | Dec-Aug | Sep-Oct |
|---|---|---|
| Kalshi mid | 0.05780 | 0.05233 |
| Current model | 0.05814 | 0.05268 |
| Model recalibrated | 0.05799 | 0.05256 |
| Model + time in band | 0.05794 | 0.05253 |

Full tables: strategy/range_time_in_band_v2.xlsx.
