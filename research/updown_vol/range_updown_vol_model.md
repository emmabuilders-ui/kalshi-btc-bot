# Range model with up/down volatility split (fitted 2026-11-03, NOT used by any bot)

Same structure as the current range model, with one change in how volatility is measured.

From the last 100 one-minute log returns r (closes ending at t-1):
- s100 = std(r) (current model), s10 = std of the last 10; vol state vb = 0/1/2 for ln(s10/s100) < -0.3 / middle / > 0.3
- up-vol  = sqrt(2 * mean(max(r,0)^2))   (only up-minutes count)
- down-vol = sqrt(2 * mean(min(r,0)^2))  (only down-minutes count)
- te = sqrt(m - 0.5), m = minutes left; time bucket 30 (m > 20), 15 (m > 8), 5

Final move R = ln(settlement / BRTI now), settlement = 60-second BRTI average before the close.
Standardized move: z = R / (up-vol * te) if R >= 0, else R / (down-vol * te).
G = empirical CDF of z per (time bucket, vb), built from every minute 1-30 min before each hourly close, Dec 2025 - Aug 2026
(197,100 samples; sorted values in range_updown_vol_tables.npz, keys ud_<bucket>_<vb>; "sym" keys = same thing with s100).

    P(YES) = G(xh) - G(xl)
    xh = ln((hi + 0.01)/BRTI) / (vol_h * te),  vol_h = up-vol if hi >= BRTI else down-vol
    xl = ln(lo/BRTI) / (vol_l * te),           vol_l = up-vol if lo >= BRTI else down-vol

| Brier (lower = better) | Dec-Aug | Sep-Oct |
|---|---|---|
| Kalshi mid | 0.05780 | 0.05233 |
| Current range model (range_cdf.json) | 0.05814 | 0.05268 |
| Same model, table rebuilt on Dec-Aug ("sym") | 0.05777 | 0.05245 |
| Up/down split ("ud") | 0.05780 | 0.05248 |

Scores by cent and minutes left: strategy/range_updown_vol_model.xlsx.
