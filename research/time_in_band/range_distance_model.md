# Range "distance + time in band" pricing model (fitted 2026-11-03, NOT used by any bot)

Standalone model for an hourly range band (does not use Kalshi's price or the current range model).

Inputs at minute mark t (1-second BRTI before t):
- move = BRTI * s100 * sqrt(m - 0.5)  (s100 = std of last 100 one-minute log returns, m = minutes left)
- center = (lo + hi + 0.01)/2, half = (hi + 0.01 - lo)/2
- zc = |BRTI - center| / move ; wz = half / move ; logt = ln(m - 0.5)
- dp = min(|BRTI - center| / half, 5)   (distance from center, 1 = band edge)
- in_this_hour, in_prev_hour, in_last15 = share of seconds BRTI was inside the band (this hour so far, previous hour, last 15 min)
- vb1, vb2 = 1 if vol state (10 vs 100 min) is normal / high
- zc2 = zc^2, wz2 = wz^2, zcwz = zc*wz, zc_lt = zc*logt, wz_lt = wz*logt, zc_clip = min(zc,4), wz_clip = min(wz,4)

    z = -0.45405 + (+0.15976)*zc + (+1.99625)*wz + (-0.24547)*logt + (-0.06503)*in_this_hour + (+0.16096)*in_prev_hour + (+0.37780)*in_last15 + (+0.02227)*vb1 + (+0.02033)*vb2 + (-0.12186)*zc2 + (+0.15320)*wz2 + (-0.19655)*zcwz + (+0.33319)*zc_lt + (-0.18142)*wz_lt + (-0.18849)*dp + (-1.91607)*zc_clip + (+0.37399)*wz_clip
    P(YES) = 1 / (1 + exp(-z))

Fit: logistic regression on Dec 2025 - Aug 2026 band-minutes (last 30 min, 14 bands nearest T-30, spread <= 5c).

| Brier (lower = better) | Dec-Aug | Sep-Oct |
|---|---|---|
| Kalshi mid | 0.05780 | 0.05233 |
| Current range model | 0.05813 | 0.05268 |
| This model | 0.05816 | 0.05276 |
| Same inputs, boosted trees (no formula) | 0.05602 | 0.05253 |

Scores by cent and by minutes left: strategy/range_distance_model_scores.xlsx.
