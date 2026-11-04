# Range "distance + above/below" pricing model, no volatility (fitted 2026-11-03, NOT used by any bot)

Inputs at minute mark t (1-second BRTI before t), for band [lo, hi]:
- d = |BRTI - center| / half-width (0 = center, 1 = edge), in 10% bins (0, 10, ... 300+) as separate terms
- same15 / opp15 = share of the last 15 min BRTI spent outside the band on the SAME side it is now / the OPPOSITE side
- same60 / opp60 = the same for the last 60 min
- (version 2 only) minutes left as separate terms, plus d x ln(minutes left - 0.5)

Strict version: P(YES) = 1 / (1 + exp(-(intercept + sum of the terms))). Coefficients in
range_distance_abovebelow_model.json. Count-table version: P(YES) = share of Dec-Aug band-minutes with the same
distance bin and minutes left that settled inside (sheet count_table_DecAug of range_distance_abovebelow_model.xlsx).

| Brier (lower = better) | Dec-Aug | Sep-Oct |
|---|---|---|
| Kalshi mid | 0.05780 | 0.05233 |
| Current range model (uses volatility) | 0.05813 | 0.05268 |
| Strict: distance + above/below | 0.06300 | 0.05666 |
| Distance + above/below + minutes left | 0.06343 | 0.05705 |
| Count table: distance x minutes left | 0.06163 | 0.05578 |
