"""Per-horizon HAR3 variance forecasts, h = 1..30 min, frozen Dec-Aug fit (and expanding monthly)."""
import numpy as np, pandas as pd, sys
W = sys.argv[1]
D = pd.read_parquet(W + '/vol_minutes.parquet'); D = D[D.month >= '2025-12']
EPS = 1e-14; L = lambda x: np.log(np.maximum(x, EPS))
cols = ['rv15', 'rv60', 'rv1440']
need = D.dropna(subset=cols + ['s100', 'ewma', 'atr14'])
X = np.column_stack([np.ones(len(need))] + [L(need[c].to_numpy()) for c in cols])
MONTHS = sorted(D.month.unique()); DA = [m for m in MONTHS if m < '2026-09']
out = pd.DataFrame({'t': need.t.to_numpy()}); outx = out.copy(); atr_c = {}; atr_cx = pd.DataFrame({'t': need.t.to_numpy()})
mon = need.month.to_numpy()
for h in range(1, 31):
    y = need[f'fwd{h}'].to_numpy(); ok = np.isfinite(y) & (y > 0)
    fit = ok & np.isin(mon, DA)
    b, *_ = np.linalg.lstsq(X[fit], L(y[fit]), rcond=None); s2 = (L(y[fit]) - X[fit] @ b).var()
    out[f'h{h}'] = np.exp(X @ b + s2 / 2)
    atr_c[h] = float(np.mean(y[fit] / need.atr14.to_numpy()[fit] ** 2))
    # expanding
    e = np.full(len(need), np.nan)
    for M in MONTHS[1:]:
        f = ok & (mon < M); a = mon == M
        if a.sum() == 0: continue
        b, *_ = np.linalg.lstsq(X[f], L(y[f]), rcond=None); s2 = (L(y[f]) - X[f] @ b).var()
        e[a] = np.exp(X[a] @ b + s2 / 2)
    outx[f'h{h}'] = e
out.to_parquet(W + '/fc_hz_frozen.parquet'); outx.to_parquet(W + '/fc_hz_expanding.parquet')
pd.Series(atr_c).to_json(W + '/atr_scale.json')
print('done', len(out)); print(pd.Series(atr_c).round(2).to_dict())
