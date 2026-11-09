import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
S = pd.read_parquet('work/replay.parquet')
eps = 1e-4
lg = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
S['x'] = lg(S.mid)
fit = S.period == 'Dec-Aug'
# 1 Kalshi as is
S['p_kalshi'] = S.mid
# 2 Static calibration: Platt per minutes-left, fit Dec-Aug only
S['p_static'] = np.nan
for m, g in S.groupby('min_left'):
    f = g[g.period == 'Dec-Aug']
    lr = LogisticRegression(C=1e6).fit(f[['x']], f.res)
    S.loc[g.index, 'p_static'] = lr.predict_proba(g[['x']])[:, 1]
# 3 Rolling recalibration: Platt per minutes-left on contracts that CLOSED before the day starts, trailing 30 days; refit each day
S['p_roll'] = np.nan
days = np.sort(S.day.unique())
dts = S.day.values
for d in days:
    d = pd.Timestamp(d); lo = d - pd.Timedelta(days=30)
    tr = S[(S.close_ts < d.timestamp()) & (S.close_ts >= lo.timestamp())]
    if tr.ticker.nunique() < 500: continue
    today = S[S.day == d]
    for m, g in today.groupby('min_left'):
        t = tr[tr.min_left == m]
        if len(t) < 200 or t.res.nunique() < 2: continue
        lr = LogisticRegression(C=1e6).fit(t[['x']], t.res)
        S.loc[g.index, 'p_roll'] = lr.predict_proba(g[['x']])[:, 1]
# 4 Simple vol model: P = Phi(dist / (c*sig15*sqrt(T_eff))), T_eff = secs_left - 40 (60-s average), c fit Dec-Aug
Teff = (S.secs_left - 40).clip(lower=1)
def pv(c, idx=slice(None)):
    z = S.dist[idx] / (c * S.sig15[idx].clip(lower=1e-7) * np.sqrt(Teff[idx]))
    return norm.cdf(z)
r = minimize_scalar(lambda c: np.mean((pv(c, fit) - S.res[fit]) ** 2), bounds=(0.3, 3), method='bounded')
S['p_vol'] = pv(r.x); print('vol model scale c =', round(r.x, 3))
S.to_parquet('work/replay_scored.parquet')
