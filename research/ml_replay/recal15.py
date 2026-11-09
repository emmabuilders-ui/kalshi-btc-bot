import numpy as np, pandas as pd, time
from sklearn.linear_model import LogisticRegression
S = pd.read_parquet('work/models_scored.parquet').sort_values('snap_ts').reset_index(drop=True)
ts = lambda s: pd.Timestamp(s, tz='UTC').timestamp(); cut = ts('2026-09-01')
S = S[S.snap_ts >= ts('2026-07-25')].reset_index(drop=True)  # 30-day lookback before Sep 1 plus holdout
# live model output: monthly out-of-fold before Sep, weekly refit in Sep-Oct (both leak-free)
S['corr_live'] = np.where(S.snap_ts >= cut, S.p_corr_wk, S.p_corr); S['direct_live'] = np.where(S.snap_ts >= cut, S.p_direct_wk, S.p_direct)
eps = 1e-4; lg = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
BASE = {'kalshi': 'p_kalshi', 'corr': 'corr_live', 'direct': 'direct_live'}
X = {b: np.column_stack([lg(S[c]), lg(S[c]) * S.min_left, S.min_left]) for b, c in BASE.items()}
y = S.res.to_numpy(); close = S.close_ts.to_numpy()
WINDOWS = {'1d': 86400, '7d': 7 * 86400, '30d': 30 * 86400}
out = {f'p_{b}_cal{w}': np.full(len(S), np.nan) for b in BASE for w in WINDOWS}
hold = S[S.snap_ts >= cut]; t0 = time.time()
for n, (op, g) in enumerate(hold.groupby('open_ts')):  # one refit per contract, using only contracts closed at or before this one opened
    idx = g.index.to_numpy()
    for w, sec in WINDOWS.items():
        tr = np.where((close <= op) & (close > op - sec))[0]
        if len(tr) < 300: continue
        for b in BASE:
            lr = LogisticRegression(C=1.0, max_iter=200).fit(X[b][tr], y[tr])
            out[f'p_{b}_cal{w}'][idx] = lr.predict_proba(X[b][idx])[:, 1]
    if n % 500 == 0: print(n, round(time.time() - t0), flush=True)
for k, v in out.items(): S[k] = v
S[S.snap_ts >= cut].to_parquet('work/recal15.parquet'); print('done', n)
