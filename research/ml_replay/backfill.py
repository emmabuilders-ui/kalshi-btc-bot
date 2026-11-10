# Walk-forward backfill of every model through Mar-Aug 2026 (only data closed before each refit). Each part saves work/bf_<part>.parquet.
import numpy as np, pandas as pd, json, sys, time, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from joblib import Parallel, delayed
ts = lambda s: pd.Timestamp(s, tz='UTC').timestamp()
MAR, SEP = ts('2026-03-01'), ts('2026-09-01')
WEEKS = pd.date_range('2026-03-01', '2026-09-01', freq='7D', tz='UTC')
def weeks():
    for w in WEEKS:
        a = w.timestamp(); b = min(a + 7 * 86400, SEP)
        if a < SEP: yield a, b
DIRECT = ['dist', 'z', 'z60', 'secs_left', 'sig1', 'sig5', 'sig15', 'sig60', 'vol_ratio', 'vol_ratio_short', 'ret_10', 'ret_30', 'ret_60', 'ret_300', 'ret_900', 'dist_hi', 'dist_lo', 'hour_utc', 'dow']
CORR = DIRECT + ['x', 'spread', 'log_cumvol', 'ask_age_c', 'bid_age_c']
mk = lambda: HGB(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=0)
T0 = time.time(); log = lambda *a: print(round(time.time() - T0), *a, flush=True)
part = sys.argv[1]
S = pd.read_parquet('work/models_scored.parquet').sort_values('snap_ts').reset_index(drop=True)
if part == 'static_vol':   # static calibration and vol model refit monthly on everything closed before the month
    out = S[['ticker', 'k']].copy(); out['p_static'] = np.nan; out['p_vol'] = np.nan
    Teff = (S.secs_left - 40).clip(lower=1)
    for m in pd.date_range('2026-03-01', '2026-08-01', freq='MS', tz='UTC'):
        a, b = m.timestamp(), (m + pd.offsets.MonthBegin(1)).timestamp()
        tr = S.close_ts <= a; te = (S.snap_ts >= a) & (S.snap_ts < b)
        for ml, g in S[te].groupby('min_left'):
            f = S[tr & (S.min_left == ml)]
            out.loc[g.index, 'p_static'] = LogisticRegression(C=1e6).fit(f[['x']], f.res).predict_proba(g[['x']])[:, 1]
        pv = lambda c, idx: norm.cdf(S.dist[idx] / (c * S.sig15[idx].clip(lower=1e-7) * np.sqrt(Teff[idx])))
        c = minimize_scalar(lambda c: np.mean((pv(c, tr) - S.res[tr]) ** 2), bounds=(0.3, 3), method='bounded').x
        out.loc[te, 'p_vol'] = pv(c, te); log('static/vol', m.date(), round(c, 3))
    out[S.snap_ts >= MAR].dropna().to_parquet('work/bf_static_vol.parquet')
if part == 'weekly':       # Direct and Correction ML refit every week on everything closed before the week
    out = S[['ticker', 'k']].copy(); out['p_direct_wk'] = np.nan; out['p_corr_wk'] = np.nan
    for a, b in weeks():
        tr = S[S.close_ts <= a]; te = S[(S.snap_ts >= a) & (S.snap_ts < b)]
        out.loc[te.index, 'p_direct_wk'] = mk().fit(tr[DIRECT], tr.res).predict_proba(te[DIRECT])[:, 1]
        out.loc[te.index, 'p_corr_wk'] = mk().fit(tr[CORR], tr.res).predict_proba(te[CORR])[:, 1]; log('weekly', pd.Timestamp(a, unit='s').date())
    out.dropna().to_parquet('work/bf_weekly.parquet')
if part == 'recal':        # logistic recal on trailing 30 days, refit after every settled contract (same as Sep-Oct version)
    W = pd.read_parquet('work/bf_weekly.parquet')
    S = S[S.snap_ts >= ts('2026-01-29')].drop(columns=['p_corr_wk', 'p_direct_wk']).merge(W, on=['ticker', 'k'], how='left').reset_index(drop=True)
    S['corr_live'] = S.p_corr_wk.fillna(S.p_corr); S['direct_live'] = S.p_direct_wk.fillna(S.p_direct)
    eps = 1e-4; lg = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    BASE = {'kalshi': 'p_kalshi', 'corr': 'corr_live', 'direct': 'direct_live'}
    S = S.dropna(subset=list(BASE.values())).reset_index(drop=True)
    X = {b: np.column_stack([lg(S[c]), lg(S[c]) * S.min_left, S.min_left]) for b, c in BASE.items()}
    y = S.res.to_numpy(); close = S.close_ts.to_numpy()
    groups = [(op, g.index.to_numpy()) for op, g in S[(S.snap_ts >= MAR) & (S.snap_ts < SEP)].groupby('open_ts')]
    def one(op, idx):
        tr = np.where((close <= op) & (close > op - 30 * 86400))[0]; r = {}
        if len(tr) < 300: return idx, r
        for b in BASE: r[b] = LogisticRegression(C=1.0, max_iter=200).fit(X[b][tr], y[tr]).predict_proba(X[b][idx])[:, 1]
        return idx, r
    res = Parallel(n_jobs=4, batch_size=64)(delayed(one)(op, idx) for op, idx in groups)
    out = S[['ticker', 'k']].copy()
    for b in BASE:
        col = np.full(len(S), np.nan)
        for idx, r in res:
            if b in r: col[idx] = r[b]
        out[f'p_{b}_cal30d'] = col
    out.dropna().to_parquet('work/bf_recal.parquet'); log('recal contracts', len(groups))
if part == 'flow':         # order-flow models: weekly Kalshi+OF and Correction+OF where taker side exists (Mar-Apr, Aug); regime+OF gate monthly OOF
    O = pd.read_parquet('work/models_of.parquet').sort_values('snap_ts').reset_index(drop=True)
    OF = ['of_signed_open'] + [f'of_{a}_{w}' for w in (10, 30, 60, 180) for a in ('imb', 'vol', 'n')] + ['of_big_180', 'of_dpx_30', 'of_dpx_60', 'of_dpx_180', 'of_px_vs_mid'] + [f'of_vwap_vs_mid_{w}' for w in (10, 30, 60, 180)]
    KOF = ['x', 'secs_left', 'spread'] + OF
    O['p_kof_wk'] = np.nan
    for a, b in weeks():
        te = O[(O.snap_ts >= a) & (O.snap_ts < b)]
        if te.empty: continue
        tr = O[O.close_ts <= a]; O.loc[te.index, 'p_kof_wk'] = mk().fit(tr[KOF], tr.res).predict_proba(te[KOF])[:, 1]; log('kof wk', pd.Timestamp(a, unit='s').date())
    EX = ['p_kalshi', 'p_roll', 'p_corr_of', 'p_direct_of']; O['p_regime_of'] = np.nan
    oof = O[(O.snap_ts >= ts('2026-02-01')) & (O.snap_ts < SEP)].dropna(subset=EX)
    for m in ['2026-03-01', '2026-04-01', '2026-08-01']:
        a = ts(m); b = (pd.Timestamp(m, tz='UTC') + pd.offsets.MonthBegin(1)).timestamp()
        h0 = oof[oof.close_ts <= a]; q1, q2 = h0.sig15.quantile([1 / 3, 2 / 3])
        cell = lambda d: np.where(d.sig15 < q1, 0, np.where(d.sig15 < q2, 1, 2)) * 9 + np.where(d.min_left <= 3, 0, np.where(d.min_left <= 8, 1, 2)) * 3 + np.where(d.mid < .2, 0, np.where(d.mid < .8, 1, 2))
        gate = {c: (min(EX, key=lambda e: np.mean((h[e] - h.res) ** 2)) if len(h) >= 500 else 'p_kalshi') for c, h in h0.assign(c=cell(h0)).groupby('c')}
        te = O[(O.snap_ts >= a) & (O.snap_ts < b)].dropna(subset=EX); te = te.assign(c=cell(te))
        O.loc[te.index, 'p_regime_of'] = [te.at[i, gate.get(c, 'p_kalshi')] for i, c in zip(te.index, te.c)]; log('gate', m, pd.Series(gate).value_counts().to_dict())
    O[(O.snap_ts >= MAR) & (O.snap_ts < SEP)][['ticker', 'k', 'p_corr_of', 'p_kof_wk', 'p_regime_of']].to_parquet('work/bf_flow.parquet')
if part == 'auto':         # auto-feature models; feature ranking redone using only data before Mar (train to Jan, validate Feb)
    D = pd.read_parquet('work/autofeat.parquet').sort_values('snap_ts').reset_index(drop=True)
    META = ['ticker', 'k', 'snap_ts', 'open_ts', 'close_ts', 'min_left', 'res', 'mid', 'ask', 'bid', 'day', 'month', 'period']
    ALL = [c for c in D.columns if c not in META]
    KAL = ['kmid', 'klogit', 'kspread', 'k_vs_vol'] + [c for c in ALL if c.startswith('kmid_chg')]; DIR = [c for c in ALL if c not in KAL]
    gbm = lambda: HGB(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=0)
    lasso = lambda: make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(penalty='l1', C=0.05, solver='saga', max_iter=300))
    import os
    if os.path.exists('work/bf_auto_rank.json'): rank = json.load(open('work/bf_auto_rank.json'))
    else:
        tr = D[D.close_ts <= ts('2026-02-01')]; va = D[(D.snap_ts >= ts('2026-02-01')) & (D.snap_ts < MAR)]
        rank = {}
        for fam, cols in (('correction', ALL), ('direct', DIR)):
            m = gbm().fit(tr[cols], tr.res); base = np.mean((m.predict_proba(va[cols])[:, 1] - va.res) ** 2); rng = np.random.default_rng(0); imp = {}
            for c in cols:
                v = va[cols].copy(); v[c] = rng.permutation(v[c].to_numpy()); imp[c] = np.mean((m.predict_proba(v)[:, 1] - va.res) ** 2) - base
            rank[fam] = sorted(imp, key=imp.get, reverse=True); log(fam, 'rank', rank[fam][:10])
    json.dump(rank, open('work/bf_auto_rank.json', 'w'))
    CAND = {'p_lasso_corr': (lasso, ALL), 'p_lasso_corr_wk': (lasso, ALL), 'p_gbm_corr_top5_wk': (gbm, rank['correction'][:5]), 'p_gbm_dir_top10_wk': (gbm, rank['direct'][:10])}
    for n in CAND: D[n] = np.nan
    for m in pd.date_range('2026-03-01', '2026-08-01', freq='MS', tz='UTC'):   # "frozen" lasso = monthly refit
        a, b = m.timestamp(), (m + pd.offsets.MonthBegin(1)).timestamp(); te = D[(D.snap_ts >= a) & (D.snap_ts < b)]; trm = D[D.close_ts <= a]
        ck = f'work/bf_auto_ck/m{int(a)}.npy'
        if not os.path.exists(ck): np.save(ck, lasso().fit(trm[ALL], trm.res).predict_proba(te[ALL])[:, 1])
        D.loc[te.index, 'p_lasso_corr'] = np.load(ck); log('lasso month', m.date())
    for a, b in weeks():
        te = D[(D.snap_ts >= a) & (D.snap_ts < b)]; trw = D[D.close_ts <= a]
        for n in ['p_lasso_corr_wk', 'p_gbm_corr_top5_wk', 'p_gbm_dir_top10_wk']:
            f, cols = CAND[n]; ck = f'work/bf_auto_ck/{n}_{int(a)}.npy'
            if not os.path.exists(ck): np.save(ck, f().fit(trw[cols], trw.res).predict_proba(te[cols])[:, 1])
            D.loc[te.index, n] = np.load(ck)
        log('auto wk', pd.Timestamp(a, unit='s').date())
    D[(D.snap_ts >= MAR) & (D.snap_ts < SEP)][['ticker', 'k'] + list(CAND)].to_parquet('work/bf_auto.parquet')
log('done', part)
