"""Test 2: futures flow as an input to our models. (A) Direct / Correction ML on the 15-min gap-rule snapshots: full re-run of the
rule with p_new; (B) lock-in: existing looks only (model_prob in the replay file; trades can be dropped, not added); (C) every bot:
logistic won ~ logit(price) + flow + flow x minutes left, Dec-Aug fit, Sep-Oct check. Writes work/wm_t2_*.parquet."""
import numpy as np, pandas as pd, warnings, statsmodels.api as sm
warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, 'with_models'); sys.path.insert(0, '/home/user/kalshi-btc-bot')
from kalshi_btc_bot.model_gap_paper import CUSHION_MIN, PX_MIN, PX_MAX, MIN_GAP
rng = np.random.default_rng(11)
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
expit = lambda x: 1 / (1 + np.exp(-x))
FEE = lambda p: np.ceil(7 * p * (1 - p))  # cents, p in dollars
PER = ['Dec-Aug', 'Sep-Oct']
out = {}

def fit_shift(y, p0, z, ml):
    """logit(p_new) = logit(p0) + b1 z + b2 z*ml (z = 0 when flow missing)."""
    X = np.column_stack([z, z * ml]); X = np.nan_to_num(X)
    try:
        r = sm.GLM(y, X, family=sm.families.Binomial(), offset=logit(p0)).fit()
        return np.asarray(r.params), np.asarray(r.pvalues)
    except Exception:
        return np.zeros(2), np.ones(2)

def apply_shift(p0, z, ml, b):
    X = np.nan_to_num(np.column_stack([z, z * ml])); return expit(logit(p0) + X @ b)

def trade_stats(d):
    if len(d) == 0: return {'n': 0}
    bym = d.groupby('month').edge.mean()
    return {'n': len(d), 'avg_price': d.price.mean(), 'win_pct': 100 * d.won.mean(), 'edge': d.edge.mean(),
            'edge_day_median': d.groupby('day').edge.mean().median(), 'months_pos': f'{int((bym > 0).sum())}/{len(bym)}'}

# ---------------- (A) Direct / Correction ----------------
S = pd.read_parquet('work/gap_snaps.parquet')
fl = pd.read_parquet('work/merged_15m.parquet', columns=['ticker', 'k', 'nvz_comb_3', 'nvz_comb_1', 'vi_comb_3'])
S = S.merge(fl, on=['ticker', 'k'], how='left')
S['day'] = pd.to_datetime(S.snap_ts, unit='s').dt.floor('D'); S['ml'] = S.min_left.astype(float)
S['y'] = (S.res == 1).astype(int)

def run_rule(S, pcol, cond, min_gap=MIN_GAP):
    """One look per contract: first k with a side ask 70-99c and gap >= min_gap; NO only; Direct needs the condition."""
    d = S.sort_values(['ticker', 'k']).copy()
    p = d[pcol].to_numpy()
    gy = p - d.yes_ask - 0.07 * d.yes_ask * (1 - d.yes_ask); gn = (1 - p) - d.no_ask - 0.07 * d.no_ask * (1 - d.no_ask)
    ly = (d.yes_ask.between(PX_MIN, PX_MAX)) & (gy >= min_gap); ln = (d.no_ask.between(PX_MIN, PX_MAX)) & (gn >= min_gap)
    look = np.where(ln, 'no', np.where(ly, 'yes', None))  # evaluate() loops yes then no, so NO wins when both
    d['look'] = look
    d = d[d.look.notna()].groupby('ticker', sort=False).head(1)
    d = d[d.look == 'no']
    if cond:
        c = (-d.f_z > CUSHION_MIN) | (d.f_brti < d.f_pre_lo); d = d[c]
    d = d.assign(price=100 * d.no_ask, won=(d.res == 0).astype(int)); d['fee'] = FEE(d.no_ask); d['edge'] = 100 * d.won - d.price - d.fee
    return d

def match_count(S, pcol, cond, target):
    lo, hi = 0.0, 0.3
    for _ in range(30):
        mid = (lo + hi) / 2; n = len(run_rule(S, pcol, cond, mid))
        if n > target: lo = mid
        else: hi = mid
    return run_rule(S, pcol, cond, (lo + hi) / 2), (lo + hi) / 2

def brier_ci(S, pa, pb, nboot=300):
    """Brier(pa) - Brier(pb), contract bootstrap (negative = pa better)."""
    y = S.y.to_numpy(); d = (pa - y) ** 2 - (pb - y) ** 2
    tk = S.ticker.to_numpy(); u, inv = np.unique(tk, return_inverse=True)
    per = np.bincount(inv, weights=d); cnt = np.bincount(inv)
    bs = []
    for _ in range(nboot):
        idx = rng.integers(0, len(u), len(u)); bs.append(per[idx].sum() / cnt[idx].sum())
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

rowsA, trA, brA = [], [], []
for name, pcol, cond in [('Direct NO gap', 'p_direct', True), ('Correction NO gap', 'p_corr', False)]:
    for zc in ['nvz_comb_3', 'nvz_comb_1', 'vi_comb_3']:
        S['z'] = S[zc]
        # out-of-fold by month on Dec-Aug, frozen Dec-Aug fit for Sep-Oct
        pn = np.full(len(S), np.nan); coefs = []
        da = S.period == 'Dec-Aug'
        for m in sorted(S[da].month.unique()):
            tr = da & (S.month != m); te = S.month == m
            b, pv = fit_shift(S.y[tr], S[pcol][tr], S.z[tr], S.ml[tr]); pn[te] = apply_shift(S[pcol][te], S.z[te], S.ml[te], b)
            coefs.append({'model': name, 'flow': zc, 'fold': m, 'b_flow': b[0], 'b_flow_x_min': b[1], 'p_flow': pv[0], 'p_flow_x_min': pv[1]})
        b, pv = fit_shift(S.y[da], S[pcol][da], S.z[da], S.ml[da]); so = S.period == 'Sep-Oct'; pn[so] = apply_shift(S[pcol][so], S.z[so], S.ml[so], b)
        coefs.append({'model': name, 'flow': zc, 'fold': 'Dec-Aug all (used for Sep-Oct)', 'b_flow': b[0], 'b_flow_x_min': b[1], 'p_flow': pv[0], 'p_flow_x_min': pv[1]})
        S['p_new'] = pn
        rowsA += coefs
        variants = {'model (as is)': pcol, 'model + flow (fit)': 'p_new'}
        for kk in (1, 2, 3):
            sgn = np.sign(np.nan_to_num(S.z)) * (np.abs(np.nan_to_num(S.z)) > 0.5); S[f'p_shift{kk}'] = np.clip(S[pcol] + kk / 100 * sgn, 0.001, 0.999)
            variants[f'model + {kk} pt shift'] = f'p_shift{kk}'
        base = {per: run_rule(S[S.period == per], pcol, cond) for per in PER}
        for vn, vc in variants.items():
            for per in PER:
                Sp = S[S.period == per]
                d = run_rule(Sp, vc, cond); st = trade_stats(d); st.update(model=name, flow=zc, variant=vn, period=per, matched=False, min_gap=MIN_GAP); trA.append(st)
                d.assign(bot=name, variant=vn, flow=zc).to_parquet(f'work/wm_t2_trades_{name.split()[0]}_{zc}_{vn.replace(" ", "_").replace("+", "plus")}_{per}.parquet')
                if vc != pcol:
                    dm, mg = match_count(Sp, vc, cond, len(base[per])); st = trade_stats(dm); st.update(model=name, flow=zc, variant=vn, period=per, matched=True, min_gap=mg); trA.append(st)
            if vc != pcol:
                for per in PER:
                    Sp = S[S.period == per]; m, lo, hi = brier_ci(Sp, Sp[vc].to_numpy(), Sp[pcol].to_numpy())
                    brA.append({'model': name, 'flow': zc, 'variant': vn, 'period': per, 'brier_diff_new_minus_model': m, 'ci_lo': lo, 'ci_hi': hi, 'n_snaps': len(Sp)})
        print(name, zc, 'done', flush=True)
pd.DataFrame(rowsA).to_parquet('work/wm_t2_coefs.parquet'); TA = pd.DataFrame(trA); TA.to_parquet('work/wm_t2_rules.parquet'); pd.DataFrame(brA).to_parquet('work/wm_t2_brier.parquet')
print(TA[TA.flow == 'nvz_comb_3'].pivot_table(index=['model', 'variant', 'matched'], columns='period', values=['n', 'avg_price', 'win_pct', 'edge'], aggfunc='first').round(2).to_string())
print(pd.DataFrame(brA)[lambda d: d.flow == 'nvz_comb_3'].round(5).to_string())

# ---------------- (B) lock-in: existing looks ----------------
T = pd.read_parquet('work/wm_trades.parquet'); L = T[T.bot == 'Lock-in'].copy()
L['y'] = L.won; L['p0'] = L.p_model; L['ml'] = L.secs_left / 60
rowsB = []
for zc in ['svi_60', 'svi_30', 'nvz_comb_3']:
    aug = L.period == 'Dec-Aug'; b, pv = fit_shift(L.y[aug], L.p0[aug], L[zc][aug], L.ml[aug])
    L['p_new'] = apply_shift(L.p0, L[zc], L.ml, b)
    for per in PER:
        P = L[L.period == per]
        for vn, keep in [('all looks (bot as is)', np.ones(len(P), bool)), ('keep if p_new - price - fee >= bot threshold (5 pts)', (100 * P.p_new - P.price - P.fee >= 5).to_numpy()),
                         ('keep if p_new >= p_model', (P.p_new >= P.p0).to_numpy())]:
            st = trade_stats(P[keep]); st.update(flow=zc, variant=vn, period=per, b_flow=b[0], p_flow=pv[0]); rowsB.append(st)
RB = pd.DataFrame(rowsB); RB.to_parquet('work/wm_t2_lockin.parquet'); print(RB.round(2).to_string())

# ---------------- (C) every bot: logistic on price + flow ----------------
rowsC = []
for bot, B in T.groupby('bot'):
    B = B.copy(); B['ml'] = B.min_left.astype(float); B['lp'] = logit(B.price / 100)
    for zc in ['nvz_comb_3', 'nvz_comb_1', 'vi_comb_3']:
        z = np.nan_to_num(B[zc].to_numpy()) * B.dir.to_numpy() if B.kind.iloc[0] == 'directional' else np.abs(np.nan_to_num(B[zc].to_numpy()))
        B['zs'] = z
        for per in PER:
            P = B[B.period == per]
            if len(P) < 60 or P.won.nunique() < 2: continue
            X = sm.add_constant(np.column_stack([P.lp, P.zs, P.zs * P.ml]))
            try:
                r = sm.Logit(P.won, X).fit(disp=0, maxiter=200)
                rowsC.append({'bot': bot, 'flow': zc, 'period': per, 'n': len(P), 'coef_flow': np.asarray(r.params)[2], 'p_flow': np.asarray(r.pvalues)[2], 'coef_flow_x_min': np.asarray(r.params)[3], 'p_flow_x_min': np.asarray(r.pvalues)[3],
                              'note': 'flow signed toward the bought side' if B.kind.iloc[0] == 'directional' else '|flow| (band bots: strong flow should hurt)'})
            except Exception as e:
                rowsC.append({'bot': bot, 'flow': zc, 'period': per, 'n': len(P), 'note': f'fit failed: {e}'})
RC = pd.DataFrame(rowsC); RC.to_parquet('work/wm_t2_logit_bots.parquet'); print(RC[RC.flow == 'nvz_comb_3'].round(3).to_string())
