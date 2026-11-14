"""Score CME futures order flow vs Kalshi 15-min (KXBTC15M) and hourly above/below (KXBTCD).
Outputs work/*.parquet tables and the Excel workbook."""
import numpy as np, pandas as pd, warnings, sys
from scipy import stats
import statsmodels.api as sm
from flow import Flow, WINDOWS
warnings.filterwarnings('ignore')
rng = np.random.default_rng(7)
SIGS = [f'{nm}_{s}_{W}' for nm in ['nvz', 'vi', 'ti'] for s in ['comb', 'btc', 'mbt'] for W in WINDOWS]
PRIMARY = 'nvz_comb_3'
OUT = {}

def fee(p):  # Kalshi taker fee, cents per contract, p = price in dollars
    return np.ceil(7 * p * (1 - p)) / 100

def auc(x, y):
    """AUC of x for y==1 (Mann-Whitney), y binary."""
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = stats.rankdata(x); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def tau(x, y, nmax=60000):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) > nmax:
        i = rng.choice(len(x), nmax, replace=False); x, y = x[i], y[i]
    if len(x) < 100: return np.nan, np.nan
    t, p = stats.kendalltau(x, y); return t, p

def logit(p): p = np.clip(p, 0.005, 0.995); return np.log(p / (1 - p))

def load(product):
    if product == '15m':
        S = pd.read_parquet('work/replay.parquet')
        M = pd.read_parquet('work/markets_audited.parquet')[['ticker', 'brti_avg']]
        S = S.merge(M, on='ticker'); S = S.rename(columns={'brti_avg': 'settle'})
        S['secs_left'] = S.close_ts - S.snap_ts
    else:
        S = pd.read_parquet('work/hourly.parquet'); S['secs_left'] = S.close_ts - S.snap_ts
    S = S.reset_index(drop=True)
    F = Flow(); G = F.at(S.snap_ts.to_numpy())
    S = pd.concat([S, G], axis=1)
    S['move'] = S.settle - S.brti
    S['up'] = (S.move > 0).astype(float); S.loc[S.move == 0, 'up'] = np.nan
    S['pk'] = S.mid.clip(0.01, 0.99)
    S['month'] = pd.to_datetime(S.snap_ts, unit='s').dt.strftime('%Y-%m')
    return S

def coverage(S, product):
    rows = []
    for W in WINDOWS:
        v = S[f'valid_{W}']
        rows.append(dict(product=product, window=W, snapshots=len(S), valid=int(v.sum()), valid_pct=round(100 * v.mean(), 1),
                         contracts_valid=S.loc[v, 'ticker'].nunique(), contracts=S.ticker.nunique()))
    return pd.DataFrame(rows)

def screen(S, product):
    """tau vs move, AUC vs up, and residual logistic coefficient, per signal and period."""
    rows = []
    for sig in SIGS:
        r = dict(product=product, signal=sig)
        for per in ['Dec-Aug', 'Sep-Oct']:
            d = S[(S.period == per) & S[sig].notna()]
            t, tp = tau(d[sig].to_numpy(), d.move.to_numpy())
            r[f'n_{per}'] = len(d); r[f'tau_{per}'] = t; r[f'auc_{per}'] = auc(d[sig].to_numpy(), d.up.to_numpy())
            # residual: logit(P yes) = a + b1*logit(kalshi) + b2*sig  (same-sign, p<0.01 both periods is the bar)
            x = sm.add_constant(np.column_stack([logit(d.pk.to_numpy()), np.clip(d[sig].to_numpy(), -6, 6)]))
            try:
                fit = sm.Logit(d.res.to_numpy().astype(float), x).fit(disp=0, maxiter=100)
                r[f'coef_{per}'] = fit.params[2]; r[f'p_{per}'] = fit.pvalues[2]
            except Exception:
                r[f'coef_{per}'] = np.nan; r[f'p_{per}'] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)

def brier_test(S, product, sigs, nboot=300):
    """Fit logit(P) = a + b*logit(K) + c*sig (+ d*sig*minutes-left scaled) on Dec-Aug, score Sep-Oct Brier vs Kalshi mid.
    Contract-level bootstrap CI of the difference (model - Kalshi; negative = model better)."""
    rows = []
    tr = S[S.period == 'Dec-Aug']; te = S[S.period == 'Sep-Oct']
    for sig in sigs:
        a = tr[tr[sig].notna()]; b = te[te[sig].notna()]
        if len(a) < 1000 or len(b) < 500: continue
        def X(d): 
            s = np.clip(d[sig].to_numpy(), -6, 6); ml = d.secs_left.to_numpy() / 60.0
            return sm.add_constant(np.column_stack([logit(d.pk.to_numpy()), s, s * ml / 15.0]))
        fit = sm.Logit(a.res.to_numpy().astype(float), X(a)).fit(disp=0, maxiter=100)
        p = fit.predict(X(b)); y = b.res.to_numpy().astype(float); k = b.pk.to_numpy()
        # Kalshi-alone recalibrated baseline (so improvement is not just calibration)
        fit0 = sm.Logit(a.res.to_numpy().astype(float), sm.add_constant(logit(a.pk.to_numpy()))).fit(disp=0)
        p0 = fit0.predict(sm.add_constant(logit(k)))
        d_raw = (p - y) ** 2 - (k - y) ** 2; d_cal = (p - y) ** 2 - (p0 - y) ** 2
        tick = b.ticker.to_numpy(); u, inv = np.unique(tick, return_inverse=True)
        sr = np.bincount(inv, d_raw); sc = np.bincount(inv, d_cal); cnt = np.bincount(inv)
        bs_r = []; bs_c = []
        for _ in range(nboot):
            i = rng.integers(0, len(u), len(u)); bs_r.append(sr[i].sum() / cnt[i].sum()); bs_c.append(sc[i].sum() / cnt[i].sum())
        rows.append(dict(product=product, signal=sig, n_test=len(b), brier_kalshi=np.mean((k - y) ** 2), brier_model=np.mean((p - y) ** 2),
                         diff_vs_kalshi=d_raw.mean(), ci_lo=np.percentile(bs_r, 2.5), ci_hi=np.percentile(bs_r, 97.5),
                         diff_vs_kalshi_recal=d_cal.mean(), ci_lo_recal=np.percentile(bs_c, 2.5), ci_hi_recal=np.percentile(bs_c, 97.5),
                         coef_sig=fit.params[2], coef_sig_x_time=fit.params[3]))
    return pd.DataFrame(rows)

def deciles(S, product, sig):
    rows = []
    for per in ['Dec-Aug', 'Sep-Oct']:
        d = S[(S.period == per) & S[sig].notna()].copy()
        cuts = np.quantile(S.loc[(S.period == 'Dec-Aug') & S[sig].notna(), sig], np.linspace(0, 1, 11)); cuts[0] = -np.inf; cuts[-1] = np.inf
        d['dec'] = pd.cut(d[sig], cuts, labels=False) + 1
        g = d.groupby('dec').agg(n=('up', 'size'), sig_mean=(sig, 'mean'), up_pct=('up', lambda x: 100 * x.mean()), avg_move=('move', 'mean'),
                                yes_win_pct=('res', lambda x: 100 * x.mean()), kalshi_mid=('mid', lambda x: 100 * x.mean()))
        g['yes_win_minus_mid'] = g.yes_win_pct - g.kalshi_mid; g['period'] = per; g['product'] = product; g['signal'] = sig
        rows.append(g.reset_index())
    return pd.concat(rows)

def trades(S, product, sig, ks=(1.0, 1.5, 2.0), with_flow=True, by=('band10', 'tgroup')):
    """One trade per contract per rule: first valid snapshot where |sig| > k (sig in SD units of its Dec-Aug distribution).
    with_flow: sig>+k buys YES at ask, sig<-k buys NO at 1-bid. Fade = opposite. Edge = win% - price - fee."""
    sd = S.loc[(S.period == 'Dec-Aug'), sig].std() if not sig.startswith('nvz') else 1.0
    out = []
    for k in ks:
        z = S[sig] / sd
        d = S[z.abs() > k].sort_values(['ticker', 'snap_ts']).drop_duplicates('ticker').copy()
        side_yes = (z.loc[d.index] > 0) if with_flow else (z.loc[d.index] < 0)
        d['side'] = np.where(side_yes, 'YES', 'NO')
        d['price'] = np.where(side_yes, d.ask, 1 - d.bid)
        d['win'] = np.where(side_yes, d.res == 1, d.res == 0).astype(float)
        d['fee'] = fee(d.price); d['pnl'] = d.win - d.price - d.fee  # dollars per contract
        d['band10'] = (np.floor(d.price * 10).clip(0, 9) * 10).astype(int).astype(str) + 'c'
        d['tgroup'] = pd.cut(d.secs_left / 60, [0, 5, 9, 14, 30, 61], labels=['1-5', '6-9', '10-14', '15-30', '31-60'], right=True).astype(str)
        d['k'] = k
        out.append(d)
    D = pd.concat(out)
    def agg(g):
        r = dict(trades=len(g), avg_price=100 * g.price.mean(), win_pct=100 * g.win.mean(), fee_c=100 * g.fee.mean())
        r['win_minus_price'] = r['win_pct'] - r['avg_price']; r['edge_after_fee_c'] = 100 * g.pnl.mean()
        r['t'] = g.pnl.mean() / (g.pnl.std() / np.sqrt(len(g))) if len(g) > 1 and g.pnl.std() > 0 else np.nan
        m = g.groupby('month').pnl.mean(); r['months_pos'] = f'{(m > 0).sum()}/{len(m)}'
        return pd.Series(r)
    res = []
    for keys, g in D.groupby(['k', 'period']): res.append(pd.concat([pd.Series(dict(zip(['k', 'period'], keys), side='all', band10='all', tgroup='all')), agg(g)]))
    for keys, g in D.groupby(['k', 'period', 'side']): res.append(pd.concat([pd.Series(dict(zip(['k', 'period', 'side'], keys), band10='all', tgroup='all')), agg(g)]))
    for keys, g in D.groupby(['k', 'period', 'side', 'tgroup']): res.append(pd.concat([pd.Series(dict(zip(['k', 'period', 'side', 'tgroup'], keys), band10='all')), agg(g)]))
    for keys, g in D.groupby(['k', 'period', 'side', 'band10', 'tgroup']): res.append(pd.concat([pd.Series(dict(zip(['k', 'period', 'side', 'band10', 'tgroup'], keys))), agg(g)]))
    R = pd.DataFrame(res); R.insert(0, 'rule', 'with flow' if with_flow else 'fade flow'); R.insert(0, 'signal', sig); R.insert(0, 'product', product)
    return R, D

if __name__ == '__main__':
    products = sys.argv[1:] or ['15m', 'hourly']
    cov, scr, bri, dec, trd = [], [], [], [], []
    for prod in products:
        S = load(prod); print(prod, len(S), 'snapshots', flush=True)
        S.to_parquet(f'work/merged_{prod}.parquet')
        cov.append(coverage(S, prod))
        sc = screen(S, prod); scr.append(sc); print(sc[sc.signal == PRIMARY].T, flush=True)
        bri.append(brier_test(S, prod, SIGS)); print('brier done', flush=True)
        dec.append(deciles(S, prod, PRIMARY))
        for sig in [PRIMARY, 'vi_comb_3', 'ti_comb_3', 'nvz_comb_1', 'nvz_comb_5', 'nvz_comb_10', 'nvz_btc_3', 'nvz_mbt_3']:
            for wf in [True, False]:
                R, D = trades(S, prod, sig, with_flow=wf); trd.append(R)
                if sig == PRIMARY and wf: D.to_parquet(f'work/trades_{prod}_primary.parquet')
        print('trades done', flush=True)
    pd.concat(cov).to_parquet('work/coverage.parquet'); pd.concat(scr).to_parquet('work/screen.parquet'); pd.concat(bri).to_parquet('work/brier.parquet')
    pd.concat(dec).to_parquet('work/deciles.parquet'); pd.concat(trd).to_parquet('work/trades.parquet')
    print('saved')
