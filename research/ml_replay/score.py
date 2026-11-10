import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
S = pd.read_parquet('work/replay_scored.parquet')
MODELS = ['p_kalshi', 'p_static', 'p_roll', 'p_vol']
NAMES = {'p_kalshi': 'Kalshi mid as is', 'p_static': 'Static calibration (fit Dec-Aug)', 'p_roll': 'Rolling 30-day recalibration (daily refit)', 'p_vol': 'Simple vol model (fit Dec-Aug)'}
C = S[S.p_roll.notna()].copy()
rng = np.random.default_rng(0)
def boot_diff(g, m, n=500):  # day-block bootstrap of Brier(model) - Brier(Kalshi)
    d = ((g[m] - g.res) ** 2 - (g.p_kalshi - g.res) ** 2).groupby(g.day).agg(['sum', 'size'])
    s, k = d['sum'].to_numpy(), d['size'].to_numpy(); out = []
    for _ in range(n):
        i = rng.integers(0, len(s), len(s)); out.append(s[i].sum() / k[i].sum())
    return np.percentile(out, [2.5, 97.5])
rows = []
for per, g in C.groupby('period'):
    for m in MODELS:
        b = np.mean((g[m] - g.res) ** 2); lo, hi = boot_diff(g, m) if m != 'p_kalshi' else (0, 0)
        rows.append(dict(period=per, model=NAMES[m], snapshots=len(g), contracts=g.ticker.nunique(), brier=round(b, 5),
                         vs_kalshi=round(b - np.mean((g.p_kalshi - g.res) ** 2), 5), ci95_lo=round(lo, 5), ci95_hi=round(hi, 5),
                         auc=round(roc_auc_score(g.res, g[m]), 4), in_sample=(per == 'Dec-Aug' and m in ('p_static', 'p_vol'))))
brier = pd.DataFrame(rows); print(brier.to_string())
# Brier by minutes left, Sep-Oct
bm = C[C.period == 'Sep-Oct'].groupby('min_left').apply(lambda g: pd.Series({NAMES[m]: np.mean((g[m] - g.res) ** 2) for m in MODELS} | {'snapshots': len(g)})).round(5)
# calibration: Kalshi mid bucket vs win rate, by period
C['bucket'] = (np.floor(C.mid * 20) / 20).clip(0, 0.95)
cal = C.groupby(['period', 'bucket']).agg(snapshots=('res', 'size'), contracts=('ticker', 'nunique'), kalshi_mid=('mid', 'mean'), win_rate=('res', 'mean')).reset_index()
cal['win_minus_price_pts'] = (cal.win_rate - cal.kalshi_mid) * 100
# trades: one per contract, first qualifying minute
fee = lambda p: 0.07 * p * (1 - p)
T = []
for m in MODELS[1:]:
    for thr in (0.05, 0.07, 0.10):
        y = C[C[m] - C.ask >= thr].assign(side='YES', price=lambda d: d.ask, win=lambda d: d.res)
        n = C[C.bid - C[m] >= thr].assign(side='NO', price=lambda d: 1 - d.bid, win=lambda d: 1 - d.res)
        t = pd.concat([y, n]).sort_values('snap_ts').drop_duplicates('ticker')
        t['model'] = NAMES[m]; t['thr_pts'] = int(thr * 100); t['cost'] = t.price + fee(t.price); t['pnl'] = t.win - t.cost
        T.append(t[['model', 'thr_pts', 'side', 'ticker', 'period', 'month', 'min_left', 'price', 'cost', 'win', 'pnl', m]].rename(columns={m: 'model_p'}))
T = pd.concat(T)
def summ(g):
    mo = g.groupby('month').pnl.mean()
    days = (g.month.nunique() and None)
    return pd.Series(dict(trades=len(g), win_pct=100 * g.win.mean(), avg_price_c=100 * g.price.mean(), breakeven_pct=100 * g.cost.mean(),
                          win_minus_price_pts=100 * (g.win.mean() - g.cost.mean()), ev_per_contract_c=100 * g.pnl.mean(),
                          se_c=100 * g.pnl.std() / np.sqrt(len(g)), months_positive=f"{(mo > 0).sum()}/{len(mo)}"))
tr = T.groupby(['model', 'thr_pts', 'period']).apply(summ).reset_index().round(2)
trs = T.groupby(['model', 'thr_pts', 'period', 'side']).apply(summ).reset_index().round(2)
print(tr.to_string())
# frequency and sample size: trades per day observed, n needed for target edge at observed price, 95% two-sided, 80% power
ndays = C.groupby('period').day.nunique()
ss = []
for (mname, thr, per), g in T.groupby(['model', 'thr_pts', 'period']):
    p0 = g.cost.mean(); perday = len(g) / ndays[per]
    for d in (0.03, 0.05, 0.07, 0.10):
        p1 = min(p0 + d, 0.999)
        n = ((1.96 * np.sqrt(p0 * (1 - p0)) + 0.8416 * np.sqrt(p1 * (1 - p1))) / d) ** 2
        nb = ((2.81 * np.sqrt(p0 * (1 - p0)) + 0.8416 * np.sqrt(p1 * (1 - p1))) / d) ** 2  # 10 tests, Bonferroni (alpha .005)
        ss.append(dict(model=mname, thr_pts=thr, period=per, trades_per_day=round(perday, 2), breakeven_pct=round(100 * p0, 1), true_edge_pts=int(d * 100),
                       n_needed=int(np.ceil(n)), days_needed=round(n / perday, 0) if perday else np.nan, n_needed_10_tests=int(np.ceil(nb)), days_needed_10_tests=round(nb / perday, 0) if perday else np.nan))
ss = pd.DataFrame(ss)
with pd.ExcelWriter('work/ml_replay_baselines.xlsx') as w:
    brier.to_excel(w, sheet_name='Brier_AUC', index=False); bm.to_excel(w, sheet_name='Brier_by_min_left_SepOct')
    cal.round(4).to_excel(w, sheet_name='Calibration_by_5c', index=False); tr.to_excel(w, sheet_name='Trades_summary', index=False)
    trs.to_excel(w, sheet_name='Trades_by_side', index=False); ss.to_excel(w, sheet_name='Sample_size', index=False)
T.to_parquet('work/trades.parquet'); print(ss[ss.period == 'Sep-Oct'].to_string())
