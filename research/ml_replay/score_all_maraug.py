# Every model scored walk-forward on Mar-Aug (backfill.py) next to Sep-Oct: Brier vs Kalshi, and NO 70-99c gap rules.
import numpy as np, pandas as pd
src = open('monthly.py').read().split("months = sorted")[0].replace("'min_left','res','mid'", "'min_left','res','mid','ask','bid'")
exec(src)                                     # A: Mar-Oct with Sep-Oct values of every model; N names
pre = A.snap_ts < pd.Timestamp('2026-09-01', tz='UTC').timestamp()
BF = {'bf_static_vol': ['p_static', 'p_vol'], 'bf_weekly': ['p_direct_wk', 'p_corr_wk'], 'bf_recal': ['p_kalshi_cal30d', 'p_corr_cal30d', 'p_direct_cal30d'],
      'bf_flow': ['p_corr_of', 'p_kof_wk', 'p_regime_of'], 'bf_auto': ['p_lasso_corr', 'p_lasso_corr_wk', 'p_gbm_corr_top5_wk', 'p_gbm_dir_top10_wk']}
for f, cols in BF.items():
    B = pd.read_parquet(f'work/{f}.parquet')[['ticker', 'k'] + cols].rename(columns={c: c + '_bf' for c in cols})
    A = A.merge(B, on=['ticker', 'k'], how='left')
    for c in cols: A.loc[pre, c] = A.loc[pre, c + '_bf']
A['per'] = np.where(pre, 'Mar-Aug', 'Sep-Oct')
fee = lambda p: .07 * p * (1 - p)
rows, rule = [], []
for m in M:
    if m == 'p_kalshi': continue
    for per, h in A.dropna(subset=[m]).groupby('per'):
        d = (h[m] - h.res) ** 2 - (h.p_kalshi - h.res) ** 2
        g = d.groupby(h.ticker).sum(); se = g.std() * np.sqrt(len(g)) / len(d)
        mo = d.groupby(h.month).mean()
        rows.append(dict(model=N[m], per=per, snaps=len(h), months=h.month.nunique(), brier_vs_kalshi_x1000=round(1000 * d.mean(), 3),
                         ci95=f'{1000*(d.mean()-2*se):+.3f} to {1000*(d.mean()+2*se):+.3f}', months_better=f'{(mo<0).sum()}/{len(mo)}'))
        for G in (2, 3):
            px = 1 - h.bid
            T = h.assign(px=px, win=1 - h.res)[(100 * ((1 - h[m]) - px - fee(px)) >= G) & (px >= .70) & (px <= .99) & h.min_left.between(6, 14)]
            T = T.sort_values('min_left', ascending=False).drop_duplicates('ticker'); T['pnl'] = T.win - T.px - fee(T.px)
            if len(T) < 20: continue
            mo2 = T.groupby('month').pnl.mean()
            rule.append(dict(model=N[m], gap=G, per=per, n=len(T), win_vs_price=f'{100*T.win.mean():.1f} vs {100*T.px.mean():.1f}',
                             edge_c=round(100 * T.pnl.mean(), 2), t=round(T.pnl.mean() / (T.pnl.std() / np.sqrt(len(T))), 2), months=f'{(mo2>0).sum()}/{len(mo2)}'))
R = pd.DataFrame(rows); Q = pd.DataFrame(rule)
RW = R.pivot_table(index='model', columns='per', values=['brier_vs_kalshi_x1000', 'ci95', 'months_better', 'months'], aggfunc='first')
RW.columns = [f'{b} {a}' for a, b in RW.columns]; RW = RW.sort_values('Mar-Aug brier_vs_kalshi_x1000')
QW = Q.pivot_table(index=['model', 'gap'], columns='per', values=['n', 'win_vs_price', 'edge_c', 't', 'months'], aggfunc='first')
QW.columns = [f'{b} {a}' for a, b in QW.columns]
with pd.ExcelWriter('/mnt/project-files/strategy/ml_replay/all_models_mar_aug_walk_forward.xlsx') as w:
    RW.reset_index().to_excel(w, sheet_name='Brier vs Kalshi', index=False); QW.reset_index().to_excel(w, sheet_name='NO 70-99c 6-14 min', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print(RW.to_string()); print(QW.to_string())
