# Untouched Oct 26 - Nov 9: every model vs Kalshi (Brier), NO 70-99c 6-14 min gap rules, and the two paper bots' exact rules.
import numpy as np, pandas as pd, sys
S = pd.read_parquet('work/features.parquet'); NOV = pd.Timestamp('2026-10-26', tz='UTC').timestamp()
S = S[S.snap_ts >= NOV].merge(pd.read_parquet('work/nov_models.parquet'), on=['ticker', 'k'])
S['p_static'] = S.p_static; fee = lambda p: .07 * p * (1 - p)
NAMES = {'p_static': 'Static calibration', 'p_roll': 'Rolling 30-day recal (daily)', 'p_kalshi_cal30d': 'Kalshi recal every 15 min (30d)', 'p_vol': 'Simple volatility model',
         'p_direct': 'Direct ML', 'p_direct_wk': 'Direct ML weekly', 'p_corr': 'Correction ML', 'p_cl': 'Correction ML live (bot)', 'p_corr_wk': 'Correction ML weekly',
         'p_corr_cal30d': 'Correction ML + 15-min recal', 'p_direct_cal30d': 'Direct ML + 15-min recal', 'p_regime': 'Regime-switching', 'p_corr_of': 'Correction ML + order flow',
         'p_kof_wk': 'Kalshi + order flow weekly', 'p_regime_of': 'Regime + order flow', 'p_lasso_corr': 'Auto features, sparse logistic',
         'p_lasso_corr_wk': 'Auto features, sparse logistic weekly', 'p_gbm_corr_top5_wk': 'Auto features, trees top 5 weekly', 'p_gbm_dir_top10_wk': 'Auto features, trees BRTI only weekly'}
S = S.dropna(subset=['ask', 'bid'])
print('contracts', S.ticker.nunique(), 'snaps', len(S), 'days', S.day.nunique())
rows, rules = [], []
for m, nm in NAMES.items():
    h = S.dropna(subset=[m]); d = (h[m] - h.res) ** 2 - (h.p_kalshi - h.res) ** 2; g = d.groupby(h.ticker).sum(); se = g.std() * np.sqrt(len(g)) / len(d)
    rows.append(dict(model=nm, snaps=len(h), brier_vs_kalshi_x1000=round(1000 * d.mean(), 3), ci95=f'{1000*(d.mean()-2*se):+.3f} to {1000*(d.mean()+2*se):+.3f}'))
    for G in (2, 3):
        px = 1 - h.bid
        T = h.assign(px=px, win=1 - h.res)[(100 * ((1 - h[m]) - px - fee(px)) >= G) & (px >= .70) & (px <= .99) & h.min_left.between(6, 14)]
        T = T.sort_values('min_left', ascending=False).drop_duplicates('ticker'); T['pnl'] = T.win - T.px - fee(T.px)
        if len(T) >= 10:
            rules.append(dict(model=nm, gap=G, n=len(T), win_vs_price=f'{100*T.win.mean():.1f} vs {100*T.px.mean():.1f}', edge_c=round(100 * T.pnl.mean(), 2),
                              t=round(T.pnl.mean() / (T.pnl.std() / np.sqrt(len(T))), 2)))
R = pd.DataFrame(rows).sort_values('brier_vs_kalshi_x1000'); Q = pd.DataFrame(rules).sort_values('edge_c', ascending=False)
# the Direct bot's exact rule: first minute 1-9 after open where a side is 70-99c with gap>=3; NO only, cushion z>1.043 or below the 60-min pre-open low
B = pd.Series(np.load('work/brti.npy')).ffill().to_numpy(); T0 = 1764460800
op = (S.open_ts.to_numpy() - T0).astype(int); S['pre_lo'] = [B[a - 3600:a].min() for a in op]; S['pre_hi'] = [B[a - 3600:a].max() for a in op]
bots = []
for nm, m, cond in (('Direct bot rule', 'p_direct', True), ('Correction bot rule', 'p_cl', False)):
    tr = []
    for t, g in S[S.k.between(1, 9)].sort_values(['ticker', 'k']).groupby('ticker'):
        for r in g.itertuples():
            ya, na = r.ask, 1 - r.bid; py = getattr(r, m)
            gy, gn = py - ya - fee(ya), (1 - py) - na - fee(na)
            look = 'no' if (.70 <= na <= .99 and gn >= .03) else 'yes' if (.70 <= ya <= .99 and gy >= .03) else None
            if look is None: continue
            if look == 'no' and (not cond or (-r.z > 1.0427010403045358) or (r.brti < r.pre_lo)): tr.append((1 - r.res, na, r.month))
            break
    a = pd.DataFrame(tr, columns=['win', 'px', 'month']); a['pnl'] = a.win - a.px - fee(a.px)
    bots.append(dict(rule=nm, n=len(a), win_vs_price=f'{100*a.win.mean():.1f} vs {100*a.px.mean():.1f}', edge_c=round(100 * a.pnl.mean(), 2),
                     t=round(a.pnl.mean() / (a.pnl.std() / np.sqrt(len(a))), 2) if len(a) > 2 else None, pnl_20ct=round(20 * a.pnl.sum(), 2)))
Bt = pd.DataFrame(bots)
with pd.ExcelWriter('/mnt/project-files/strategy/ml_replay/untouched_oct26_nov9.xlsx') as w:
    Bt.to_excel(w, sheet_name='Paper bot rules', index=False); Q.to_excel(w, sheet_name='NO 70-99c 6-14 min', index=False); R.to_excel(w, sheet_name='Brier vs Kalshi', index=False)
pd.set_option('display.width', 220); print(Bt.to_string(index=False)); print(Q.to_string(index=False)); print(R.to_string(index=False))
