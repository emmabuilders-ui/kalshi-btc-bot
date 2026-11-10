import numpy as np, pandas as pd, ruptures as rpt, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
S = pd.read_parquet('work/replay_scored.parquet')
S['bucket'] = (np.floor(S.mid * 10) / 10).clip(0, 0.9)
S['week'] = pd.to_datetime(S.snap_ts, unit='s').dt.to_period('W-SUN').dt.start_time
S['err'] = S.res - S.mid
W = S.groupby(['bucket', 'week']).agg(n=('err', 'size'), contracts=('ticker', 'nunique'), err=('err', 'mean')).reset_index()
W = W[W.n >= 50]
fig, ax = plt.subplots(5, 2, figsize=(13, 15), sharex=True)
rows, wk = [], []
for i, (b, g) in enumerate(W.groupby('bucket')):
    g = g.sort_values('week'); y = g.err.to_numpy() * 100
    a = ax.flat[i]; a.axhline(0, color='#888', lw=0.8)
    a.plot(g.week, y, color='#2a6fdb', lw=1.4, marker='o', ms=3)
    sig = np.std(y, ddof=1); pen = 2 * np.log(len(y)) * sig ** 2
    bk = rpt.Pelt(model='l2', min_size=4).fit(y.reshape(-1, 1)).predict(pen=pen)[:-1]
    segs = np.split(np.arange(len(y)), bk)
    for s in segs:
        a.hlines(y[s].mean(), g.week.iloc[s[0]], g.week.iloc[s[-1]], color='#d9480f', lw=2)
    for j, k in enumerate(bk):
        before, after = y[segs[j]], y[segs[j + 1]]
        p = ttest_ind(before, after, equal_var=False).pvalue
        rows.append(dict(kalshi_price=f"{int(b*100)}-{int(b*100)+9}c", break_week=g.week.iloc[k].date(), before_pts=round(before.mean(), 2), after_pts=round(after.mean(), 2),
                         weeks_before=len(before), weeks_after=len(after), p_value=round(p, 4)))
    wk.append(dict(kalshi_price=f"{int(b*100)}-{int(b*100)+9}c", weeks=len(y), snapshots=int(g.n.sum()), mean_win_minus_price_pts=round(np.average(y, weights=g.n), 2),
                   weekly_sd_pts=round(sig, 2), breaks_found=len(bk)))
    a.set_title(f"Kalshi {int(b*100)}-{int(b*100)+9}c: win % minus price, by week (pts)", fontsize=10); a.grid(alpha=.3)
fig.suptitle('Kalshi 15-min BTC: actual win rate minus quoted price, weekly (blue), change-point segments (orange)', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig('work/kalshi_accuracy_by_price_weekly.png', dpi=110)
B = pd.DataFrame(rows); P = pd.DataFrame(wk); print(P.to_string()); print(B.to_string())
with pd.ExcelWriter('work/ml_replay_baselines.xlsx', mode='a', engine='openpyxl') as w:
    P.to_excel(w, sheet_name='Kalshi_by_price_summary', index=False); B.to_excel(w, sheet_name='Change_points', index=False)
    W.assign(err_pts=W.err * 100).drop(columns='err').to_excel(w, sheet_name='Kalshi_by_price_weekly', index=False)
