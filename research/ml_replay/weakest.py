# Rank Kalshi's weakest cells by % wrong = actual win % minus Kalshi's average price (pts). Month x time-left x 10c band.
import numpy as np, pandas as pd
exec(open('monthly.py').read().split("months = sorted")[0])  # reuse loading/merging
A['contract_day'] = A.ticker
rows = []
for (mo, t, b), g in A.groupby(['month','time','band'], observed=True):
    if len(g) < 200: continue
    n_c = g.ticker.nunique()
    win, kp = 100*g.res.mean(), 100*g.p_kalshi.mean()
    # SE on contracts (snapshots within a contract are correlated)
    cres = g.groupby('ticker').res.first(); se = 100*np.sqrt(cres.mean()*(1-cres.mean())/max(n_c,1)) if 0<cres.mean()<1 else np.nan
    r = dict(month=mo, time_left=t, kalshi_price=f'{b}-{b+9}c', snapshots=len(g), contracts=n_c, actual_win_pct=round(win,1), kalshi_avg_pct=round(kp,1),
             kalshi_wrong_pts=round(win-kp,1), abs_wrong=round(abs(win-kp),1), z=round((win-kp)/se,2) if se==se and se>0 else np.nan,
             side_that_paid=('YES underpriced' if win>kp else 'NO underpriced'))
    for m in M[1:]:
        if m in INS and mo < '2026-09': continue
        h = g.dropna(subset=[m])
        if len(h) < 0.95*len(g): continue
        r[N[m]+' wrong pts'] = round(100*(h.res.mean()-h[m].mean()),1)
    rows.append(r)
W = pd.DataFrame(rows).sort_values('abs_wrong', ascending=False)
# pooled over all 8 months per time x price, with months same sign
P = []
for (t, b), g in A.groupby(['time','band'], observed=True):
    mm = g.groupby('month').apply(lambda h: 100*(h.res.mean()-h.p_kalshi.mean()))
    pooled = 100*(g.res.mean()-g.p_kalshi.mean())
    P.append(dict(time_left=t, kalshi_price=f'{b}-{b+9}c', snapshots=len(g), actual_win_pct=round(100*g.res.mean(),1), kalshi_avg_pct=round(100*g.p_kalshi.mean(),1),
                  kalshi_wrong_pts=round(pooled,1), abs_wrong=round(abs(pooled),1), months_same_sign=f"{int((np.sign(mm)==np.sign(pooled)).sum())} of {len(mm)}",
                  MarAug_wrong=round(100*(g[g.month<'2026-09'].res.mean()-g[g.month<'2026-09'].p_kalshi.mean()),1),
                  SepOct_wrong=round(100*(g[g.month>='2026-09'].res.mean()-g[g.month>='2026-09'].p_kalshi.mean()),1)))
P = pd.DataFrame(P).sort_values('abs_wrong', ascending=False)
pd.set_option('display.width',250)
print(W[['month','time_left','kalshi_price','contracts','actual_win_pct','kalshi_avg_pct','kalshi_wrong_pts','z']].head(20).to_string(index=False))
print('\n|z|>=2 cells:', (W.z.abs()>=2).sum(), 'of', len(W), ' expected by chance ~', round(0.046*len(W),1))
print(P.head(15).to_string(index=False))
with pd.ExcelWriter('work/kalshi_weakest_cells.xlsx') as w:
    W.to_excel(w, sheet_name='Month_time_price_sorted', index=False); P.to_excel(w, sheet_name='Pooled_Mar-Oct_sorted', index=False)
