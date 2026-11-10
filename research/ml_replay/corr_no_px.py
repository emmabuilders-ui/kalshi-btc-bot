import numpy as np, pandas as pd
src = open('monthly.py').read().split("months = sorted")[0].replace("'min_left','res','mid'", "'min_left','res','mid','ask','bid'")
exec(src)
A = A[A.month >= '2026-09'].dropna(subset=['ask', 'bid'] + M).copy()
fee = lambda p: .07 * p * (1 - p)
rows = []; out = {}
for m, G in (('p_corr_cal30d', 3), ('p_corr_wk', 2), ('p_corr_of', 1)):
    px = 1 - A.bid
    T = A.assign(px=px, win=1 - A.res)[(100 * ((1 - A[m]) - px - fee(px)) >= G) & (px >= .70) & (px < .99)]
    T = T.sort_values('min_left', ascending=False).drop_duplicates('ticker'); T['pnl'] = T.win - T.px - fee(T.px)
    T['price'] = (np.floor(T.px * 20) / 20 * 100).astype(int).astype(str) + 'c'
    T['time'] = pd.cut(T.min_left, [0, 2, 5, 9, 14], labels=['1-2', '3-5', '6-9', '10-14']).astype(str)
    for by in ('price', 'time'):
        for k, h in T.groupby(by):
            rows.append(dict(model=N[m], gap=G, by=by, level=k, n=len(h), win_vs_price=f'{100*h.win.mean():.1f} vs {100*h.px.mean():.1f}', edge=round(100 * h.pnl.mean(), 2)))
O = pd.DataFrame(rows); O.to_excel('/mnt/project-files/strategy/ml_replay/correction_no_by_price_time.xlsx', index=False)
pd.set_option('display.width', 200); print(O.to_string(index=False))
