# Every model: buy the side the model favours when model prob > entry price + fee (YES at ask, NO at 1-bid).
# One trade per contract per cell (first qualifying minute). Cells: model x gap bucket x price band x time group. Mar-Aug (out-of-fold only) and Sep-Oct.
import numpy as np, pandas as pd
src = open('monthly.py').read().split("months = sorted")[0].replace("'min_left','res','mid'", "'min_left','res','mid','ask','bid'")
exec(src)
A = A.dropna(subset=['ask', 'bid'])
A['per'] = np.where(A.month >= '2026-09', 'Sep-Oct', 'Mar-Aug')
A['time'] = pd.cut(A.min_left, [0, 2, 5, 9, 14], labels=['1-2 min', '3-5 min', '6-9 min', '10-14 min']).astype(str)
fee = lambda p: 0.07 * p * (1 - p)
rows_all, cells = [], []
for m in M[1:]:
    B = A.dropna(subset=[m])
    if m in INS: B = B[B.per == 'Sep-Oct']
    if B.empty: continue
    base = B[['ticker', 'month', 'per', 'min_left', 'time', 'res', 'ask', 'bid', m]]
    y = base.assign(side='YES', px=base.ask, win=base.res, prob=base[m]); n = base.assign(side='NO', px=1 - base.bid, win=1 - base.res, prob=1 - base[m])
    T = pd.concat([y, n]); T = T[(T.px > 0) & (T.px < 1)]
    T['fee'] = fee(T.px); T['gap'] = 100 * (T.prob - T.px - T.fee); T = T[T.gap > 0]
    T['pnl'] = T.win - T.px - T.fee
    T['gapb'] = pd.cut(T.gap, [0, 1, 2, 3, 5, 100], labels=['0-1', '1-2', '2-3', '3-5', '5+']).astype(str)
    T['band'] = (np.clip(np.floor(T.px * 100), 1, 99) // 10 * 10).astype(int)
    T = T.sort_values('min_left', ascending=False)
    # overall by gap bucket: first qualifying minute per contract per gap bucket
    for (per, gb), g in T.drop_duplicates(['ticker', 'gapb']).groupby(['per', 'gapb']):
        rows_all.append(dict(model=N[m], per=per, gap=gb, trades=len(g), win=g.win.mean(), px=g.px.mean(), pnl=g.pnl.mean(), se=g.pnl.std() / np.sqrt(len(g)),
                             months=f"{(g.groupby('month').pnl.mean() > 0).sum()}/{g.month.nunique()}"))
    F = T.drop_duplicates(['ticker', 'gapb', 'band', 'time'])
    for (per, gb, b, t), g in F.groupby(['per', 'gapb', 'band', 'time']):
        cells.append(dict(model=N[m], per=per, gap=gb, band=f'{b}-{b+9}c', time=t, trades=len(g), win=g.win.mean(), px=g.px.mean(), pnl=g.pnl.mean(),
                          se=g.pnl.std() / np.sqrt(len(g)) if len(g) > 1 else np.nan, months=f"{(g.groupby('month').pnl.mean() > 0).sum()}/{g.month.nunique()}"))
    print('done', N[m], flush=True)
R = pd.DataFrame(rows_all); C = pd.DataFrame(cells)
for D in (R, C):
    D['cell'] = (100 * D.win).round(1).astype(str) + ' vs ' + (100 * D.px).round(1).astype(str)
    D['after_fee_c'] = (100 * D.pnl).round(2); D['se_c'] = (100 * D.se).round(2); D['t'] = (D.pnl / D.se).round(2)
R.to_parquet('work/allgaps_overall.parquet'); C.to_parquet('work/allgaps_cells.parquet')
