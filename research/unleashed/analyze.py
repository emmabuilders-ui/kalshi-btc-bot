import numpy as np, pandas as pd
S = pd.read_parquet('work/unleashed_pred.parquet')
S = S[S.snap_ts >= pd.Timestamp('2026-03-01', tz='UTC').timestamp()].dropna(subset=['p', 'ask', 'bid']).copy()
S['per'] = np.where(S.period == 'Sep-Oct', 'Sep-Oct', 'Mar-Aug')
fee = lambda p: 0.07 * p * (1 - p)
rows = []
for side in ('YES', 'NO'):
    d = S.copy(); d['side'] = side
    d['ask_s'] = d.ask if side == 'YES' else 1 - d.bid
    d['mid_s'] = d.mid if side == 'YES' else 1 - d.mid
    d['p_s'] = d.p if side == 'YES' else 1 - d.p
    d['win'] = d.res if side == 'YES' else 1 - d.res
    d = d[(d.ask_s >= .01) & (d.ask_s <= .99)]
    d['cost'] = d.ask_s + fee(d.ask_s)
    d['ret'] = (d.p_s - d.cost) / d.cost
    d['gap'] = d.p_s - d.cost
    d['pnl'] = d.win - d.cost
    rows.append(d)
A = pd.concat(rows, ignore_index=True)
A['bucket'] = (np.floor(A.ask_s * 10).clip(0, 9) * 10).astype(int)
A['qual'] = A.ret >= 0.05
A['brier_m'] = (A.p_s - A.win) ** 2; A['brier_k'] = (A.mid_s - A.win) ** 2
A.to_parquet('work/unleashed_all.parquet')
# bot: first qualifying look per contract (earliest minute; both sides -> bigger return)
Q = A[A.qual].sort_values(['ticker', 'k', 'ret'], ascending=[True, True, False]).drop_duplicates('ticker')
Q.to_parquet('work/unleashed_trades.parquet')
def summ(g):
    n = len(g); mo = g.groupby('month').pnl.mean() if 'month' in g else None
    return pd.Series(dict(n=n, avg_px=100 * g.ask_s.mean(), win=100 * g.win.mean(), be=100 * g.cost.mean(),
        edge=100 * g.pnl.mean(), med=100 * g.pnl.median(), se=100 * g.pnl.std() / np.sqrt(max(n, 1)),
        gap=100 * g.gap.mean(), pred=100 * g.ret.mean(),
        mpos=(mo > 0).sum() if mo is not None else np.nan, mths=len(mo) if mo is not None else np.nan))
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); pd.set_option('display.float_format', lambda x: f'{x:.2f}')
print('BOT TRADES (first look)'); print(Q.groupby('per').apply(summ, include_groups=False))
print(Q.groupby(['per', 'side']).apply(summ, include_groups=False))
print(Q.groupby(['per', 'side', 'bucket']).apply(summ, include_groups=False))
print(Q.groupby(['per', 'min_left']).apply(summ, include_groups=False))
print('ALL QUAL SNAPSHOTS'); print(A[A.qual].groupby('per').apply(summ, include_groups=False))
