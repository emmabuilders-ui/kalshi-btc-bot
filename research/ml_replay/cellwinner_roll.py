# Adaptive version: each day, for each price band x time group, pick the model with the lowest Brier over the trailing W days
# (settled contracts only, before that day). Trade that model's gap the next day. Candidates: out-of-fold models + Kalshi.
import numpy as np, pandas as pd, sys
src = open('monthly.py').read().split("months = sorted")[0].replace("'min_left','res','mid'", "'min_left','res','mid','ask','bid'")
exec(src)
PICK = ['p_roll', 'p_direct', 'p_corr', 'p_regime']
A = A.dropna(subset=['ask', 'bid'] + PICK).copy()
A['time'] = pd.cut(A.min_left, [0, 2, 5, 9, 14], labels=['1-2', '3-5', '6-9', '10-14']).astype(str)
A['day'] = (A.snap_ts // 86400).astype(int)
A['per'] = np.where(A.month >= '2026-09', 'Sep-Oct', 'Mar-Aug')
for m in PICK + ['p_kalshi']: A['se_' + m] = (A[m] - A.res) ** 2
D = A.groupby(['band', 'time', 'day'])[['se_' + m for m in PICK + ['p_kalshi']]].agg(['sum', 'count'])
fee = lambda p: .07 * p * (1 - p)
out = []
for W in (7, 14, 30):
    # rolling sums per cell over previous W days (exclusive of current day)
    S = A.groupby(['band', 'time', 'day'])[['se_' + m for m in PICK]].sum()
    C = A.groupby(['band', 'time', 'day']).size().rename('n')
    S = S.join(C).reset_index().sort_values('day')
    pick = {}
    for (b, t), g in S.groupby(['band', 'time']):
        g = g.drop(columns=['band', 'time']).set_index('day').reindex(range(g.day.min(), g.day.max() + 1), fill_value=0)
        roll = g.rolling(W, min_periods=1).sum().shift(1)          # trailing W days, before today
        for d, r in roll.iterrows():
            if r['n'] and r['n'] >= 100:
                pick[(b, t, d)] = min(PICK, key=lambda m: r['se_' + m] / r['n'])
    A['pm'] = [pick.get(k) for k in zip(A.band, A.time, A.day)]
    B = A.dropna(subset=['pm']).copy()
    B['pw'] = B.lookup(B.index, B.pm) if hasattr(B, 'lookup') else [B.at[i, m] for i, m in zip(B.index, B.pm)]
    # accuracy of the adaptive pick vs Kalshi
    for per, h in B.groupby('per'):
        out.append(dict(window=W, kind='Brier vs Kalshi x1000', per=per, value=round(1000 * (np.mean((h.pw - h.res) ** 2) - np.mean((h.p_kalshi - h.res) ** 2)), 3), n=len(h)))
    for G in (1, 2, 3, 5):
        for side in ('YES', 'NO'):
            px = B.ask if side == 'YES' else 1 - B.bid; wn = B.res if side == 'YES' else 1 - B.res; pr = B.pw if side == 'YES' else 1 - B.pw
            T = B.assign(px=px, win=wn, g=100 * (pr - px - fee(px)))
            T = T[(T.g >= G) & (T.px > .02) & (T.px < .99)].sort_values('min_left', ascending=False).drop_duplicates('ticker')
            T['pnl'] = T.win - T.px - fee(T.px)
            for per, h in T.groupby('per'):
                mo = h.groupby('month').pnl.mean()
                out.append(dict(window=W, kind=f'trade gap>={G} {side}', per=per, n=len(h), value=round(100 * h.pnl.mean(), 2),
                                win_vs_price=f'{100*h.win.mean():.1f} vs {100*h.px.mean():.1f}', t=round(h.pnl.mean() / (h.pnl.std() / np.sqrt(len(h))), 2), months=f'{(mo>0).sum()}/{len(mo)}'))
    print('done', W, flush=True)
R = pd.DataFrame(out)
W2 = R.pivot_table(index=['window', 'kind'], columns='per', values=['n', 'win_vs_price', 'value', 't', 'months'], aggfunc='first')
W2.columns = [f'{b} {a}' for a, b in W2.columns]
W2 = W2[[c for c in [f'{p} {x}' for p in ['Mar-Aug', 'Sep-Oct'] for x in ['n', 'win_vs_price', 'value', 't', 'months']] if c in W2.columns]].reset_index()
W2.to_excel('work/cell_winner_daily_repick.xlsx', index=False)
pd.set_option('display.width', 250); print(W2.to_string(index=False))
