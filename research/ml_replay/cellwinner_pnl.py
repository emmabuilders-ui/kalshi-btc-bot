# Each day, for each 10c price band x time group x side, pick the model (or plain Kalshi = buy every snapshot) whose
# gap trades made the most money over the trailing W days (settled, before that day). Trade only if that trailing
# profit > 0 and it had >= MINN contracts. One trade per contract (first qualifying minute). Fee 0.07 p(1-p).
import numpy as np, pandas as pd
src = open('monthly.py').read().split("months = sorted")[0].replace("'min_left','res','mid'", "'min_left','res','mid','ask','bid'")
exec(src)
PICK = ['p_roll', 'p_direct', 'p_corr', 'p_regime']
A = A.dropna(subset=['ask', 'bid'] + PICK).copy()
A['time'] = pd.cut(A.min_left, [0, 2, 5, 9, 14], labels=['1-2', '3-5', '6-9', '10-14']).astype(str)
A['day'] = (A.snap_ts // 86400).astype(int)
A['per'] = np.where(A.month >= '2026-09', 'Sep-Oct', 'Mar-Aug')
fee = lambda p: .07 * p * (1 - p)
rows = []
for side in ('YES', 'NO'):
    px = A.ask if side == 'YES' else 1 - A.bid
    T = A.assign(side=side, px=px, win=A.res if side == 'YES' else 1 - A.res)
    T = T[(T.px > .02) & (T.px < .99)]
    T['sband'] = (np.floor(T.px * 10) / 10).round(1)        # band of the price paid on this side
    T['pnl'] = T.win - T.px - fee(T.px)
    for G in (1, 2, 3, 5):
        for m in PICK + ['kalshi']:
            if m == 'kalshi':
                if G != 1: continue
                S = T
            else:
                pr = T[m] if side == 'YES' else 1 - T[m]
                S = T[100 * (pr - T.px - fee(T.px)) >= G]
            S = S.sort_values('min_left', ascending=False).drop_duplicates(['ticker', 'sband', 'time'])
            rows.append(S[['ticker', 'day', 'month', 'per', 'sband', 'time', 'min_left', 'px', 'win', 'pnl']].assign(side=side, model=m, G=G if m != 'kalshi' else 0))
R = pd.concat(rows, ignore_index=True)
print('candidate trades', len(R), flush=True)
out = []
for W in (14, 30, 60):
    for MINN in (20, 50):
        for G in (1, 2, 3, 5):
            C = R[(R.G == G) | (R.model == 'kalshi')]
            daily = C.groupby(['side', 'sband', 'time', 'model', 'day']).pnl.agg(['sum', 'count']).reset_index()
            days = np.arange(A.day.min(), A.day.max() + 1)
            pick = {}
            for k, g in daily.groupby(['side', 'sband', 'time', 'model']):
                s = g.set_index('day')[['sum', 'count']].reindex(days, fill_value=0).rolling(W, min_periods=1).sum().shift(1)
                for d, r in s.iterrows():
                    if r['count'] >= MINN:
                        key = k[:3] + (d,); avg = r['sum'] / r['count']
                        if avg > 0 and avg > pick.get(key, (None, 0))[1]: pick[key] = (k[3], avg)
            sel = pd.Series([pick.get(k, (None,))[0] for k in zip(C.side, C.sband, C.time, C.day)], index=C.index)
            Tr = C[sel.values == C.model.values]
            Tr = Tr.sort_values('min_left', ascending=False).drop_duplicates(['ticker', 'side'])   # one per contract per side
            for (side, per), h in Tr.groupby(['side', 'per']):
                mo = h.groupby('month').pnl.mean()
                out.append(dict(W=W, minn=MINN, G=G, side=side, per=per, n=len(h), win=round(100 * h.win.mean(), 1), price=round(100 * h.px.mean(), 1),
                                edge=round(100 * h.pnl.mean(), 2), t=round(h.pnl.mean() / (h.pnl.std() / np.sqrt(len(h))), 2), months=f'{(mo>0).sum()}/{len(mo)}',
                                mix=h.model.value_counts().to_dict()))
        print('done', W, MINN, flush=True)
O = pd.DataFrame(out)
O.to_pickle('work/cellwinner_pnl.pkl')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
print(O.drop(columns='mix').to_string(index=False))
