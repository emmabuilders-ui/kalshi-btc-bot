# Condition = cushion top third OR BTC broke out (60-min pre-open levels) in the side's direction.
import numpy as np, pandas as pd
T0 = 1764460800
b = pd.Series(np.load('work/brti.npy').astype(float)).ffill().to_numpy(); nmin = len(b)//60; Mm = b[:nmin*60].reshape(nmin, 60)
hi60 = pd.Series(Mm.max(1)).rolling(60).max().to_numpy(); lo60 = pd.Series(Mm.min(1)).rolling(60).min().to_numpy()
X = pd.read_parquet('work/models_scored.parquet', columns=['ticker', 'snap_ts', 'open_ts', 'brti']).drop_duplicates(['ticker', 'snap_ts'])
def s_(g):
    mo = g.groupby('month').pnl.mean()
    return f"{len(g):>5} tr  {100*g.win.mean():.1f}% vs {100*g.px.mean():.1f}c  BE {100*(g.px+g.fee).mean():.1f}  {100*g.pnl.mean():+.2f}c  t {g.pnl.mean()/(g.pnl.std()/np.sqrt(len(g))):.2f}  mo {(mo>0).sum()}/{len(mo)}  /day {len(g)/g.day.nunique() if 'day' in g else 0:.1f}"
rows = []
for m, mn in [('p_direct', 'Direct'), ('p_corr', 'Correction')]:
    T = pd.read_parquet(f'work/addf_{m}.parquet').merge(X, on=['ticker', 'snap_ts'], how='left')
    T['day'] = pd.to_datetime(T.snap_ts, unit='s').dt.date
    cz = T[T.per == 'Mar-Aug']['Cushion (z to strike, own side)'].quantile(2/3)
    idx = ((T.open_ts - T0)//60 - 1).astype(int)
    brk = np.where(T.side == 'YES', T.brti > hi60[idx], T.brti < lo60[idx])
    cus = T['Cushion (z to strike, own side)'] > cz
    for name, mask in [('no condition', np.ones(len(T), bool)), ('cushion only', cus), ('breakout only', brk), ('EITHER', cus | brk), ('BOTH', cus & brk), ('NEITHER', ~(cus | brk))]:
        for side in ['both', 'NO', 'YES']:
            g0 = T[mask & ((T.side == side) | (side == 'both'))]
            for per, g in g0.groupby('per'):
                mo = g.groupby('month').pnl.mean()
                rows.append(dict(rule=mn, condition=name, side=side, per=per, trades=len(g), per_day=round(len(g)/g.day.nunique(), 1), win_pct=round(100*g.win.mean(), 1),
                                 avg_price=round(100*g.px.mean(), 1), breakeven=round(100*(g.px+g.fee).mean(), 1), after_fee_c=round(100*g.pnl.mean(), 2),
                                 t=round(g.pnl.mean()/(g.pnl.std()/np.sqrt(len(g))), 2), months_pos=f'{(mo>0).sum()}/{len(mo)}'))
R = pd.DataFrame(rows)
W = R.pivot_table(index=['rule', 'condition', 'side'], columns='per', values=['trades', 'per_day', 'win_pct', 'avg_price', 'breakeven', 'after_fee_c', 't', 'months_pos'], aggfunc='first', sort=False)
W.columns = [f'{b} {a}' for a, b in W.columns]
W = W[[f'{p} {c}' for p in ['Mar-Aug', 'Sep-Oct'] for c in ['trades', 'per_day', 'win_pct', 'avg_price', 'breakeven', 'after_fee_c', 't', 'months_pos']]].reset_index()
W.to_excel('work/model_gap_cushion_or_breakout.xlsx', index=False)
pd.set_option('display.width', 300); print(W.to_string(index=False))
