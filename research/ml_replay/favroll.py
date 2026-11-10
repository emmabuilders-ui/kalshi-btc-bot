# Favourite 80-99c with 6-14 min left: buy favourite side at ask; split by Rolling agrees (p_roll side prob > ask + fee).
# ONE trade per contract: first qualifying minute (earliest, most minutes left). Months, SE, by band and time.
import numpy as np, pandas as pd
A = pd.read_parquet('work/models_scored.parquet', columns=['ticker','month','min_left','ask','bid','res','p_roll'])
A = A[A.month >= '2026-03'].dropna(subset=['p_roll','ask','bid'])
y = A.assign(side='YES', px=A.ask, win=A.res, prob=A.p_roll); n = A.assign(side='NO', px=1-A.bid, win=1-A.res, prob=1-A.p_roll)
T = pd.concat([y, n]); T = T[(T.px >= .8) & (T.px < 1) & (T.min_left >= 6)]
T['fee'] = .07*T.px*(1-T.px); T['pnl'] = T.win - T.px - T.fee
T['per'] = np.where(T.month >= '2026-09', 'Sep-Oct', 'Mar-Aug'); T['band'] = (T.px*100//10*10).astype(int).astype(str) + 's'
T['tm'] = np.where(T.min_left <= 9, '6-9', '10-14'); T['rule'] = np.where(T.prob > T.px + T.fee, 'Rolling agrees', 'Rolling disagrees')
def first(df): return df.sort_values('min_left', ascending=False).drop_duplicates('ticker')
def s(g):
    mo = g.groupby('month').pnl.mean()
    return pd.Series(dict(trades=len(g), cell=f'{100*g.win.mean():.1f} vs {100*g.px.mean():.1f}', win_minus_px=round(100*(g.win.mean()-g.px.mean()),2),
        after_fee_c=round(100*g.pnl.mean(),2), se_c=round(100*g.pnl.std()/np.sqrt(len(g)),2), t=round(g.pnl.mean()/(g.pnl.std()/np.sqrt(len(g))),2),
        months_pos=f'{(mo>0).sum()}/{len(mo)}', by_month=' '.join(f'{m[5:]}:{100*v:+.1f}' for m, v in mo.items())))
out = {}
# (a) overall: each rule, first qualifying minute per contract
out['overall'] = pd.concat([first(T[T.rule==r]).assign(rule=r) for r in T.rule.unique()]).groupby(['rule','per']).apply(s)
# (b) per band x time cell, first per contract within cell
parts = []
for (r, b, t), g in T.groupby(['rule','band','tm']): parts.append(first(g))
F = pd.concat(parts)
out['band_time'] = F.groupby(['band','tm','rule','per']).apply(s)
out['band'] = pd.concat([first(g) for _, g in T.groupby(['rule','band'])]).groupby(['band','rule','per']).apply(s)
pd.set_option('display.width', 260); pd.set_option('display.max_colwidth', 80)
for k, v in out.items(): print(k); print(v.to_string()); print()
with pd.ExcelWriter('work/favourite_rolling_filter_one_per_contract.xlsx') as w:
    for k, v in out.items(): v.reset_index().to_excel(w, sheet_name=k, index=False)
