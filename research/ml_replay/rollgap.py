# Rolling vs Kalshi trades bucketed by trade gap = Rolling prob - (entry price + fee), in pts.
import numpy as np, pandas as pd
exec(open('rolltrade.py').read().split("def summ")[0])
T['gap'] = 100*(T.prob - T.px - T.fee)
bins = [0, 0.5, 1, 1.5, 2, 3, 100]; labs = ['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-3', '3+']
T['gap_bucket'] = pd.cut(T.gap, bins, labels=labs)
def s(g):
    mo = g.groupby('month').pnl.mean(); c = g.groupby('ticker').pnl.mean()
    return pd.Series(dict(trades=len(g), winners=int(g.win.sum()), losers=int(len(g)-g.win.sum()), win_pct=round(100*g.win.mean(),2), avg_price=round(100*g.px.mean(),2),
        avg_gap=round(g.gap.mean(),2), win_minus_price=round(100*(g.win.mean()-g.px.mean()),2), after_fee_c=round(100*g.pnl.mean(),2), se_c=round(100*c.std()/np.sqrt(len(c)),2), months_pos=f'{(mo>0).sum()}/{len(mo)}'))
out = {tuple(k): T.groupby(k, observed=True).apply(s).reset_index() for k in [['per','gap_bucket'], ['per','side','gap_bucket'], ['per','side','band','gap_bucket'], ['per','side','time','gap_bucket']]}
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
print(out['per_gap_bucket' if False else list(out)[0]].to_string(index=False)); print(); print(out[list(out)[1]].to_string(index=False))
# avg gap of winners vs losers
print(T.groupby(['per', 'win']).gap.describe()[['count','mean','50%']].round(2))
with pd.ExcelWriter('work/rolling_trades_by_gap.xlsx') as w:
    for k, v in out.items(): v.to_excel(w, sheet_name='_'.join(k)[:31], index=False)
