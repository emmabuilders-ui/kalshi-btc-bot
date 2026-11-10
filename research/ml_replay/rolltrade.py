# Rolling 30-day recal vs Kalshi: buy the side Rolling favours when Rolling prob > entry price + fee.
# YES at ask, NO at 1-bid. One trade per contract per (minute-left, side) cell; win% - price - fee by price band and minutes left.
import numpy as np, pandas as pd
A = pd.read_parquet('work/models_scored.parquet', columns=['ticker','month','day','min_left','ask','bid','res','p_roll'])
A = A[A.month >= '2026-03'].dropna(subset=['p_roll','ask','bid'])
fee = lambda p: 0.07*p*(1-p)
y = A.assign(side='YES', px=A.ask, win=A.res, prob=A.p_roll); n = A.assign(side='NO', px=1-A.bid, win=1-A.res, prob=1-A.p_roll)
T = pd.concat([y, n]); T = T[(T.px > 0) & (T.px < 1)]
T['fee'] = fee(T.px); T = T[T.prob > T.px + T.fee]
T['per'] = np.where(T.month >= '2026-09', 'Sep-Oct', 'Mar-Aug')
T['band'] = (np.clip(np.floor(T.px*100),1,99)//10*10).astype(int)
T['time'] = pd.cut(T.min_left,[0,2,5,9,14],labels=['1-2 min','3-5 min','6-9 min','10-14 min'])
T = T.sort_values('min_left', ascending=False).drop_duplicates(['ticker','side','min_left'])  # one per contract per minute per side
T['pnl'] = T.win - T.px - T.fee
def summ(g):
    mo = g.groupby('month').pnl.mean()
    return pd.Series(dict(trades=len(g), contracts=g.ticker.nunique(), avg_price=100*g.px.mean(), win_pct=100*g.win.mean(), edge_pts=100*(g.win.mean()-g.px.mean()),
                          after_fee_c=100*g.pnl.mean(), se_c=100*g.groupby('ticker').pnl.mean().std()/np.sqrt(g.ticker.nunique()), months_pos=f'{(mo>0).sum()}/{len(mo)}'))
out = {}
for key in [['per'], ['per','side'], ['per','side','band'], ['per','side','time'], ['per','side','time','band'], ['per','side','min_left']]:
    out['_'.join(key)] = T.groupby(key, observed=True).apply(summ).round(2).reset_index()
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
for k in ['per','per_side','per_side_band','per_side_time']: print(out[k].to_string(index=False)); print()
G = out['per_side_time_band']
W = G.pivot_table(index=['side','time','band'], columns='per', values=['trades','win_pct','avg_price','after_fee_c','months_pos'], aggfunc='first')
W.columns = [f'{a} {b}' for a, b in W.columns]
W = W[[c for c in ['trades Mar-Aug','avg_price Mar-Aug','win_pct Mar-Aug','after_fee_c Mar-Aug','months_pos Mar-Aug','trades Sep-Oct','avg_price Sep-Oct','win_pct Sep-Oct','after_fee_c Sep-Oct','months_pos Sep-Oct']]]
both = W[(W['after_fee_c Mar-Aug']>0)&(W['after_fee_c Sep-Oct']>0)&(W['trades Sep-Oct']>=50)]
print('Cells positive after fee in BOTH periods (>=50 Sep-Oct trades):'); print(both.to_string())
with pd.ExcelWriter('work/rolling_vs_kalshi_trades.xlsx') as w:
    W.reset_index().to_excel(w, sheet_name='Side_time_price', index=False)
    for k, v in out.items(): v.to_excel(w, sheet_name=k[:31], index=False)
