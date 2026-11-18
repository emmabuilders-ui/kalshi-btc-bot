import numpy as np, pandas as pd
S = pd.read_parquet('work/lastmin.parquet')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300); pd.set_option('display.float_format', lambda x: f'{x:.2f}')
fee = lambda p: 0.07*p*(1-p); R = []
for side in ('YES','NO'):
    d = S.copy(); d['side'] = side
    d['ask_s'] = d.ask if side=='YES' else 1-d.bid; d['mid_s'] = d.mid if side=='YES' else 1-d.mid
    d['p_s'] = d.p if side=='YES' else 1-d.p; d['win'] = d.res if side=='YES' else 1-d.res
    d = d[(d.ask_s>=.01)&(d.ask_s<=.99)]; d['cost'] = d.ask_s+fee(d.ask_s); d['ret'] = (d.p_s-d.cost)/d.cost; d['pnl'] = d.win-d.cost; R.append(d)
A = pd.concat(R); A['bucket'] = (np.floor(A.ask_s*10).clip(0,9)*10).astype(int)
A['bk'] = (A.brier_m if False else 0)
Q = A[A.ret>=.05].sort_values(['ticker','off','ret'], ascending=[True,False,False]).drop_duplicates('ticker')   # earliest (45 s) look first
def summ(g): n=len(g); return pd.Series(dict(n=n, px=100*g.ask_s.mean(), win=100*g.win.mean(), be=100*g.cost.mean(), ev=100*g.pnl.mean(), med=100*g.pnl.median(), se=100*g.pnl.std()/np.sqrt(max(n,1))))
print('qualifying looks by seconds left'); print(A[A.ret>=.05].groupby(['period','off']).apply(summ, include_groups=False))
print('by side'); print(A[A.ret>=.05].groupby(['period','side','off']).apply(summ, include_groups=False))
A['pgrp'] = pd.cut(A.ask_s*100, [0,10,50,90,100], right=False)
print('by price group (qualifying)'); print(A[A.ret>=.05].groupby(['period','pgrp','off'], observed=True).apply(summ, include_groups=False))
print('first look per contract'); print(Q.groupby('period').apply(summ, include_groups=False))
print('model vs kalshi Brier'); 
A['bm']=(A.p_s-A.win)**2; A['bk']=(A.mid_s-A.win)**2
print(A[A.side=='YES'].groupby(['period','off']).apply(lambda g: pd.Series(dict(n=len(g), brier_kal=1000*g.bk.mean(), brier_model=1000*g.bm.mean())), include_groups=False))
with pd.ExcelWriter('/mnt/project-files/strategy/correction_unleashed/correction_unleashed_last_minute.xlsx') as w:
    A[A.ret>=.05].groupby(['period','side','off','bucket']).apply(summ, include_groups=False).reset_index().to_excel(w, sheet_name='qualifying by side px secs', index=False)
    A.assign(bm=A.bm, bk=A.bk).groupby(['period','side','off','bucket']).apply(lambda g: pd.Series(dict(n=len(g), mid=100*g.mid_s.mean(), win=100*g.win.mean(), model_p=100*g.p_s.mean(), brier_kal=1000*g.bk.mean(), brier_model=1000*g.bm.mean())), include_groups=False).reset_index().to_excel(w, sheet_name='model test side px secs', index=False)
