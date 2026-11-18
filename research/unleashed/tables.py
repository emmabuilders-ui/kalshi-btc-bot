import numpy as np, pandas as pd
A = pd.read_parquet('work/unleashed_all.parquet'); Q = pd.read_parquet('work/unleashed_trades.parquet')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 30); pd.set_option('display.float_format', lambda x: f'{x:.2f}')
def summ(g):
    n = len(g)
    return pd.Series(dict(n=n, px=100*g.ask_s.mean(), win=100*g.win.mean(), be=100*g.cost.mean(), edge=100*g.pnl.mean(), med=100*g.pnl.median(),
        se=100*g.pnl.std()/np.sqrt(max(n,1)), gap=100*g.gap.mean()))
# MODEL TEST (all snapshots, no rules)
def mt(g):
    return pd.Series(dict(n=len(g), mid=100*g.mid_s.mean(), kal_win_minus_mid=100*(g.win-g.mid_s).mean(), model_p=100*g.p_s.mean(),
        model_win_minus_p=100*(g.win-g.p_s).mean(), brier_kal=1000*g.brier_k.mean(), brier_model=1000*g.brier_m.mean(), diff_x1000=1000*(g.brier_m-g.brier_k).mean()))
print('MODEL: by period x side'); print(A.groupby(['per','side']).apply(mt, include_groups=False))
print('MODEL: by period x price bucket (YES side prices only, 10c of YES mid)')
A['ybucket'] = (np.floor((A.mid_s if False else A.mid_s)*10).clip(0,9)*10).astype(int)
Y = A[A.side=='YES']; print(Y.assign(yb=(np.floor(Y.mid*10).clip(0,9)*10).astype(int)).groupby(['per','yb']).apply(mt, include_groups=False))
print('MODEL: by period x minute left (YES)'); print(Y.groupby(['per','min_left']).apply(mt, include_groups=False))
# deciles of pred return: all snapshot-sides
for per, g in A.groupby('per'):
    g = g.copy(); g['dec'] = pd.qcut(g.ret, 10, labels=False, duplicates='drop')
    r = g.groupby('dec').apply(lambda h: pd.concat([summ(h), pd.Series(dict(ret_lo=100*h.ret.min(), ret_hi=100*h.ret.max()))]), include_groups=False)
    print('DECILES of predicted return', per); print(r)
# qualifying by predicted-return band
Q['band'] = pd.cut(Q.ret*100, [5,10,15,20,30,50,1000], right=False)
print('BOT trades by predicted return band'); print(Q.groupby(['per','band'], observed=True).apply(summ, include_groups=False))
# price groups
Q['pg'] = pd.cut(Q.ask_s*100, [0,10,20,30,50,70,100], right=False)
print('BOT trades by price group'); print(Q.groupby(['per','pg'], observed=True).apply(summ, include_groups=False))
# flat 3pt comparison (all prices, any minute) first look
B = A[A.gap >= 0.03].sort_values(['ticker','k','gap'], ascending=[True,True,False]).drop_duplicates('ticker')
print('FLAT 3pt gap any side/price/minute'); print(B.groupby('per').apply(summ, include_groups=False))
# original rule within data (NO 70-99, 6-14, gap>=3)
O = A[(A.side=='NO')&(A.ask_s>=.70)&(A.ask_s<=.99)&(A.min_left.between(6,14))&(A.gap>=.03)].sort_values(['ticker','k']).drop_duplicates('ticker')
print('ORIGINAL corr rule replay'); print(O.groupby('per').apply(summ, include_groups=False))
