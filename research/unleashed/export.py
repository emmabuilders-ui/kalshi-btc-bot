import numpy as np, pandas as pd
A = pd.read_parquet('work/unleashed_all.parquet'); Q = pd.read_parquet('work/unleashed_trades.parquet')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); pd.set_option('display.float_format', lambda x: f'{x:.2f}')
def summ(g):
    n = len(g); mo = g.groupby('month').pnl.mean()
    return pd.Series(dict(trades=n, avg_price_c=100*g.ask_s.mean(), win_pct=100*g.win.mean(), breakeven_c=100*g.cost.mean(),
        win_minus_price_c=100*(g.win-g.ask_s).mean(), ev_avg_c=100*g.pnl.mean(), ev_median_c=100*g.pnl.median(), se_c=100*g.pnl.std()/np.sqrt(max(n,1)),
        avg_gap_c=100*g.gap.mean(), months_pos=int((mo>0).sum()), months=len(mo)))
def grid(D): return D.groupby(['per','side','min_left','bucket']).apply(summ, include_groups=False).reset_index()
def dec(D, name):
    out = []
    for per, g in D.groupby('per'):
        g = g.copy(); g['decile'] = pd.qcut(g.ret, 10, labels=False, duplicates='drop') + 1
        r = g.groupby('decile').apply(summ, include_groups=False).reset_index(); r.insert(0, 'per', per)
        r['ret_lo_pct'] = g.groupby('decile').ret.min().values*100; r['ret_hi_pct'] = g.groupby('decile').ret.max().values*100; out.append(r)
    return pd.concat(out)
with pd.ExcelWriter('/mnt/project-files/strategy/correction_unleashed/correction_unleashed_results.xlsx') as w:
    pd.DataFrame({'note': ['Correction Unleashed ML test. Mar-Aug = monthly out-of-fold model; Sep-Oct = frozen Dec-Aug fit (shipped model JSON). Dec-Feb have no out-of-sample prediction.',
        'Price = ask of the side bought; break-even = ask + fee 0.07p(1-p); EV = win - ask - fee per contract. Median of a win/loss payoff is always one of two values, read with the average.']}).to_excel(w, sheet_name='README', index=False)
    grid(Q).to_excel(w, sheet_name='BOT min x px x side', index=False)
    grid(A[A.qual]).to_excel(w, sheet_name='ALL qualifying looks grid', index=False)
    Q.groupby(['per','side']).apply(summ, include_groups=False).reset_index().to_excel(w, sheet_name='BOT by side', index=False)
    Q.groupby(['per','side','bucket']).apply(summ, include_groups=False).reset_index().to_excel(w, sheet_name='BOT side x price', index=False)
    Q.groupby(['per','side','min_left']).apply(summ, include_groups=False).reset_index().to_excel(w, sheet_name='BOT side x minute', index=False)
    dec(A, 'all').to_excel(w, sheet_name='Deciles of predicted return', index=False)
    mt = lambda g: pd.Series(dict(n=len(g), kalshi_mid_c=100*g.mid_s.mean(), kalshi_win_minus_mid_c=100*(g.win-g.mid_s).mean(), model_p_c=100*g.p_s.mean(), model_win_minus_p_c=100*(g.win-g.p_s).mean(), brier_kalshi_x1000=1000*g.brier_k.mean(), brier_model_x1000=1000*g.brier_m.mean(), brier_diff_x1000=1000*(g.brier_m-g.brier_k).mean()))
    A.assign(mbucket=(np.floor(A.mid_s*10).clip(0,9)*10).astype(int)).groupby(['per','side','min_left','mbucket']).apply(mt, include_groups=False).reset_index().to_excel(w, sheet_name='MODEL min x px x side', index=False)
# isolation
Q['p'] = Q.ask_s
for name, m in [('all unleashed', Q.ask_s>0), ('ask>=70c', Q.ask_s>=.7), ('ask>=70c & 6-14min', (Q.ask_s>=.7)&Q.min_left.between(6,14)), ('NO only', Q.side=='NO'), ('NO ask>=70c', (Q.side=='NO')&(Q.ask_s>=.7)),('ask 50-70c',(Q.ask_s>=.5)&(Q.ask_s<.7)),('ask<50c',Q.ask_s<.5),('ask 30-50c',(Q.ask_s>=.3)&(Q.ask_s<.5))]:
    print(name); print(Q[m].groupby('per').apply(summ, include_groups=False)[['trades','win_pct','breakeven_c','ev_avg_c','ev_median_c','se_c','months_pos']].to_string())
