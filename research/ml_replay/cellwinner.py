# Per price band x time group: the model with the best Mar-Aug Brier (out-of-fold models + Kalshi). Does it stay best in Sep-Oct?
# Then trade that model's gap (>= G pts over ask+fee), first qualifying minute per contract.
import numpy as np, pandas as pd
src = open('monthly.py').read().split("months = sorted")[0].replace("'min_left','res','mid'", "'min_left','res','mid','ask','bid'")
exec(src)
A = A.dropna(subset=['ask', 'bid']).copy()
A['per'] = np.where(A.month >= '2026-09', 'Sep-Oct', 'Mar-Aug')
A['time'] = pd.cut(A.min_left, [0, 2, 5, 9, 14], labels=['1-2', '3-5', '6-9', '10-14']).astype(str)
PICK = ['p_roll', 'p_direct', 'p_corr', 'p_regime']
A = A.dropna(subset=PICK)
rows = []; win = {}
for (b, t), g in A.groupby(['band', 'time']):
    tr, te = g[g.per == 'Mar-Aug'], g[g.per == 'Sep-Oct']
    sc = {m: np.mean((tr[m] - tr.res) ** 2) for m in PICK + ['p_kalshi']}
    so = {m: np.mean((te[m] - te.res) ** 2) for m in PICK + ['p_kalshi']} if len(te) else {}
    w = min(PICK, key=sc.get); win[(b, t)] = w
    rank_so = sorted(so, key=so.get) if so else []
    rows.append(dict(price=f'{b}-{b+9}c', time=t, winner=N[w], MarAug_vs_kalshi_x1000=round(1000 * (sc[w] - sc['p_kalshi']), 3),
                     SepOct_vs_kalshi_x1000=round(1000 * (so[w] - so['p_kalshi']), 3) if so else np.nan,
                     SepOct_rank_of_5=(rank_so.index(w) + 1) if so else np.nan, SepOct_best=N[rank_so[0]] if so else '', snaps_SepOct=len(te)))
W = pd.DataFrame(rows)
print(W.to_string(index=False))
print('winner still beats Kalshi in Sep-Oct:', (W.SepOct_vs_kalshi_x1000 < 0).sum(), 'of', W.SepOct_vs_kalshi_x1000.notna().sum(),
      '| winner still #1 of 5:', (W.SepOct_rank_of_5 == 1).sum(), '| beat Kalshi in Mar-Aug:', (W.MarAug_vs_kalshi_x1000 < 0).sum())
fee = lambda p: .07 * p * (1 - p)
A['pw'] = [A.at[i, win[(b, t)]] for i, b, t in zip(A.index, A.band, A.time)]
res = []
for G in [1, 2, 3, 5]:
    for side in ['YES', 'NO']:
        px = A.ask if side == 'YES' else 1 - A.bid; wn = A.res if side == 'YES' else 1 - A.res; pr = A.pw if side == 'YES' else 1 - A.pw
        T = A.assign(px=px, win=wn, g=100 * (pr - px - fee(px)))
        T = T[(T.g >= G) & (T.px > .02) & (T.px < .99)].sort_values('min_left', ascending=False).drop_duplicates('ticker')
        T['pnl'] = T.win - T.px - fee(T.px)
        for per, h in T.groupby('per'):
            mo = h.groupby('month').pnl.mean()
            res.append(dict(gap=G, side=side, per=per, trades=len(h), win_vs_price=f'{100*h.win.mean():.1f} vs {100*h.px.mean():.1f}', after_fee_c=round(100 * h.pnl.mean(), 2),
                            t=round(h.pnl.mean() / (h.pnl.std() / np.sqrt(len(h))), 2), months=f'{(mo>0).sum()}/{len(mo)}'))
R = pd.DataFrame(res); print(R.to_string(index=False))
with pd.ExcelWriter('work/cell_winner_model.xlsx') as w:
    W.to_excel(w, sheet_name='Winner_per_cell', index=False); R.to_excel(w, sheet_name='Trade_winner_gap', index=False)
