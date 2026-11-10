# Sep-Oct only (every model out of sample there). Each contract trades the gap of the model that was best on the
# previous contract (by Brier over its snapshots; if Kalshi itself was best -> no trade). 18 models + Kalshi.
# Controls: each model always; a random model each contract. One trade per contract, fee 0.07 p(1-p).
import numpy as np, pandas as pd
src = open('monthly.py').read().split("months = sorted")[0].replace("'min_left','res','mid'", "'min_left','res','mid','ask','bid'")
exec(src)
A = A[A.month >= '2026-09'].dropna(subset=['ask', 'bid'] + M).copy()
P = [m for m in M if m != 'p_kalshi']
A['open'] = (A.snap_ts // 900 * 900).astype(int)
fee = lambda p: .07 * p * (1 - p)
se = A.assign(**{m: (A[m] - A.res) ** 2 for m in M}).groupby('open')[M].mean()
consec = se.index.to_series().diff().eq(900)
best = se.idxmin(axis=1)
rng = np.random.default_rng(7)
def trades(pm, G, label):
    B = A.assign(pm=pm.reindex(A.open.values).values).dropna(subset=['pm'])
    B = B[B.pm != 'p_kalshi']
    B['pw'] = B.to_numpy()[np.arange(len(B)), [B.columns.get_loc(m) for m in B.pm]].astype(float)
    res = []
    for side in ('YES', 'NO'):
        px = B.ask if side == 'YES' else 1 - B.bid; pr = B.pw if side == 'YES' else 1 - B.pw; wn = B.res if side == 'YES' else 1 - B.res
        T = B.assign(px=px, win=wn)[(100 * (pr - px - fee(px)) >= G) & (px > .02) & (px < .99)]
        T = T.sort_values('min_left', ascending=False).drop_duplicates('ticker'); T['pnl'] = T.win - T.px - fee(T.px)
        for fav, F in (('all', T), ('70-99c', T[T.px >= .70])):
            if len(F) < 20: continue
            mo = F.groupby('month').pnl.mean()
            res.append(dict(pick=label, G=G, side=side, px=fav, n=len(F), win_vs_price=f'{100*F.win.mean():.1f} vs {100*F.px.mean():.1f}',
                            edge=round(100 * F.pnl.mean(), 2), t=round(F.pnl.mean() / (F.pnl.std() / np.sqrt(len(F))), 2), months=f'{(mo>0).sum()}/{len(mo)}'))
    return res
out = []
prev = best.shift(1).where(consec)
for G in (1, 2, 3, 5):
    out += trades(prev, G, 'last contract winner')
    out += trades(pd.Series(rng.choice(P, len(se)), index=se.index).where(consec), G, 'random model')
    for m in P: out += trades(pd.Series(m, index=se.index), G, 'always ' + N[m])
    print('done', G, flush=True)
O = pd.DataFrame(out)
O.to_excel('/mnt/project-files/strategy/ml_replay/best_model_last_contract_all_models_sep_oct.xlsx', index=False)
print('pick share (incl Kalshi):'); print((best.value_counts(normalize=True) * 100).round(1).rename(index=N).to_string())
print('same winner as previous contract', round(100 * (best == best.shift(1))[consec].mean(), 1), '%  (random 1/19 = 5.3%)')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 600)
print(O[O.pick.isin(['last contract winner', 'random model'])].to_string(index=False))
print(O[O.pick.str.startswith('always')].sort_values('edge', ascending=False).head(25).to_string(index=False))
