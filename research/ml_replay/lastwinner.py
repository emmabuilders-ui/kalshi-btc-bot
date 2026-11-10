# For each 15-min contract, pick the model that was best on the PREVIOUS contract (settled at this contract's open):
#  'brier' = lowest squared error over that contract's snapshots; 'pnl' = most after-fee profit from its gap trades there.
# Trade the picked model's gap on this contract, first qualifying minute, one trade per contract. Fee 0.07 p(1-p).
import numpy as np, pandas as pd
src = open('monthly.py').read().split("months = sorted")[0].replace("'min_left','res','mid'", "'min_left','res','mid','ask','bid'")
exec(src)
PICK = ['p_roll', 'p_direct', 'p_corr', 'p_regime']
A = A.dropna(subset=['ask', 'bid'] + PICK).copy()
A['per'] = np.where(A.month >= '2026-09', 'Sep-Oct', 'Mar-Aug')
A['open'] = (A.snap_ts // 900 * 900).astype(int)
fee = lambda p: .07 * p * (1 - p)
# previous contract's score per model
se = A.assign(**{m: (A[m] - A.res) ** 2 for m in PICK}).groupby('open')[PICK].mean()
best_b = se.idxmin(axis=1)
def pnl_score(G):
    sc = {}
    for m in PICK:
        tot = 0
        for side in ('YES', 'NO'):
            px = A.ask if side == 'YES' else 1 - A.bid; pr = A[m] if side == 'YES' else 1 - A[m]; wn = A.res if side == 'YES' else 1 - A.res
            T = A.assign(px=px, win=wn)[(100 * (pr - px - fee(px)) >= G) & (px > .02) & (px < .99)]
            T = T.sort_values('min_left', ascending=False).drop_duplicates('ticker')
            tot = tot + (T.win - T.px - fee(T.px)).groupby(T.open).sum().reindex(se.index, fill_value=0)
        sc[m] = tot
    S = pd.DataFrame(sc); b = S.idxmax(axis=1); b[S.max(axis=1) <= 0] = None; return b   # no profitable model -> skip
out = []
for G in (1, 2, 3, 5):
    for how in ('brier', 'pnl'):
        best = best_b if how == 'brier' else pnl_score(G)
        prev = best.shift(1).reindex(A.open.values)                          # previous contract (900 s earlier)
        ok = (se.index.to_series().diff() == 900).reindex(A.open.values).values  # previous contract must exist
        B = A[ok & prev.notna().values].copy(); B['pm'] = prev[ok & prev.notna().values].values
        B['pw'] = [B.at[i, m] for i, m in zip(B.index, B.pm)]
        for side in ('YES', 'NO'):
            px = B.ask if side == 'YES' else 1 - B.bid; pr = B.pw if side == 'YES' else 1 - B.pw; wn = B.res if side == 'YES' else 1 - B.res
            T = B.assign(px=px, win=wn)[(100 * (pr - px - fee(px)) >= G) & (px > .02) & (px < .99)]
            T = T.sort_values('min_left', ascending=False).drop_duplicates('ticker'); T['pnl'] = T.win - T.px - fee(T.px)
            for fav, F in (('all prices', T), ('70-99c', T[T.px >= .70])):
                for per, h in F.groupby('per'):
                    mo = h.groupby('month').pnl.mean()
                    out.append(dict(pick=how, G=G, side=side, px=fav, per=per, n=len(h), wvp=f'{100*h.win.mean():.1f} vs {100*h.px.mean():.1f}',
                                    edge=round(100 * h.pnl.mean(), 2), t=round(h.pnl.mean() / (h.pnl.std() / np.sqrt(len(h))), 2), months=f'{(mo>0).sum()}/{len(mo)}'))
    print('done', G, flush=True)
# reference: each single model always, same G
for G in (1, 2, 3, 5):
    for m in PICK:
        for side in ('YES', 'NO'):
            px = A.ask if side == 'YES' else 1 - A.bid; pr = A[m] if side == 'YES' else 1 - A[m]; wn = A.res if side == 'YES' else 1 - A.res
            T = A.assign(px=px, win=wn)[(100 * (pr - px - fee(px)) >= G) & (px > .02) & (px < .99)]
            T = T.sort_values('min_left', ascending=False).drop_duplicates('ticker'); T['pnl'] = T.win - T.px - fee(T.px)
            for per, h in T.groupby('per'):
                mo = h.groupby('month').pnl.mean()
                out.append(dict(pick='always ' + m, G=G, side=side, px='all prices', per=per, n=len(h), wvp=f'{100*h.win.mean():.1f} vs {100*h.px.mean():.1f}',
                                edge=round(100 * h.pnl.mean(), 2), t=round(h.pnl.mean() / (h.pnl.std() / np.sqrt(len(h))), 2), months=f'{(mo>0).sum()}/{len(mo)}'))
O = pd.DataFrame(out); O.to_pickle('work/lastwinner.pkl')
print('brier pick share', best_b.value_counts(normalize=True).round(3).to_dict())
print('same pick as previous contract', round((best_b == best_b.shift(1)).mean(), 3))
W = O.pivot_table(index=['pick', 'G', 'side', 'px'], columns='per', values=['n', 'wvp', 'edge', 't', 'months'], aggfunc='first')
W.columns = [f'{b[:3]} {a}' for a, b in W.columns]
W = W[[f'{p} {x}' for p in ['Mar', 'Sep'] for x in ['n', 'wvp', 'edge', 't', 'months']]].reset_index()
W.to_excel('/mnt/project-files/strategy/ml_replay/best_model_last_contract.xlsx', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); print(W.to_string(index=False))
