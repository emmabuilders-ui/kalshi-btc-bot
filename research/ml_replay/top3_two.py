# Per price band x time group: pick the 3 models with the lowest Mar-Aug Brier (out-of-fold models only), then trade only when
# ALL 3 put the same side above ask + fee by at least G points. First qualifying minute per contract. Scored Mar-Aug (selection period) and Sep-Oct.
import numpy as np, pandas as pd
src = open('monthly.py').read().split("months = sorted")[0].replace("'min_left','res','mid'", "'min_left','res','mid','ask','bid'")
exec(src)
A = A.dropna(subset=['ask', 'bid']).copy()
A['per'] = np.where(A.month >= '2026-09', 'Sep-Oct', 'Mar-Aug')
A['time'] = pd.cut(A.min_left, [0, 2, 5, 9, 14], labels=['1-2', '3-5', '6-9', '10-14']).astype(str)
CAND = ['p_roll', 'p_direct', 'p_corr', 'p_regime', 'p_static', 'p_vol']   # static/vol are in-sample before Sep: excluded from picking
PICK = ['p_roll', 'p_direct', 'p_corr', 'p_regime']
tr = A[A.per == 'Mar-Aug'].dropna(subset=PICK + ['p_kalshi'])
best = {}
for (b, t), g in tr.groupby(['band', 'time']):
    sc = {m: np.mean((g[m] - g.res) ** 2) for m in PICK}; sc['p_kalshi'] = np.mean((g.p_kalshi - g.res) ** 2)
    best[(b, t)] = sorted(PICK, key=sc.get)[:3]
B = pd.DataFrame([(b, t, *v) for (b, t), v in best.items()], columns=['band', 'time', 'm1', 'm2', 'm3'])
print(B.to_string(index=False))
fee = lambda p: .07 * p * (1 - p)
A = A.dropna(subset=PICK)
yes_px, no_px = A.ask, 1 - A.bid
gy = {m: 100 * (A[m] - yes_px - fee(yes_px)) for m in PICK}; gn = {m: 100 * ((1 - A[m]) - no_px - fee(no_px)) for m in PICK}
key = list(zip(A.band, A.time))
sel = [best.get(k) for k in key]
idx = {m: i for i, m in enumerate(PICK)}
GY = np.vstack([gy[m] for m in PICK]).T; GN = np.vstack([gn[m] for m in PICK]).T
ok = np.array([s is not None for s in sel])
cols = np.array([[idx[m] for m in s] if s else [0, 0, 0] for s in sel])
r = np.arange(len(A))[:, None]
A['minY'] = np.where(ok, np.sort(GY[r, cols], 1)[:, 1], -99); A['minN'] = np.where(ok, np.sort(GN[r, cols], 1)[:, 1], -99)
rows = []
for G in [0, 1, 2, 3, 5]:
    for side, mcol, px, win in [('YES', 'minY', yes_px, A.res), ('NO', 'minN', no_px, 1 - A.res)]:
        T = A.assign(px=px, win=win, side=side)[A[mcol] >= G]
        T = T[(T.px > 0.02) & (T.px < 0.99)]
        T = T.sort_values('min_left', ascending=False).drop_duplicates('ticker')
        T['pnl'] = T.win - T.px - fee(T.px); T['pb'] = (T.px * 100 // 10 * 10).astype(int)
        for grp in [['per'], ['per', 'pb', 'time']]:
            for k, g in T.groupby(grp):
                k = k if isinstance(k, tuple) else (k,)
                mo = g.groupby('month').pnl.mean()
                rows.append(dict(gap=G, side=side, per=k[0], price=f'{k[1]}-{k[1]+9}c' if len(k) > 1 else 'all', time=k[2] if len(k) > 1 else 'all',
                                 trades=len(g), win_vs_price=f'{100*g.win.mean():.1f} vs {100*g.px.mean():.1f}', after_fee_c=round(100 * g.pnl.mean(), 2),
                                 t=round(g.pnl.mean() / (g.pnl.std() / np.sqrt(len(g))), 2) if len(g) > 2 else np.nan, months=f'{(mo>0).sum()}/{len(mo)}'))
R = pd.DataFrame(rows)
W = R.pivot_table(index=['gap', 'side', 'price', 'time'], columns='per', values=['trades', 'win_vs_price', 'after_fee_c', 't', 'months'], aggfunc='first')
W.columns = [f'{b} {a}' for a, b in W.columns]
W = W[[f'{p} {c}' for p in ['Mar-Aug', 'Sep-Oct'] for c in ['trades', 'win_vs_price', 'after_fee_c', 't', 'months']]].reset_index()
with pd.ExcelWriter('work/top3_models_two_agree.xlsx') as w:
    B.to_excel(w, sheet_name='Top3_per_cell', index=False); W.to_excel(w, sheet_name='Results', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
print(W[W.price == 'all'].to_string(index=False))
el = W[(W.price != 'all') & (W['Mar-Aug trades'] >= 50) & (W['Sep-Oct trades'] >= 30)]
pos = el[(el['Mar-Aug after_fee_c'] > 0) & (el['Sep-Oct after_fee_c'] > 0)]
print('eligible', len(el), 'positive both', len(pos)); print(pos.sort_values('Sep-Oct t', ascending=False).head(30).to_string(index=False))
