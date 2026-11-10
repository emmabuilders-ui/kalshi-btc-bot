import numpy as np, pandas as pd
exec(open('either.py').read().split('rows = []')[0])
m = 'p_direct'
T = pd.read_parquet(f'work/addf_{m}.parquet').merge(X, on=['ticker', 'snap_ts'], how='left')
cz = T[T.per == 'Mar-Aug']['Cushion (z to strike, own side)'].quantile(2/3); idx = ((T.open_ts - T0)//60 - 1).astype(int)
brk = np.where(T.side == 'YES', T.brti > hi60[idx], T.brti < lo60[idx]); T = T[(T['Cushion (z to strike, own side)'] > cz) | brk]
T['band'] = (T.px*100//5*5).astype(int).astype(str) + 'c'; T['band10'] = (T.px*100//10*10).astype(int).astype(str) + 's'
out = []
for keys in [['side', 'band10', 'min_left'], ['side', 'band', 'min_left'], ['side', 'band10'], ['side', 'min_left'], ['band10', 'min_left']]:
    for k, g0 in T.groupby(keys):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        for per, g in g0.groupby('per'):
            r[f'{per} trades'] = len(g); r[f'{per} win vs price'] = f'{100*g.win.mean():.1f} vs {100*g.px.mean():.1f}'; r[f'{per} after fee c'] = round(100*g.pnl.mean(), 2)
        r['view'] = '+'.join(keys); out.append(r)
R = pd.DataFrame(out); R.to_excel('work/direct_either_by_price_time.xlsx', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
for v in ['side+band10', 'side+min_left', 'side+band10+min_left']:
    print(R[R.view == v].dropna(axis=1, how='all').to_string(index=False)); print()
