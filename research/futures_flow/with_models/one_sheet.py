"""One sheet: Product, Period, Price bucket, Min left, Side, Flow (All/Against/Agree/Neutral), n, avg price, win %, avg edge, median edge (per-day), avg-median.
Agree = 3-min flow z > +0.5 for YES / < -0.5 for NO; Against = opposite; Neutral = |z| <= 0.5 or no flow (CME closed/thin); All = everything."""
import numpy as np, pandas as pd
W = '/mnt/project-files/strategy/futures_flow/futures_flow_edge_by_price_minute_side_flow.xlsx'
FEE = lambda p: np.ceil(7 * p * (1 - p))
out = []
for prod, label in [('15m', '15-min'), ('hourly', 'Hourly')]:
    d = pd.read_parquet(f'work/merged_{prod}.parquet', columns=['snap_ts', 'ask', 'bid', 'res', 'min_left', 'period', 'nvz_comb_3'])
    d = d[(d.ask > 0) & (d.ask < 1) & (d.bid > 0) & (d.bid < 1)]
    d['day'] = (d.snap_ts // 86400).astype(int)
    for side in ('YES', 'NO'):
        x = d[['day', 'min_left', 'period', 'nvz_comb_3']].copy()
        x['price'] = 100 * d.ask if side == 'YES' else 100 * (1 - d.bid)
        x['won'] = d.res if side == 'YES' else 1 - d.res
        x['edge'] = 100 * x.won - x.price - FEE(x.price / 100)
        x['band'] = (np.clip(x.price, 0, 99.99) // 10 * 10).astype(int)
        z = x.nvz_comb_3.to_numpy() * (1 if side == 'YES' else -1)
        x['flow'] = np.where(np.isnan(z) | (np.abs(z) <= 0.5), 'Neutral', np.where(z > 0, 'Agree', 'Against'))
        x['Product'] = label; x['Side'] = side
        out.append(x)
X = pd.concat(out, ignore_index=True)
X2 = pd.concat([X.assign(flow='All'), X], ignore_index=True)
dm = X2.groupby(['Product', 'period', 'band', 'min_left', 'Side', 'flow', 'day']).edge.mean().rename('dm').reset_index()
med = dm.groupby(['Product', 'period', 'band', 'min_left', 'Side', 'flow']).dm.median().rename('Median edge')
g = X2.groupby(['Product', 'period', 'band', 'min_left', 'Side', 'flow']).agg(n=('edge', 'size'), px=('price', 'mean'), win=('won', 'mean'), edge=('edge', 'mean')).join(med).reset_index()
g['win'] = 100 * g.win
g['Price bucket'] = g.band.astype(str) + '-' + (g.band + 9).astype(str) + 'c'
g['Avg edge - median edge'] = g.edge - g['Median edge']
g = g.rename(columns={'period': 'Period', 'min_left': 'Min left', 'flow': 'Flow', 'n': 'n trades', 'px': 'Avg price', 'win': 'Win %', 'edge': 'Avg edge'})
g['Flow'] = pd.Categorical(g.Flow, ['All', 'Against', 'Agree', 'Neutral'], ordered=True)
g['Period'] = pd.Categorical(g.Period, ['Dec-Aug', 'Sep-Oct'], ordered=True); g['Product'] = pd.Categorical(g.Product, ['15-min', 'Hourly'], ordered=True)
g = g.sort_values(['Product', 'Period', 'band', 'Min left', 'Side', 'Flow'], ascending=[True, True, True, False, False, True])
cols = ['Product', 'Period', 'Price bucket', 'Min left', 'Side', 'Flow', 'n trades', 'Avg price', 'Win %', 'Avg edge', 'Median edge', 'Avg edge - median edge']
g = g[cols]
for c in ['Avg price', 'Win %', 'Avg edge', 'Median edge', 'Avg edge - median edge']: g[c] = g[c].round(2)
readme = pd.DataFrame({'note': ['One row per price bucket x minutes left x side x flow x period. Every Kalshi snapshot, bought at the ask. Edge = win % - price - Kalshi fee, cents per contract.',
 'Flow: Agree = last-3-minute CME futures flow z > +0.5 for YES / < -0.5 for NO; Against = opposite; Neutral = |z| <= 0.5 or CME closed/thin (no flow reading); All = every snapshot.',
 'Median edge = median over days of that cell\'s daily average edge (a per-trade median would just be a win or a loss). Last column = avg edge - median edge.',
 'Periods: Dec-Aug (Dec 10 2025 - Aug 31 2026), Sep-Oct (Sep 1 - Oct 26 2026). Filter with the header dropdowns.']})
with pd.ExcelWriter(W, engine='openpyxl') as xw:
    g.to_excel(xw, sheet_name='Edge table', index=False); readme.to_excel(xw, sheet_name='README', index=False)
    ws = xw.sheets['Edge table']; ws.auto_filter.ref = ws.dimensions; ws.freeze_panes = 'G2'
print(len(g)); print(g.head(8).to_string()); print(g[(g.Product == '15-min') & (g.Period == 'Sep-Oct') & (g['Price bucket'] == '70-79c') & (g['Min left'] == 9)].to_string())
