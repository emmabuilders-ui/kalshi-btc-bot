"""Every minute x 10c bucket x YES/NO x period: Kalshi as is (no model) vs each vol-input model (buy when model >= price + 2c)."""
import numpy as np, pandas as pd
MODELS = ['s100 (live input)', 'EWMA 0.97', 'ATR14 x scale', 'HAR3 frozen']
SETS = {'BTC hourly bands': ('out/stage2_hourly_probs.parquet', 'y', 'yes_ask', 'yes_bid'),
        'BTC 15-min': ('out/stage2_15m_probs.parquet', 'res', 'ask', 'bid'),
        'ETH hourly bands': ('out_ETH/stage2_ETH_probs.parquet', 'y', 'yes_ask', 'yes_bid'),
        'XRP hourly bands': ('out_XRP/stage2_XRP_probs.parquet', 'y', 'yes_ask', 'yes_bid')}
fee = lambda p: 0.07 * p * (1 - p)
def cells(df, side, pcol, model):
    g = df.groupby(['L', 'bucket', 'period'])
    out = g.agg(n=('win', 'size'), avg_price=('price', 'mean'), win=('win', 'mean'), edge=('net', 'mean'), gap_avg=('gap', 'mean'), gap_med=('gap', 'median')).reset_index()
    mp = df.groupby(['L', 'bucket', 'period', 'month']).net.mean().gt(0).groupby(['L', 'bucket', 'period']).agg(['sum', 'size']).reset_index()
    out = out.merge(mp, on=['L', 'bucket', 'period']); out['months_pos'] = out['sum'].astype(str) + '/' + out['size'].astype(str)
    out['side'] = side; out['model'] = model
    out['avg_price'] = (out.avg_price * 100).round(1); out['win'] = (out.win * 100).round(1); out['win_minus_price'] = (out.win - out.avg_price).round(1)
    out['edge'] = (out.edge * 100).round(2); out['gap_avg'] = (out.gap_avg * 100).round(1); out['gap_med'] = (out.gap_med * 100).round(1)
    return out[['side', 'model', 'L', 'bucket', 'period', 'n', 'avg_price', 'win', 'win_minus_price', 'edge', 'gap_avg', 'gap_med', 'months_pos']]
allout = {}
for name, (fn, ycol, acol, bcol) in SETS.items():
    D = pd.read_parquet(fn); D = D[np.all([np.isfinite(D[m]) for m in MODELS], 0)]
    rows = []
    for side in ('YES', 'NO'):
        base = pd.DataFrame({'L': D.L, 'period': D.period, 'month': D.month,
                             'price': D[acol] if side == 'YES' else 1 - D[bcol], 'win': D[ycol] if side == 'YES' else 1 - D[ycol]})
        base = base[(base.price >= 0.01) & (base.price <= 0.99)]
        base['bucket'] = (np.floor(base.price * 10 + 1e-9).clip(0, 9) * 10).astype(int).astype(str).str.zfill(2) + '-' + (np.floor(base.price * 10 + 1e-9).clip(0, 9) * 10 + 9).astype(int).astype(str) + 'c'
        base['net'] = base.win - base.price - fee(base.price)
        b = base.copy(); b['gap'] = np.nan; rows.append(cells(b, side, 'price', 'Kalshi price, no model'))
        for m in MODELS:
            pm = (D[m] if side == 'YES' else 1 - D[m]).loc[base.index]
            b = base.copy(); b['gap'] = pm - base.price; b = b[b.gap >= 0.02]
            rows.append(cells(b, side, 'price', m))
    R = pd.concat(rows); R['set'] = name; allout[name] = R
    print(name, len(R), 'cells')
A = pd.concat(allout.values())
A.to_parquet('out/scan_all.parquet')
# summary: cells (set, side, L, bucket) where a model is positive after fee in BOTH periods with n>=100 each and beats no-model in both
piv = A.pivot_table(index=['set', 'side', 'L', 'bucket', 'model'], columns='period', values=['n', 'edge', 'win', 'avg_price'])
piv.columns = [f'{a}_{b}' for a, b in piv.columns]; piv = piv.reset_index()
nm = piv[piv.model == 'Kalshi price, no model'][['set', 'side', 'L', 'bucket', 'edge_Dec-Aug', 'edge_Sep-Oct']].rename(columns={'edge_Dec-Aug': 'nomodel_DA', 'edge_Sep-Oct': 'nomodel_SO'})
S = piv[piv.model != 'Kalshi price, no model'].merge(nm, on=['set', 'side', 'L', 'bucket'])
S = S[(S['n_Dec-Aug'] >= 100) & (S['n_Sep-Oct'] >= 100) & (S['edge_Dec-Aug'] > 0) & (S['edge_Sep-Oct'] > 0) & (S['edge_Dec-Aug'] > S.nomodel_DA) & (S['edge_Sep-Oct'] > S.nomodel_SO)]
S = S.sort_values(['set', 'side', 'bucket', 'L', 'model'])
S.to_parquet('out/scan_pass.parquet')
print('passing cells', len(S)); print(S.groupby(['set', 'side', 'model']).size())
with pd.ExcelWriter('/mnt/project-files/strategy/har_vol/har_vol_price_time_scan.xlsx') as xw:
    pd.DataFrame({'note': ['Every minute to settlement x 10c price bucket x YES/NO x period. "Kalshi price, no model" = all band-minutes at that price (one row per band per minute). Model rows = band-minutes where that model is >= price + 2c.',
                           'n = trades (band-minutes), avg_price in cents, win = win %, win_minus_price = points, edge = mean(win - price - fee) in cents after the 0.07*p*(1-p) fee, gap_avg / gap_med = model probability minus price (points) on the trades taken, months_pos = months with positive edge.',
                           'Pass sheet: cells with n>=100 in each period, edge > 0 in both periods, and edge above the no-model row in both periods.',
                           'Hourly bands: minutes 1-30 left, spread <= 5c, 14 nearest bands (BTC) / 6-14 nearest within 2.5% (ETH, XRP). BTC 15-min: minutes 1-14 left, fresh quotes. Prices: YES = ask, NO = 1 - bid.']}).to_excel(xw, sheet_name='README', index=False)
    S.to_excel(xw, sheet_name='Pass both periods', index=False)
    for name, R in allout.items():
        R.sort_values(['side', 'model', 'bucket', 'L', 'period']).to_excel(xw, sheet_name=name[:31], index=False)
print('xlsx ok')
