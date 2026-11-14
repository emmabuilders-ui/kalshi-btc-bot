"""Complete tables by price bucket: minute to settlement x YES/NO x period, flow > 1.5 SD, each minute a look.
Per cell: trades, avg price, median price, win %, avg edge after fee (c), median edge = median of per-day mean edge (days with a trade), days."""
import numpy as np, pandas as pd, sys
from analyze import fee
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5; SUF = '' if THR == 1.5 else f'_z{THR}'
def cells(prod):
    S = pd.read_parquet(f'work/merged_{prod}.parquet', columns=['ticker', 'snap_ts', 'secs_left', 'ask', 'bid', 'res', 'period', 'nvz_comb_3'])
    S = S[S.nvz_comb_3.abs() > THR].copy(); S['ml'] = (S.secs_left // 60).astype(int)
    yes = S.nvz_comb_3 > 0; S['side'] = np.where(yes, 'YES', 'NO'); S['price'] = np.where(yes, S.ask, 1 - S.bid)
    S['win'] = np.where(yes, S.res == 1, S.res == 0).astype(float); S['pnl'] = 100 * (S.win - S.price - fee(S.price))
    S['band'] = (np.floor(S.price * 10).clip(0, 9) * 10).astype(int); S['day'] = pd.to_datetime(S.snap_ts, unit='s').dt.date
    def agg(g):
        d = g.groupby('day').pnl.mean()
        return pd.Series(dict(trades=len(g), avg_price=100 * g.price.mean(), med_price=100 * g.price.median(), win_pct=100 * g.win.mean(), avg_edge_c=g.pnl.mean(), median_daily_edge_c=d.median(), days=len(d)))
    return S.groupby(['band', 'side', 'period', 'ml']).apply(agg, include_groups=False).reset_index()
out = {}
for prod in ['15m', 'hourly']:
    out[prod] = cells(prod); out[prod].to_parquet(f'work/complete_{prod}{SUF}.parquet')
with pd.ExcelWriter('/mnt/project-files/strategy/futures_flow/futures_flow_test.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as xw:
    for prod, C in out.items():
        C.round(2).to_excel(xw, sheet_name=f'complete_{prod}_by_price{SUF}', index=False)
        for band in sorted(C.band.unique()):
            c = C[C.band == band]
            w = c.pivot_table(index='ml', columns=['side', 'period'], values=['trades', 'avg_price', 'win_pct', 'avg_edge_c', 'median_daily_edge_c', 'days']).round(2)
            w = w.reorder_levels([1, 2, 0], axis=1).sort_index(axis=1, level=[0, 1], sort_remaining=False)
            w.to_excel(xw, sheet_name=f'{prod}_{band}c{SUF}')
# chat tables (markdown) per band and side: Min | D-A n | win vs price | avg | med | S-O n | win vs price | avg | med
def md(prod, band, side, maxm):
    C = out[prod]; c = C[(C.band == band) & (C.side == side)]
    A = c[c.period == 'Dec-Aug'].set_index('ml'); B = c[c.period == 'Sep-Oct'].set_index('ml')
    lines = [f'**{band}–{band+9}c, {side}**', '', '| Min | D–A n | win vs price | avg | med | S–O n | win vs price | avg | med |', '|---|---|---|---|---|---|---|---|---|']
    for m in range(1, maxm + 1):
        def cell(X):
            if m not in X.index: return ['–', '–', '–', '–']
            r = X.loc[m]; return [f'{int(r.trades)}', f'{r.win_pct:.1f} vs {r.avg_price:.1f}', f'{r.avg_edge_c:+.1f}', f'{r.median_daily_edge_c:+.1f}']
        lines.append('| ' + ' | '.join([str(m)] + cell(A) + cell(B)) + ' |')
    return '\n'.join(lines)
for prod, maxm in [('15m', 14), ('hourly', 30)]:
    for side in ['YES', 'NO']:
        open(f'work/chat_{prod}_{side}{SUF}.md', 'w').write('\n\n'.join(md(prod, b, side, maxm) for b in [50, 60, 70, 80, 90]))
print(open('work/chat_15m_YES.md').read()[:3000])
